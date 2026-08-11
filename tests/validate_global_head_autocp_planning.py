#!/usr/bin/env python3
"""Standalone FlashQLA regression for head-sharding-sensitive auto-CP planning.

This validation intentionally has no model or CausalViT dependency.  It compares
one 16-head FlashQLA launch with four independent 4-head launches using identical
per-head synthetic BF16 inputs and mixed variable-length sequence boundaries.

Expected behavior:

* upstream base 6ef4858, no override:
  auto-CP chooses different segmentations and the outputs diverge;
* either revision with auto-CP disabled:
  the 16-head and 4x4-head outputs are bit-exact;
* fixed revision with FLASH_QLA_AUTO_CP_PLANNING_HEADS=16:
  auto-CP chooses the same segmentation and the outputs are bit-exact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from flash_qla import chunk_gated_delta_rule_fwd
from flash_qla.ops.gated_delta_rule.chunk.cp_context import _calc_cp_seqs
from flash_qla.utils import l2norm


LENGTHS = (988, 16128, 936, 16128)
NUM_HEADS = 16
HEADS_PER_SHARD = 4
HEAD_DIM = 128
CHUNK_SIZE = 64
SEED = 20260729


def _stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    lhs_f = lhs.float()
    rhs_f = rhs.float()
    delta = lhs_f - rhs_f
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (delta.norm() / rhs_f.norm().clamp_min(1e-12)).item(),
        "exact_fraction": (lhs == rhs).float().mean().item(),
    }


def _make_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(SEED)
    total_tokens = sum(LENGTHS)
    shape = (1, total_tokens, NUM_HEADS, HEAD_DIM)
    head_shape = (1, total_tokens, NUM_HEADS)
    return {
        "q": l2norm(torch.randn(shape, device=device, dtype=torch.bfloat16)),
        "k": l2norm(torch.randn(shape, device=device, dtype=torch.bfloat16)),
        "v": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "g": torch.nn.functional.logsigmoid(
            torch.randn(head_shape, device=device, dtype=torch.float32)
        )
        / 16,
        "beta": torch.randn(
            head_shape, device=device, dtype=torch.float32
        ).sigmoid(),
    }


def _run(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    *,
    heads_per_call: int,
    auto_cp: bool,
) -> torch.Tensor:
    outputs = []
    for start in range(0, NUM_HEADS, heads_per_call):
        head_slice = slice(start, start + heads_per_call)
        _, _, output, _, _ = chunk_gated_delta_rule_fwd(
            q=inputs["q"][..., head_slice, :].contiguous(),
            k=inputs["k"][..., head_slice, :].contiguous(),
            v=inputs["v"][..., head_slice, :].contiguous(),
            g=inputs["g"][..., head_slice].contiguous(),
            beta=inputs["beta"][..., head_slice].contiguous(),
            scale=HEAD_DIM**-0.5,
            initial_state=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
            output_h=False,
            auto_cp=auto_cp,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=-2)


def _planning(cu_seqlens: torch.Tensor, heads: int) -> dict[str, object]:
    use_cp, cp_cu, *_ = _calc_cp_seqs(cu_seqlens, CHUNK_SIZE, heads)
    return {
        "use_cp": bool(use_cp),
        "segment_count": int(cp_cu.numel() - 1) if cp_cu is not None else None,
        "boundaries": cp_cu.tolist() if cp_cu is not None else None,
    }


def _assert_exact(stats: dict[str, float], label: str) -> None:
    assert stats["max_abs"] == 0.0, (label, stats)
    assert stats["relative_l2"] == 0.0, (label, stats)
    assert stats["exact_fraction"] == 1.0, (label, stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect",
        required=True,
        choices=("upstream-divergence", "fixed-exact"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires an SM90 CUDA GPU")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(f"Expected SM90, found compute capability {capability}")

    lengths = torch.tensor(LENGTHS, device=device, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        lengths.cumsum(0, dtype=torch.int32), (1, 0)
    )
    planning = {
        "h16": _planning(cu_seqlens, NUM_HEADS),
        "h4": _planning(cu_seqlens, HEADS_PER_SHARD),
    }

    inputs = _make_inputs(device)
    with torch.inference_mode():
        auto_cp_h16 = _run(
            inputs, cu_seqlens, heads_per_call=NUM_HEADS, auto_cp=True
        )
        auto_cp_4x4 = _run(
            inputs, cu_seqlens, heads_per_call=HEADS_PER_SHARD, auto_cp=True
        )
        no_auto_cp_h16 = _run(
            inputs, cu_seqlens, heads_per_call=NUM_HEADS, auto_cp=False
        )
        no_auto_cp_4x4 = _run(
            inputs, cu_seqlens, heads_per_call=HEADS_PER_SHARD, auto_cp=False
        )

    auto_cp_stats = _stats(auto_cp_4x4, auto_cp_h16)
    no_auto_cp_stats = _stats(no_auto_cp_4x4, no_auto_cp_h16)
    _assert_exact(no_auto_cp_stats, "auto_cp_disabled")

    if args.expect == "upstream-divergence":
        assert planning["h16"]["boundaries"] != planning["h4"]["boundaries"], planning
        assert auto_cp_stats["max_abs"] > 0.0, auto_cp_stats
        assert auto_cp_stats["relative_l2"] > 0.0, auto_cp_stats
        assert auto_cp_stats["exact_fraction"] < 1.0, auto_cp_stats
    else:
        assert os.environ.get("FLASH_QLA_AUTO_CP_PLANNING_HEADS") == "16"
        assert planning["h16"]["boundaries"] == planning["h4"]["boundaries"], planning
        _assert_exact(auto_cp_stats, "fixed_auto_cp")

    result = {
        "expectation": args.expect,
        "flash_qla_file": __import__("flash_qla").__file__,
        "planning_heads_env": os.environ.get(
            "FLASH_QLA_AUTO_CP_PLANNING_HEADS"
        ),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "seed": SEED,
        "lengths": list(LENGTHS),
        "shape": [1, sum(LENGTHS), NUM_HEADS, HEAD_DIM],
        "planning": planning,
        "auto_cp_h16_vs_4x4": auto_cp_stats,
        "auto_cp_disabled_h16_vs_4x4": no_auto_cp_stats,
        "status": "PASS",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
