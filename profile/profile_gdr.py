# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""
Kernel profiling tool for Gated Delta Rule.

Compares FLA vs QLA kernel timings (fwd and bwd) and prints speedup.
No correctness checks — use ``pytest tests/test_gdr_unit.py`` for that.

Usage examples::

    python utils/profile_gdr.py --set develop
    python utils/profile_gdr.py --set profile --skip-bwd
    python utils/profile_gdr.py --set develop --no-cp
    python utils/profile_gdr.py --set develop --cp-cache
    python utils/profile_gdr.py --set develop --state-v-first
    python utils/profile_gdr.py --set develop --no-h0
    python utils/profile_gdr.py --set develop --data-dtype float16
"""

import argparse
import math
import os
import sys

import torch
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))  # for ref_gdr

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_fla,
)
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_fla,
)

from flash_qla import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_qla
from flash_qla import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_qla
from flash_qla.utils import l2norm, pack, profile


# ---------------------------------------------------------------------------
# Single profiling function
# ---------------------------------------------------------------------------

def profile_gated_delta_rule(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int = 128,
    head_dim_v: int = 128,
    varlen: bool = False,
    cu_seqlens: list[int] | None = None,
    use_h0: bool = True,
    chunk_size: int = 64,
    data_dtype: str = "bfloat16",
    device: torch.device = "cuda",
    random_seed: int = 42,
    auto_cp: bool = True,
    swa_ratio: float = 0.75,
    skip_bwd: bool = False,
    state_v_first: bool = False,
    enable_fwd_cp_cache: bool = False,
    fla_bwd_ok: bool = True,
):
    data_dtype = getattr(torch, data_dtype)
    torch.manual_seed(random_seed)
    scale = head_dim_k ** (-0.5)

    # ---- generate inputs ----
    q = l2norm(torch.randn(
        (batch_size, num_tokens, num_k_heads, head_dim_k),
        device=device, dtype=data_dtype,
    ))
    k = l2norm(torch.randn(
        (batch_size, num_tokens, num_k_heads, head_dim_k),
        device=device, dtype=data_dtype,
    ))
    v = torch.randn(
        (batch_size, num_tokens, num_v_heads, head_dim_v),
        device=device, dtype=data_dtype,
    )
    g = (
        torch.nn.functional.logsigmoid(torch.randn(
            (batch_size, num_tokens, num_v_heads),
            device=device, dtype=torch.float32,
        )) / 16
    )
    beta = torch.randn(
        (batch_size, num_tokens, num_v_heads),
        device=device, dtype=torch.float32,
    ).sigmoid()
    h0 = (
        torch.randn(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            device=device, dtype=torch.float32,
        )
        if use_h0 else None
    )
    do = torch.randn_like(v)
    dht = (
        torch.randn(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            device=device, dtype=torch.float32,
        ) / 8
        if use_h0 else None
    )

    # QLA state layout (may transpose for state_v_first)
    h0_qla = (
        h0.transpose(-1, -2).contiguous()
        if state_v_first and h0 is not None else h0
    )
    dht_qla = (
        dht.transpose(-1, -2).contiguous()
        if state_v_first and dht is not None else dht
    )

    # ---- SWA mask ----
    swa_mask = torch.zeros((num_v_heads,), dtype=torch.bool, device=device)
    swa_mask[:math.ceil(swa_ratio * num_v_heads)] = 1
    swa_mask = swa_mask[torch.randperm(num_v_heads, device=device)]
    g[:, :, ~swa_mask] = 0.0

    # ---- varlen ----
    if varlen:
        if cu_seqlens is None:
            cu_seqlens_t = torch.randint(
                1, num_tokens, (batch_size,), device=device, dtype=torch.int32,
            )
            cu_seqlens_t = torch.nn.functional.pad(
                torch.cumsum(cu_seqlens_t, dim=-1), (1, 0),
            )
            q = pack(q, cu_seqlens_t)
            k = pack(k, cu_seqlens_t)
            v = pack(v, cu_seqlens_t)
            g = pack(g, cu_seqlens_t)
            beta = pack(beta, cu_seqlens_t)
            do = pack(do, cu_seqlens_t)
            cu_seqlens = cu_seqlens_t
        else:
            assert batch_size == 1
            assert cu_seqlens[0] == 0
            assert cu_seqlens[-1] == num_tokens
            cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
            if use_h0:
                real_batch_size = cu_seqlens.shape[0] - 1
                h0 = torch.randn(
                    (real_batch_size, num_v_heads, head_dim_k, head_dim_v),
                    device=device, dtype=torch.float32,
                )
                dht = torch.randn(
                    (real_batch_size, num_v_heads, head_dim_k, head_dim_v),
                    device=device, dtype=torch.float32,
                ) / 8
                h0_qla = (
                    h0.transpose(-1, -2).contiguous()
                    if state_v_first else h0
                )
                dht_qla = (
                    dht.transpose(-1, -2).contiguous()
                    if state_v_first else dht
                )
    else:
        cu_seqlens = None

    print(
        f"Shape: B={batch_size} Hk={num_k_heads} Hv={num_v_heads} "
        f"T={num_tokens} VarLen={varlen} StateVFirst={state_v_first} "
        f"AutoCP={auto_cp} CPCache={enable_fwd_cp_cache}"
    )

    # ==================================================================
    # Forward profiling
    # ==================================================================

    # FLA fwd
    prof_fla = profile(
        chunk_gated_delta_rule_fwd_fla,
        [q, k, v, g, beta, scale, h0, True, cu_seqlens],
    )

    # QLA fwd
    prof_qla = profile(
        chunk_gated_delta_rule_fwd_qla,
        [q, k, v, g, beta, scale, h0_qla, cu_seqlens,
         True, False, auto_cp, state_v_first, enable_fwd_cp_cache],
    )

    nan = float("nan")

    def _get(prof, name, occurrence=0):
        key = f"{name}#{occurrence}" if occurrence > 0 else name
        return prof.get(key, nan)

    # Per-kernel breakdown (only when profiler captured events)
    has_fla_kernels = len(prof_fla) > 1  # more than just "total"
    has_qla_kernels = len(prof_qla) > 1
    if has_fla_kernels and has_qla_kernels:
        result_fla = {
            "[fwd] csum": _get(prof_fla, "chunk_local_cumsum_scalar_kernel"),
            "[fwd] solve": _get(prof_fla, "chunk_gated_delta_rule_fwd_kkt_solve_kernel"),
            "[fwd] wu": _get(prof_fla, "recompute_w_u_fwd_kernel"),
            "[fwd] gdr": _get(prof_fla, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
            "[fwd] o": _get(prof_fla, "chunk_fwd_kernel_o"),
        }
        result_qla = {
            "[fwd] csum": _get(prof_qla, "tilelang_chunk_local_cumsum_kernel_kernel"),
            "[fwd] solve": _get(prof_qla, "tilelang_kkt_solve_kernel_kernel"),
            "[fwd] gdr": _get(prof_qla, "tilelang_fused_chunk_gdr_fwd_kernel_kernel"),
        }

        # CP kernels (only present when auto_cp is on and sequence is long enough)
        if "tilelang_get_warmup_chunks_kernel_kernel" in prof_qla:
            result_fla["[fwd] cp-w"] = None
            result_fla["[fwd] cp-h"] = None
            result_fla["[fwd] cp-c"] = None
            result_qla["[fwd] cp-w"] = _get(prof_qla, "tilelang_get_warmup_chunks_kernel_kernel")
            result_qla["[fwd] cp-h"] = _get(prof_qla, "tilelang_prepare_h_kernel_kernel")
            result_qla["[fwd] cp-c"] = _get(prof_qla, "tilelang_correct_h0_kernel_kernel")

        result_fla["total"] = prof_fla["total"]
        result_qla["total"] = prof_qla["total"]

        results = {"fla": result_fla, "flash_qla": result_qla}
        df = pd.DataFrame(results)
        print(df.round(3))
    else:
        print(f"  fla total:       {prof_fla['total']:.3f} ms")
        print(f"  flash_qla total: {prof_qla['total']:.3f} ms")

    speedup = prof_fla["total"] / prof_qla["total"]
    print(f"Fwd speedup: {speedup:.2f}x")

    if skip_bwd:
        return

    # ==================================================================
    # Run a single QLA fwd to get g_qla, A_qla (and cp_cache if needed)
    # ==================================================================
    g_fla, _, A_fla, _, _, _ = chunk_gated_delta_rule_fwd_fla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, initial_state=h0,
        output_final_state=True, cu_seqlens=cu_seqlens,
    )
    g_qla, A_qla, _, _, _, cp_cache = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=auto_cp, state_v_first=state_v_first,
        enable_fwd_cp_cache=enable_fwd_cp_cache,
    )

    # ==================================================================
    # Backward profiling
    # ==================================================================

    # FLA bwd (may crash on sm_100 — skip if probe failed at startup)
    if fla_bwd_ok:
        prof_fla = profile(
            chunk_gated_delta_rule_bwd_fla,
            [q, k, v, g_fla, beta, A_fla, scale, h0, do, dht, cu_seqlens],
        )
    else:
        prof_fla = None

    # QLA bwd
    bwd_args = [
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, auto_cp,
    ]
    bwd_kwargs = {}
    if enable_fwd_cp_cache and cp_cache is not None:
        bwd_kwargs["cp_cache"] = cp_cache
    prof_qla = profile(
        lambda *a: chunk_gated_delta_rule_bwd_qla(*a, **bwd_kwargs),
        bwd_args,
    )

    # Per-kernel breakdown for bwd
    has_fla_kernels = prof_fla is not None and len(prof_fla) > 1
    has_qla_kernels = len(prof_qla) > 1

    # Build QLA result dict (always, when kernel events are available)
    result_qla = None
    if has_qla_kernels:
        result_qla = {
            "[bwd] csum": _get(prof_qla, "tilelang_chunk_local_cumsum_kernel_kernel"),
            "[bwd] recom": _get(
                prof_qla, "tilelang_prepare_h_kernel_kernel", 1
            ) if "tilelang_prepare_h_kernel_kernel#1" in prof_qla else _get(
                prof_qla, "tilelang_prepare_h_kernel_kernel"
            ),
            "[bwd] gdr": _get(prof_qla, "tilelang_fused_chunk_gdr_bwd_kernel_kernel"),
            "[bwd] cp-w": _get(prof_qla, "tilelang_get_warmup_chunks_bidi_kernel_kernel"),
            "[bwd] cp-h": (
                _get(prof_qla, "tilelang_prepare_h_kernel_kernel")
                if "tilelang_prepare_h_kernel_kernel#1" in prof_qla
                else nan
            ),
            "[bwd] cp-c": _get(prof_qla, "tilelang_correct_h0_kernel_kernel"),
            "[bwd] cp-c-dht": _get(prof_qla, "tilelang_correct_h0_kernel_kernel", 1),
            "[bwd] cp-dh": _get(prof_qla, "tilelang_prepare_dh_kernel_kernel"),
        }
        if num_k_heads < num_v_heads:
            result_qla["[bwd] reduc"] = (
                _get(prof_qla, "tilelang_group_reduce_vector_kernel_kernel") * 2
            )
        result_qla["total"] = prof_qla["total"]

    if has_fla_kernels and result_qla is not None:
        result_fla = {
            "[bwd] csum": _get(prof_fla, "chunk_local_cumsum_scalar_kernel"),
            "[bwd] recom": (
                _get(prof_fla, "recompute_w_u_fwd_kernel")
                + _get(prof_fla, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
            ),
            "[bwd] dv": _get(prof_fla, "chunk_bwd_kernel_dv_local"),
            "[bwd] gdr": _get(prof_fla, "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"),
            "[bwd] dqkwg": _get(prof_fla, "kernel_kernel"),
            "[bwd] wy": _get(prof_fla, "prepare_wy_repr_bwd_kernel"),
            "[bwd] cp-w": nan,
            "[bwd] cp-h": nan,
            "[bwd] cp-c": nan,
            "[bwd] cp-dh": nan,
        }
        if num_k_heads < num_v_heads:
            result_fla["[bwd] reduc"] = _get(prof_fla, "compress_heads_kernel")
        result_fla["total"] = prof_fla["total"]

        results = {"fla": result_fla, "flash_qla": result_qla}
        df = pd.DataFrame(results)
        print(df.round(3))
        speedup = prof_fla["total"] / prof_qla["total"]
        print(f"Bwd speedup: {speedup:.2f}x")
    elif result_qla is not None:
        df = pd.DataFrame({"flash_qla": result_qla})
        print(df.round(3))
        if prof_fla is not None:
            speedup = prof_fla["total"] / prof_qla["total"]
            print(f"Bwd speedup: {speedup:.2f}x")
    elif prof_fla is not None:
        print(f"  fla total:       {prof_fla['total']:.3f} ms")
        print(f"  flash_qla total: {prof_qla['total']:.3f} ms")
        speedup = prof_fla["total"] / prof_qla["total"]
        print(f"Bwd speedup: {speedup:.2f}x")
    else:
        print(f"  flash_qla bwd total: {prof_qla['total']:.3f} ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile Gated Delta Rule kernels (FLA vs QLA)"
    )
    parser.add_argument(
        "--set", type=str, default="develop",
        help="Preset name (loads from tests/settings/{set}.csv)",
    )
    parser.add_argument(
        "--seqlen", "--num-tokens", type=int, default=16384,
        help="Sequence length (overrides CSV if set)",
    )
    parser.add_argument(
        "--nkh", "--num-k-heads", type=int, default=0,
        help="Number of K heads (0 = same as V heads)",
    )
    parser.add_argument(
        "--nvh", "--num-heads", "--num-v-heads", type=int, default=64,
        help="Number of V heads",
    )
    parser.add_argument(
        "--no-h0", action="store_true",
        help="Disable initial state",
    )
    parser.add_argument(
        "--skip-bwd", action="store_true",
        help="Only profile forward",
    )
    parser.add_argument(
        "--no-cp", "--disable-auto-cp", action="store_true",
        help="Disable auto intra-card CP",
    )
    parser.add_argument(
        "--cp-cache", action="store_true",
        help="Enable enable_fwd_cp_cache",
    )
    parser.add_argument(
        "--swa-ratio", type=float, default=0.75,
        help="Ratio of sliding-window heads",
    )
    parser.add_argument(
        "--state-v-first", action="store_true",
        help="Use [N, H, V, K] state layout instead of [N, H, K, V]",
    )
    parser.add_argument(
        "--data-dtype", type=str, default="bfloat16",
        help="Data type for inputs",
    )
    parser.add_argument(
        "--seed", "--random-seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if args.nkh <= 0:
        args.nkh = args.nvh

    # Probe FLA bwd in a subprocess — CUDA errors are fatal to the process
    _FLA_BWD_OK = True
    if not args.skip_bwd:
        import subprocess
        probe_code = (
            "import torch;"
            "from fla.ops.gated_delta_rule.chunk import "
            "chunk_gated_delta_rule_fwd as fwd, "
            "chunk_gated_delta_rule_bwd as bwd;"
            "q=torch.randn(1,64,1,128,device='cuda',dtype=torch.bfloat16);"
            "k=torch.randn_like(q);v=torch.randn(1,64,1,128,device='cuda',dtype=torch.bfloat16);"
            "g=torch.randn(1,64,1,device='cuda',dtype=torch.float32);"
            "b=torch.randn(1,64,1,device='cuda',dtype=torch.float32).sigmoid();"
            "gf,_,af,_,_,_=fwd(q,k,v,g,b,128**-0.5,None,True,None);"
            "do=torch.randn_like(v);"
            "bwd(q,k,v,gf,b,af,128**-0.5,None,do,None,None);"
            "torch.cuda.synchronize()"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            _FLA_BWD_OK = False
            print("[INFO] FLA bwd not available on this GPU — bwd will only show QLA timings")

    metadata = {
        "head_dim_k": 128,
        "head_dim_v": 128,
        "chunk_size": 64,
        "num_tokens": args.seqlen,
        "num_k_heads": args.nkh,
        "num_v_heads": args.nvh,
        "use_h0": not args.no_h0,
        "data_dtype": args.data_dtype,
        "skip_bwd": args.skip_bwd,
        "auto_cp": not args.no_cp,
        "swa_ratio": args.swa_ratio,
        "state_v_first": args.state_v_first,
        "random_seed": args.seed,
        "device": "cuda",
        "enable_fwd_cp_cache": args.cp_cache,
        "fla_bwd_ok": _FLA_BWD_OK,
    }

    preset = pd.read_csv(
        os.path.join(PROJECT_ROOT, "profile", "settings", f"{args.set}.csv")
    )
    for i, row in preset.iterrows():
        print("-" * 64)
        torch.cuda.empty_cache()
        data = row.to_dict()
        if "cu_seqlens" in data:
            data["cu_seqlens"] = list(map(int, data["cu_seqlens"].split("-")))
        metadata.update(data)
        profile_gated_delta_rule(**metadata)
    print("-" * 64)
