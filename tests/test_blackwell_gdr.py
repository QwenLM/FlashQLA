# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _add_tilelang_bundled_tvm_to_path():
    if importlib.util.find_spec("tvm") is not None:
        return
    tilelang_spec = importlib.util.find_spec("tilelang")
    if tilelang_spec is None or tilelang_spec.origin is None:
        return
    tvm_python = Path(tilelang_spec.origin).parent / "3rdparty" / "tvm" / "python"
    if tvm_python.exists():
        sys.path.append(str(tvm_python))


_add_tilelang_bundled_tvm_to_path()


def _requires_blackwell():
    return not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10


def _assert_close(name: str, ref: torch.Tensor, got: torch.Tensor, ratio: float = 0.02):
    assert torch.isfinite(got).all(), f"{name}: non-finite output"
    diff = (got - ref).abs().max().item()
    base = ref.abs().max().item()
    limit = base * ratio + 1e-4
    print(
        f"blackwell_gdr_metric name={name} "
        f"max_abs={diff:.6g} limit={limit:.6g} ref_max={base:.6g}",
        flush=True,
    )
    assert diff <= limit, (
        f"{name}: max_abs={diff:.6g} limit={limit:.6g} ref_max={base:.6g}"
    )


@pytest.mark.skipif(
    _requires_blackwell(),
    reason="FlashQLA Blackwell precision test requires an sm100+ CUDA GPU",
)
def test_blackwell_gdr_fused_path_matches_reference():
    from flash_qla import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
    import flash_qla.ops.gated_delta_rule.chunk as chunk_backend
    from flash_qla.ops.gated_delta_rule.chunk.blackwell import (
        correct_initial_states,
        get_warmup_chunks,
    )
    from flash_qla.utils import l2norm
    from ref_gdr import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_ref
    from ref_gdr import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_ref

    torch.manual_seed(42)
    batch_size, num_tokens, num_k_heads, num_v_heads = 1, 512, 4, 12
    head_dim = 128
    dtype = torch.bfloat16
    device = torch.device("cuda")

    q = l2norm(
        torch.randn(
            batch_size, num_tokens, num_k_heads, head_dim, device=device, dtype=dtype
        )
    )
    k = l2norm(
        torch.randn(
            batch_size, num_tokens, num_k_heads, head_dim, device=device, dtype=dtype
        )
    )
    v = torch.randn(
        batch_size, num_tokens, num_v_heads, head_dim, device=device, dtype=dtype
    )
    g = (
        torch.nn.functional.logsigmoid(
            torch.randn(
                batch_size, num_tokens, num_v_heads, device=device, dtype=torch.float32
            )
        )
        / 16
    )
    beta = torch.randn(
        batch_size, num_tokens, num_v_heads, device=device, dtype=torch.float32
    ).sigmoid()
    do = torch.randn_like(v)
    scale = head_dim**-0.5

    assert chunk_backend._chunk_backend == "blackwell"
    assert ".blackwell." in chunk_backend.fused_gdr_fwd.__module__
    print(
        f"blackwell_gdr_backend backend={chunk_backend._chunk_backend} "
        f"fwd_module={chunk_backend.fused_gdr_fwd.__module__}",
        flush=True,
    )

    original_backend = chunk_backend._chunk_backend
    chunk_backend._chunk_backend = "hopper"
    try:
        with pytest.raises(RuntimeError, match="SM90/Hopper TileLang target"):
            chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                auto_cp=False,
            )
    finally:
        chunk_backend._chunk_backend = original_backend

    cp_cu_seqlens = torch.tensor([0, 256, 512], device=device, dtype=torch.int32)
    ht_mask = torch.tensor([False, True], device=device)
    with pytest.raises(RuntimeError, match="auto_cp is not supported"):
        get_warmup_chunks(
            g=g,
            cu_seqlens=cp_cu_seqlens,
            ht_mask=ht_mask,
            chunk_size=64,
            threshold=-10.0,
        )
    with pytest.raises(RuntimeError, match="auto_cp is not supported"):
        correct_initial_states(
            raw_h0=torch.empty(1, num_v_heads, head_dim, head_dim, device=device),
            ht_buffer=torch.empty(2, num_v_heads, head_dim, head_dim, device=device),
            mt_buffer=torch.empty(2, num_v_heads, device=device),
            fallback_mask=torch.empty(2, num_v_heads, dtype=torch.bool, device=device),
            seq_map_r2c=torch.tensor([0, 2], device=device, dtype=torch.int32),
        )

    with pytest.raises(RuntimeError, match="auto_cp=True is not supported on SM100"):
        chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            auto_cp=True,
        )

    g_ref, o_ref, A_ref, _, _ = chunk_gated_delta_rule_fwd_ref(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        scale=scale,
    )
    g_qla, A_qla, o_qla, _, _ = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        output_final_state=True,
        output_h=False,
        auto_cp=None,
    )

    _assert_close("g", g_ref, g_qla.float(), ratio=0.002)
    _assert_close("o", o_ref, o_qla.float(), ratio=0.02)

    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, _ = chunk_gated_delta_rule_bwd_ref(
        q.float(),
        k.float(),
        v.float(),
        g_ref,
        beta.float(),
        A_ref,
        scale,
        None,
        do.float(),
        None,
        None,
    )
    dq_qla, dk_qla, dv_qla, db_qla, dg_qla, _ = chunk_gated_delta_rule_bwd(
        q,
        k,
        v,
        g_qla,
        beta,
        A_qla,
        do,
        None,
        scale,
        None,
        None,
    )

    _assert_close("dq", dq_ref, dq_qla.float(), ratio=0.03)
    _assert_close("dk", dk_ref, dk_qla.float(), ratio=0.03)
    _assert_close("dv", dv_ref, dv_qla.float(), ratio=0.03)
    _assert_close("dbeta", db_ref, db_qla.float(), ratio=0.03)
    _assert_close("dg", dg_ref, dg_qla.float(), ratio=0.03)
