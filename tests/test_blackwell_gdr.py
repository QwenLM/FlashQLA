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
    assert torch.isfinite(ref).all(), f"{name}: non-finite reference"
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
    reason="FlashQLA Blackwell contract test requires an sm100+ CUDA GPU",
)
def test_blackwell_backend_and_auto_cp_contract():
    import tilelang
    from flash_qla import chunk_gated_delta_rule_fwd
    import flash_qla.ops.gated_delta_rule.chunk as chunk_backend
    from flash_qla.ops.gated_delta_rule.chunk.blackwell import (
        correct_initial_states,
        get_warmup_chunks,
    )
    from flash_qla.utils import l2norm

    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim = 1, 128, 2, 6, 128
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

    assert tilelang.contrib.nvcc.get_target_compute_version() == "10.0"
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
            chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, auto_cp=False)
    finally:
        chunk_backend._chunk_backend = original_backend

    cp_cu_seqlens = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
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
        chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, auto_cp=True)


def _make_inputs(
    *,
    seed: int,
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    cu_seqlens_values: tuple[int, ...] | None,
    use_initial_state: bool,
):
    from flash_qla.utils import l2norm

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
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
    cu_seqlens = None
    real_batch_size = batch_size
    if cu_seqlens_values is not None:
        assert batch_size == 1
        assert cu_seqlens_values[0] == 0 and cu_seqlens_values[-1] == num_tokens
        cu_seqlens = torch.tensor(cu_seqlens_values, device=device, dtype=torch.int32)
        real_batch_size = len(cu_seqlens_values) - 1
    h0 = None
    dht = None
    if use_initial_state:
        h0 = torch.randn(
            real_batch_size,
            num_v_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        dht = torch.randn_like(h0) / 8
    return q, k, v, g, beta, do, h0, dht, cu_seqlens


def _run_blackwell_precision_case(
    *,
    name: str,
    seed: int,
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    cu_seqlens_values: tuple[int, ...] | None = None,
    use_initial_state: bool = False,
    repeat: int = 3,
):
    from flash_qla import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
    from ref_gdr import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_ref
    from ref_gdr import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_ref

    head_dim = 128
    scale = head_dim**-0.5
    q, k, v, g, beta, do, h0, dht, cu_seqlens = _make_inputs(
        seed=seed,
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        cu_seqlens_values=cu_seqlens_values,
        use_initial_state=use_initial_state,
    )

    g_ref, o_ref, A_ref, h_ref, final_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
    )
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.float(),
        k.float(),
        v.float(),
        g_ref,
        beta.float(),
        A_ref.float(),
        scale,
        h0,
        do.float(),
        dht,
        cu_seqlens,
    )

    for run_idx in range(repeat):
        g_qla, A_qla, o_qla, h_qla, final_qla = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=h0,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            output_h=True,
            auto_cp=None,
        )
        _assert_close(f"{name}.run{run_idx}.g", g_ref, g_qla.float(), ratio=0.002)
        _assert_close(f"{name}.run{run_idx}.o", o_ref, o_qla.float(), ratio=0.02)
        _assert_close(f"{name}.run{run_idx}.h", h_ref, h_qla.float(), ratio=0.03)
        _assert_close(
            f"{name}.run{run_idx}.final_state", final_ref, final_qla.float(), ratio=0.03
        )

        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = chunk_gated_delta_rule_bwd(
            q,
            k,
            v,
            g_qla,
            beta,
            A_qla,
            do,
            dht,
            scale,
            h0,
            cu_seqlens,
        )
        _assert_close(f"{name}.run{run_idx}.dq", dq_ref, dq_qla.float(), ratio=0.03)
        _assert_close(f"{name}.run{run_idx}.dk", dk_ref, dk_qla.float(), ratio=0.03)
        _assert_close(f"{name}.run{run_idx}.dv", dv_ref, dv_qla.float(), ratio=0.03)
        _assert_close(f"{name}.run{run_idx}.dbeta", db_ref, db_qla.float(), ratio=0.03)
        _assert_close(f"{name}.run{run_idx}.dg", dg_ref, dg_qla.float(), ratio=0.03)
        if dht is not None:
            _assert_close(
                f"{name}.run{run_idx}.dh0", dh0_ref, dh0_qla.float(), ratio=0.03
            )


@pytest.mark.skipif(
    _requires_blackwell(),
    reason="FlashQLA Blackwell precision test requires an sm100+ CUDA GPU",
)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(
                name="full_chunks",
                seed=42,
                batch_size=1,
                num_tokens=512,
                num_k_heads=4,
                num_v_heads=12,
            ),
            id="full_chunks",
        ),
        pytest.param(
            dict(
                name="partial_initial_state",
                seed=43,
                batch_size=1,
                num_tokens=575,
                num_k_heads=2,
                num_v_heads=6,
                use_initial_state=True,
            ),
            id="partial_initial_state",
        ),
        pytest.param(
            dict(
                name="varlen_partial",
                seed=44,
                batch_size=1,
                num_tokens=575,
                num_k_heads=2,
                num_v_heads=6,
                cu_seqlens_values=(0, 63, 130, 575),
            ),
            id="varlen_partial",
        ),
    ],
)
def test_blackwell_gdr_fused_path_matches_reference(case):
    _run_blackwell_precision_case(**case)
