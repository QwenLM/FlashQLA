# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_qla
from flash_qla import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_qla
from flash_qla.utils import l2norm, pack
import tilelang
from ref_gdr import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_ref
from ref_gdr import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_ref

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RTOL = 0.02
DETERMINISM_ITERS = 1000
HEAD_DIM_K = 128
HEAD_DIM_V = 128
CHUNK_SIZE = 32 if tilelang.contrib.nvcc.get_target_compute_version() == "12.0" else 64
REF_DTYPE = torch.float64
DATA_DTYPE = torch.bfloat16
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Shape configurations (inlined from settings/*.csv)
# ---------------------------------------------------------------------------

# (B, T, Hk, Hv, varlen, cu_seqlens | None)
CORE_CONFIGS = [
    pytest.param(1, 4096, 4, 4, False, None, id="B1-T4096-H4"),
    pytest.param(1, 4096, 2, 8, False, None, id="B1-T4096-H8G2"),
    pytest.param(1, 4096, 4, 16, False, None, id="B1-T4096-H16G4"),
    pytest.param(3, 4096, 4, 4, True, None, id="B3-T4096-H4-varlen"),
    pytest.param(1, 4096, 16, 32, True,
                 [0, 410, 841, 1135, 2126, 2512, 4096],
                 id="B1-T4096-H32G16-varlen"),
    pytest.param(1, 1000, 16, 64, True,
                 [0, 211, 985],
                 id="B1-T4096-H64G16-padding"),
]

DEVELOP_CONFIGS = [
    pytest.param(1, 32768, 4, 4, False, None, id="dev-H4"),
    pytest.param(1, 32768, 8, 8, False, None, id="dev-H8"),
    pytest.param(1, 32768, 16, 16, False, None, id="dev-H16"),
]

VARLEN_CONFIGS = [
    pytest.param(11, 33, 4, 4, True, None, id="varlen-B11-T33"),
    pytest.param(7, 4321, 4, 4, True, None, id="varlen-B7-T4321"),
    pytest.param(3, 16789, 4, 4, True, None, id="varlen-B3-T16789-vl"),
    pytest.param(5, 8192, 4, 4, True, None, id="varlen-B5-T8192-vl"),
    pytest.param(10, 1024, 4, 4, True, None, id="varlen-B10-T1024-vl"),
    pytest.param(20, 512, 4, 4, True, None, id="varlen-B20-T512-vl"),
]

PRODUCT_CONFIGS = [
    pytest.param(1, 128, 4, 4, True,
                 [0, 47, 128],
                 id="prod-mixed-small"),
    pytest.param(1, 4096, 4, 4, True,
                 [0, 517, 883, 1010, 3767, 4096],
                 id="prod-mixed-segs"),
    pytest.param(1, 16384, 4, 4, True,
                 [0, 4096, 6893, 7665, 8192, 12288, 16384],
                 id="prod-mixed-large"),
    pytest.param(1, 256, 4, 4, True,
                 [0, 73, 115, 209],
                 id="prod-mixed-pad"),
]

ALL_CONFIGS = CORE_CONFIGS + DEVELOP_CONFIGS + VARLEN_CONFIGS + PRODUCT_CONFIGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, use_h0, state_v_first, seed=42,
):
    torch.manual_seed(seed)

    q = l2norm(torch.randn(
        batch_size, num_tokens, num_k_heads, HEAD_DIM_K,
        device=DEVICE, dtype=DATA_DTYPE,
    ))
    k = l2norm(torch.randn(
        batch_size, num_tokens, num_k_heads, HEAD_DIM_K,
        device=DEVICE, dtype=DATA_DTYPE,
    ))
    v = torch.randn(
        batch_size, num_tokens, num_v_heads, HEAD_DIM_V,
        device=DEVICE, dtype=DATA_DTYPE,
    )

    A = torch.zeros(num_v_heads, device=DEVICE, dtype=torch.float32) + 16
    gate_input = torch.randn(
        batch_size, num_tokens, num_v_heads,
        device=DEVICE, dtype=torch.float32,
    ) * 0.5
    dt_bias = torch.ones(num_v_heads, device=DEVICE, dtype=torch.float32)
    g = -A * torch.nn.functional.softplus(gate_input + dt_bias)

    beta = torch.randn(
        batch_size, num_tokens, num_v_heads,
        device=DEVICE, dtype=torch.float32,
    ).sigmoid()

    # h0 / dht in reference layout [B, Hv, K, V]
    h0_ref = None
    dht_ref = None
    if use_h0:
        h0_ref = torch.randn(
            batch_size, num_v_heads, HEAD_DIM_K, HEAD_DIM_V,
            device=DEVICE, dtype=torch.float32,
        )
        dht_ref = torch.randn(
            batch_size, num_v_heads, HEAD_DIM_K, HEAD_DIM_V,
            device=DEVICE, dtype=torch.float32,
        ) / 8

    do = torch.randn_like(v)

    # Handle varlen
    cu_seqlens = None
    if varlen:
        if cu_seqlens_list is not None:
            assert batch_size == 1
            cu_seqlens = torch.tensor(
                cu_seqlens_list, device=DEVICE, dtype=torch.int32,
            )
            if use_h0:
                real_batch_size = cu_seqlens.shape[0] - 1
                h0_ref = torch.randn(
                    real_batch_size, num_v_heads, HEAD_DIM_K, HEAD_DIM_V,
                    device=DEVICE, dtype=torch.float32,
                )
                dht_ref = torch.randn(
                    real_batch_size, num_v_heads, HEAD_DIM_K, HEAD_DIM_V,
                    device=DEVICE, dtype=torch.float32,
                ) / 8
        else:
            cu_seqlens = torch.randint(
                1, num_tokens, (batch_size,), device=DEVICE, dtype=torch.int32,
            )
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(cu_seqlens, dim=-1), (1, 0),
            )
            q = pack(q, cu_seqlens)
            k = pack(k, cu_seqlens)
            v = pack(v, cu_seqlens)
            g = pack(g, cu_seqlens)
            beta = pack(beta, cu_seqlens)
            do = pack(do, cu_seqlens)

    # QLA layout for h0 / dht (may need transpose for state_v_first)
    h0_qla = None
    dht_qla = None
    if h0_ref is not None:
        h0_qla = (
            h0_ref.transpose(-1, -2).contiguous()
            if state_v_first else h0_ref
        )
    if dht_ref is not None:
        dht_qla = (
            dht_ref.transpose(-1, -2).contiguous()
            if state_v_first else dht_ref
        )

    scale = HEAD_DIM_K ** (-0.5)

    return (
        q, k, v, g, beta, do,
        h0_ref, dht_ref,       # reference layout [B, H, K, V]
        h0_qla, dht_qla,       # QLA layout (may be [B, H, V, K] if state_v_first)
        cu_seqlens, scale,
    )


def _assert_relative(actual, expected, name, rtol=RTOL):
    if actual.shape[1] > expected.shape[1]:  # Padded
        assert not torch.any(torch.isnan(actual[:, expected.shape[1]:])), (
            f"{name}: got NaN in padded area"
        )
        actual = actual[:, :expected.shape[1]]
    error = torch.linalg.vector_norm(actual.double() - expected.double()).item()
    reference = torch.linalg.vector_norm(expected.double()).item()
    assert error <= reference * RTOL, (
        f"{name}: error={error:.6f}, reference={reference:.6f}, "
        f"relative={error / reference:.6f} > rtol={rtol}"
    )


def _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
             state_v_first, auto_cp):
    # Reference forward (float64)
    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(REF_DTYPE, copy=True),
        k=k.to(REF_DTYPE, copy=True),
        v=v.to(REF_DTYPE, copy=True),
        g=g.to(REF_DTYPE, copy=True),
        beta=beta.to(REF_DTYPE, copy=True),
        scale=scale,
        initial_state=h0_ref,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    # QLA forward
    g_qla, A_qla, o_qla, h_qla, s_qla, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0_qla,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=True,
        auto_cp=auto_cp,
        state_v_first=state_v_first,
    )

    return (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    )


def _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
             h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
             state_v_first, auto_cp):
    # Reference backward (float64)
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(REF_DTYPE, copy=True),
        k.to(REF_DTYPE, copy=True),
        v.to(REF_DTYPE, copy=True),
        g_ref,
        beta.to(REF_DTYPE, copy=True),
        A_ref.to(REF_DTYPE, copy=True),
        scale,
        h0_ref,
        do.to(REF_DTYPE, copy=True),
        dht_ref,
        cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    # QLA backward
    dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, auto_cp,
    )

    return (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla,
    )


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CORE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_fwd(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first, auto_cp=True)

    h_qla_cmp = h_qla.transpose(-1, -2) if state_v_first else h_qla
    s_qla_cmp = s_qla.transpose(-1, -2) if state_v_first else s_qla

    _assert_relative(o_qla, o_ref, "o_qla")
    _assert_relative(h_qla_cmp, h_ref, "h_qla")
    if h0_ref is not None:
        _assert_relative(s_qla_cmp, s_ref, "s_qla")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DEVELOP_CONFIGS + VARLEN_CONFIGS + PRODUCT_CONFIGS,
)
def test_fwd_extended(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    _assert_relative(o_qla, o_ref, "o_qla")
    _assert_relative(s_qla, s_ref, "s_qla")


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CORE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_bwd(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first, auto_cp=True)

    (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla,
    ) = _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
                 h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
                 state_v_first, auto_cp=True)

    _assert_relative(dq_qla, dq_ref, "dq_qla")
    _assert_relative(dk_qla, dk_ref, "dk_qla")
    _assert_relative(dv_qla, dv_ref, "dv_qla")
    _assert_relative(db_qla, db_ref, "db_qla")
    _assert_relative(dg_qla, dg_ref, "dg_qla")
    if dht_ref is not None:
        dh0_qla_cmp = (
            dh0_qla.transpose(-1, -2)
            if state_v_first else dh0_qla
        )
        _assert_relative(dh0_qla_cmp, dh0_ref, "dh0_qla")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DEVELOP_CONFIGS + VARLEN_CONFIGS + PRODUCT_CONFIGS,
)
def test_bwd_extended(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla,
    ) = _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
                 h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
                 state_v_first=False, auto_cp=True)

    _assert_relative(dq_qla, dq_ref, "dq_qla")
    _assert_relative(dk_qla, dk_ref, "dk_qla")
    _assert_relative(dv_qla, dv_ref, "dv_qla")
    _assert_relative(db_qla, db_ref, "db_qla")
    _assert_relative(dg_qla, dg_ref, "dg_qla")
    if dht_ref is not None:
        _assert_relative(dh0_qla, dh0_ref, "dh0_qla")


# ---------------------------------------------------------------------------
# Deterministic tests (run kernel many times, ensure no flaky results)
# ---------------------------------------------------------------------------

DETERMINISM_CONFIGS = [
    pytest.param(1, 4096, 2, 8, False, None, id="fixed-H8G2"),
    pytest.param(3, 4096, 4, 4, True, None, id="varlen-H4"),
    pytest.param(32, 1024, 2, 4, False, None, id="varlen-H4G2"),
]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DETERMINISM_CONFIGS,
)
def test_fwd_deterministic(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    for i in range(DETERMINISM_ITERS):
        _, _, o_qla_i, _, s_qla_i, _ = chunk_gated_delta_rule_fwd_qla(
            q, k, v, g, beta, scale, h0_qla, cu_seqlens,
            True, False, True, False,
        )
        _assert_relative(o_qla_i, o_ref, f"o_qla iter {i}")
        _assert_relative(s_qla_i, s_ref, f"s_qla iter {i}")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DETERMINISM_CONFIGS,
)
def test_bwd_deterministic(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        _, _, _, _, _, _,
    ) = _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
                 h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
                 state_v_first=False, auto_cp=True)

    for i in range(DETERMINISM_ITERS):
        dq_i, dk_i, dv_i, db_i, dg_i, dh0_i = chunk_gated_delta_rule_bwd_qla(
            q, k, v, g_qla, beta, A_qla, do, dht_qla,
            scale, h0_qla, cu_seqlens, False, True,
        )
        _assert_relative(dq_i, dq_ref, f"dq iter {i}")
        _assert_relative(dk_i, dk_ref, f"dk iter {i}")
        _assert_relative(dv_i, dv_ref, f"dv iter {i}")
        _assert_relative(dg_i, dg_ref, f"dg iter {i}")
        _assert_relative(db_i, db_ref, f"db iter {i}")
        if dht_ref is not None:
            _assert_relative(dh0_i, dh0_ref, f"dh0 iter {i}")


# ---------------------------------------------------------------------------
# Auto-CP specific tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    [
        pytest.param(1, 16384, 4, 4, False, None, id="long-fixed"),
        pytest.param(1, 16384, 4, 4, True,
                     [0, 4096, 8192, 12288, 16384],
                     id="long-varlen"),
    ],
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_fwd_auto_cp(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first,
):
    """auto_cp=True and auto_cp=False should produce equivalent results."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=state_v_first,
    )

    _, _, o_cp, _, s_cp, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, True, state_v_first,
    )
    _, _, o_nocp, _, s_nocp, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, False, state_v_first,
    )

    # Also check both against reference
    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(REF_DTYPE, copy=True),
        k=k.to(REF_DTYPE, copy=True),
        v=v.to(REF_DTYPE, copy=True),
        g=g.to(REF_DTYPE, copy=True),
        beta=beta.to(REF_DTYPE, copy=True),
        scale=scale,
        initial_state=h0_ref,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )
    s_ref_cmp = s_ref
    s_cp_cmp = s_cp.transpose(-1, -2) if state_v_first else s_cp
    s_nocp_cmp = s_nocp.transpose(-1, -2) if state_v_first else s_nocp

    _assert_relative(o_cp, o_ref, "o_cp_vs_ref")
    _assert_relative(o_nocp, o_ref, "o_nocp_vs_ref")
    _assert_relative(s_cp_cmp, s_ref_cmp, "s_cp_vs_ref")
    _assert_relative(s_nocp_cmp, s_ref_cmp, "s_nocp_vs_ref")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    [
        pytest.param(1, 16384, 4, 4, False, None, id="long-fixed"),
        pytest.param(1, 16384, 4, 4, True,
                     [0, 4096, 8192, 12288, 16384],
                     id="long-varlen"),
    ],
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_bwd_auto_cp(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first,
):
    """Backward: auto_cp=True and auto_cp=False should match reference."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=state_v_first,
    )

    # Run fwd for both cp modes
    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(REF_DTYPE, copy=True),
        k=k.to(REF_DTYPE, copy=True),
        v=v.to(REF_DTYPE, copy=True),
        g=g.to(REF_DTYPE, copy=True),
        beta=beta.to(REF_DTYPE, copy=True),
        scale=scale,
        initial_state=h0_ref,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    g_qla_cp, A_qla_cp, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, True, state_v_first,
    )
    g_qla_nocp, A_qla_nocp, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, False, state_v_first,
    )

    # Ref bwd
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(REF_DTYPE, copy=True),
        k.to(REF_DTYPE, copy=True),
        v.to(REF_DTYPE, copy=True),
        g_ref,
        beta.to(REF_DTYPE, copy=True),
        A_ref.to(REF_DTYPE, copy=True),
        scale, h0_ref,
        do.to(REF_DTYPE, copy=True),
        dht_ref, cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    # QLA bwd with cp
    dq_cp, dk_cp, dv_cp, db_cp, dg_cp, dh0_cp = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_cp, beta, A_qla_cp, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True,
    )
    # QLA bwd without cp
    dq_nocp, dk_nocp, dv_nocp, db_nocp, dg_nocp, dh0_nocp = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_nocp, beta, A_qla_nocp, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, False,
    )

    for prefix, dq, dk, dv, db, dg, dh0 in [
        ("cp", dq_cp, dk_cp, dv_cp, db_cp, dg_cp, dh0_cp),
        ("nocp", dq_nocp, dk_nocp, dv_nocp, db_nocp, dg_nocp, dh0_nocp),
    ]:
        _assert_relative(dq, dq_ref, f"dq_{prefix}")
        _assert_relative(dk, dk_ref, f"dk_{prefix}")
        _assert_relative(dv, dv_ref, f"dv_{prefix}")
        _assert_relative(db, db_ref, f"db_{prefix}")
        _assert_relative(dg, dg_ref, f"dg_{prefix}")
        if dht_ref is not None:
            dh0_cmp = dh0.transpose(-1, -2) if state_v_first else dh0
            _assert_relative(dh0_cmp, dh0_ref, f"dh0_{prefix}")


# ---------------------------------------------------------------------------
# CP cache tests (enable_fwd_cp_cache=True vs False)
# ---------------------------------------------------------------------------

CP_CACHE_CONFIGS = [
    pytest.param(1, 32768, 4, 4, False, None, id="cp-cache-H4"),
    pytest.param(1, 32768, 8, 8, False, None, id="cp-cache-H8"),
    pytest.param(1, 32768, 16, 16, False, None, id="cp-cache-H16"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CP_CACHE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_fwd_cp_cache(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    """Forward with enable_fwd_cp_cache should match forward without it."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    _, _, o_base, _, s_base, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=False,
    )
    _, _, o_cache, _, s_cache, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=True,
    )

    _assert_relative(o_cache, o_base, "o_cp_cache_vs_base")
    if use_h0:
        s_base_cmp = s_base.transpose(-1, -2) if state_v_first else s_base
        s_cache_cmp = s_cache.transpose(-1, -2) if state_v_first else s_cache
        _assert_relative(s_cache_cmp, s_base_cmp, "s_cp_cache_vs_base")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CP_CACHE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_bwd_cp_cache(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    """Backward with cp_cache should match backward without it."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    # Forward without cache (baseline)
    g_qla, A_qla, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=False,
    )
    # Forward with cache
    g_qla_c, A_qla_c, _, _, _, cp_cache = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=True,
    )

    # Backward without cache
    dq_base, dk_base, dv_base, db_base, dg_base, dh0_base = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True,
    )
    # Backward with cache
    dq_cache, dk_cache, dv_cache, db_cache, dg_cache, dh0_cache = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_c, beta, A_qla_c, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True, cp_cache=cp_cache,
    )

    _assert_relative(dq_cache, dq_base, "dq_cp_cache")
    _assert_relative(dk_cache, dk_base, "dk_cp_cache")
    _assert_relative(dv_cache, dv_base, "dv_cp_cache")
    _assert_relative(db_cache, db_base, "db_cp_cache")
    _assert_relative(dg_cache, dg_base, "dg_cp_cache")
    if dht_qla is not None:
        _assert_relative(dh0_cache, dh0_base, "dh0_cp_cache")


# ---------------------------------------------------------------------------
# Mixed CP control tests (fwd and bwd use different auto_cp settings)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("fwd_cp", [False, True], ids=["fwd_no_cp", "fwd_cp"])
@pytest.mark.parametrize("bwd_cp", [False, True], ids=["bwd_no_cp", "bwd_cp"])
def test_mixed_cp_control(state_v_first, fwd_cp, bwd_cp):
    """Mixing auto_cp=True/False between fwd and bwd should still match reference."""
    B, T, Hk, Hv = 1, 32768, 4, 4
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(B, T, Hk, Hv, False, None, True, state_v_first)

    # Reference
    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(REF_DTYPE, copy=True),
        k=k.to(REF_DTYPE, copy=True),
        v=v.to(REF_DTYPE, copy=True),
        g=g.to(REF_DTYPE, copy=True),
        beta=beta.to(REF_DTYPE, copy=True),
        scale=scale,
        initial_state=h0_ref,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(REF_DTYPE, copy=True),
        k.to(REF_DTYPE, copy=True),
        v.to(REF_DTYPE, copy=True),
        g_ref,
        beta.to(REF_DTYPE, copy=True),
        A_ref.to(REF_DTYPE, copy=True),
        scale, h0_ref,
        do.to(REF_DTYPE, copy=True),
        dht_ref, cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    # Case 1: fwd auto_cp=True, bwd auto_cp=False
    g_qla, A_qla, o, _, s, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, fwd_cp, state_v_first,
    )
    dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, bwd_cp,
    )

    _assert_relative(o, o_ref, "case1_o")
    _assert_relative(dq, dq_ref, "case1_dq")
    _assert_relative(dk, dk_ref, "case1_dk")
    _assert_relative(dv, dv_ref, "case1_dv")
    _assert_relative(db, db_ref, "case1_db")
    _assert_relative(dg, dg_ref, "case1_dg")
    dh0_cmp = dh0.transpose(-1, -2) if state_v_first else dh0
    _assert_relative(dh0_cmp, dh0_ref, "case1_dh0")
