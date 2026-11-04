# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import pytest
import torch

from gdr_common import (
    RTOL,
    HEAD_DIM_K,
    HEAD_DIM_V,
    CHUNK_SIZE,
    REF_DTYPE,
    DATA_DTYPE,
    DEVICE,
    chunk_gated_delta_rule_fwd_qla,
    chunk_gated_delta_rule_bwd_qla,
    chunk_gated_delta_rule_fwd_ref,
    chunk_gated_delta_rule_bwd_ref,
    make_inputs,
    assert_relative,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DETERMINISM_ITERS = 1000

# (B, T, Hk, Hv, varlen, cu_seqlens, check_h, long_memory)
CONFIGS = [
    # dense
    pytest.param(1, 4096, 4, 4, False, None, True, False, id="dense-H4"),
    pytest.param(1, 4096, 2, 8, False, None, True, False, id="dense-Hk2Hv8"),
    pytest.param(1, 4096, 4, 16, False, None, True, False, id="dense-Hk4Hv16"),
    # varlen: explicit segments, chunk-unaligned boundaries, sub-chunk, padding, GQA
    pytest.param(1, 128, 4, 4, True, [0, 47, 128], True, False, id="vl-subchunk"),
    pytest.param(1, 4096, 4, 4, True, [0, 517, 883, 1010, 3767, 4096], True, False, id="vl-disparate"),
    pytest.param(1, 4096, 16, 32, True, [0, 410, 841, 1135, 2126, 2512, 4096], True, False, id="vl-gqa"),
    pytest.param(1, 256, 4, 4, True, [0, 73, 115, 209], True, False, id="vl-padded"),
    # long stress: no h compare (o/s only)
    pytest.param(1, 16384, 4, 4, True, [0, 4096, 6893, 7665, 8192, 12288, 16384], False, False, id="vl-long"),
    pytest.param(1, 32768, 16, 16, False, None, False, False, id="dense-long"),
    # Long sequences with weak decay and weak beta, using normal AutoCP selection.
    *[
        pytest.param(1, T, 4, 4, True, [0, T], False, True,
                     id=f"long-memory-{T // 1024}k", marks=pytest.mark.slow)
        for T in (16384, 65536, 131072)
    ],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
             state_v_first, auto_cp, output_h=True):
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
        output_h=output_h,
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
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list, check_h, long_memory",
    CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_fwd(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, check_h, long_memory, state_v_first, use_h0,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
        long_memory=long_memory,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first, auto_cp=True, output_h=check_h)

    assert_relative(o_qla, o_ref, "o_qla")
    if check_h:
        h_qla_cmp = h_qla.transpose(-1, -2) if state_v_first else h_qla
        assert_relative(h_qla_cmp, h_ref, "h_qla")
    if h0_ref is not None or long_memory:
        s_qla_cmp = s_qla.transpose(-1, -2) if state_v_first else s_qla
        assert_relative(s_qla_cmp, s_ref, "s_qla")


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list, check_h, long_memory",
    CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_bwd(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, check_h, long_memory, state_v_first, use_h0,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
        long_memory=long_memory,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first, auto_cp=True, output_h=False)

    (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla,
    ) = _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
                 h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
                 state_v_first, auto_cp=True)

    assert_relative(dq_qla, dq_ref, "dq_qla")
    assert_relative(dk_qla, dk_ref, "dk_qla")
    assert_relative(dv_qla, dv_ref, "dv_qla")
    assert_relative(db_qla, db_ref, "db_qla")
    assert_relative(dg_qla, dg_ref, "dg_qla")
    if dht_ref is not None:
        dh0_qla_cmp = (
            dh0_qla.transpose(-1, -2)
            if state_v_first else dh0_qla
        )
        assert_relative(dh0_qla_cmp, dh0_ref, "dh0_qla")


# ---------------------------------------------------------------------------
# Forward CP cache
# ---------------------------------------------------------------------------
# The cache only exists once intra CP has run, so force it on. e2e drives the cache
# multi-card through the public API (on by default there); the tests above run cache-off;
# this is the single-GPU cover for the cache-on path.
CP_CACHE_CONFIGS = [
    pytest.param(1, 8192, 4, 4, False, None, id="cache-dense"),
    pytest.param(1, 8192, 4, 4, True, [0, 2048, 4096, 6144, 8192], id="cache-varlen"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CP_CACHE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_cp_cache(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first,
):
    """Forward with enable_fwd_cp_cache fed straight into the backward must match the
    float64 reference on both halves -- stronger than a cache-on==cache-off diff."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=state_v_first,
    )

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

    g_qla, A_qla, o_qla, _, s_qla, cp_cache = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=h0_qla,
        cu_seqlens=cu_seqlens, output_final_state=True, output_h=False,
        state_v_first=state_v_first, force_intra_cp=True, enable_fwd_cp_cache=True,
    )
    assert cp_cache is not None, "enable_fwd_cp_cache=True produced no cache"

    dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True,
        cp_cache=cp_cache, force_intra_cp=True,
    )

    assert_relative(o_qla, o_ref, "o_qla")
    s_qla_cmp = s_qla.transpose(-1, -2) if state_v_first else s_qla
    assert_relative(s_qla_cmp, s_ref, "s_qla")
    assert_relative(dq_qla, dq_ref, "dq_qla")
    assert_relative(dk_qla, dk_ref, "dk_qla")
    assert_relative(dv_qla, dv_ref, "dv_qla")
    assert_relative(db_qla, db_ref, "db_qla")
    assert_relative(dg_qla, dg_ref, "dg_qla")
    if dht_ref is not None:
        dh0_qla_cmp = dh0_qla.transpose(-1, -2) if state_v_first else dh0_qla
        assert_relative(dh0_qla_cmp, dh0_ref, "dh0_qla")


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
    ) = make_inputs(
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
        assert_relative(o_qla_i, o_ref, f"o_qla iter {i}")
        assert_relative(s_qla_i, s_ref, f"s_qla iter {i}")


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
    ) = make_inputs(
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
        assert_relative(dq_i, dq_ref, f"dq iter {i}")
        assert_relative(dk_i, dk_ref, f"dk iter {i}")
        assert_relative(dv_i, dv_ref, f"dv iter {i}")
        assert_relative(dg_i, dg_ref, f"dg iter {i}")
        assert_relative(db_i, db_ref, f"db iter {i}")
        if dht_ref is not None:
            assert_relative(dh0_i, dh0_ref, f"dh0 iter {i}")
