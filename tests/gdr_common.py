# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Shared scaffolding for the gated-delta-rule chunk tests.

Holds the direct-call wrappers, input construction and relative-error assertion
used by the pure-kernel correctness suite (``test_gdr_unit.py``) and the multi-card
CP end-to-end tests (``test_cp_e2e.py``).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla.ops.gated_delta_rule.chunk import (
    CHUNK_SIZE,
    chunk_gated_delta_rule_fwd as _chunk_gdr_fwd_impl,
    chunk_gated_delta_rule_bwd as _chunk_gdr_bwd_impl,
    _auto_intra_cp_context,
)
from flash_qla.utils import l2norm, pack
from ref_gdr import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_ref  # noqa: F401
from ref_gdr import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_ref  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RTOL = 0.02
HEAD_DIM_K = 128
HEAD_DIM_V = 128
REF_DTYPE = torch.float64
DATA_DTYPE = torch.bfloat16
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Direct-call wrappers
#
# The low-level chunk functions require a non-null cp_context (the null->auto
# guard now lives in the autograd Function). These test wrappers build the auto
# intra-CP context the same way the autograd Function does, so the direct-call
# tests below can keep passing `auto_cp` as before.
# ---------------------------------------------------------------------------

def chunk_gated_delta_rule_fwd_qla(
    q, k, v, g, beta, scale=None, initial_state=None, cu_seqlens=None,
    output_final_state=True, output_h=False, auto_cp=True, state_v_first=False,
    enable_fwd_cp_cache=False, cp_context=None, force_intra_cp=False, is_train=True,
):
    if cp_context is None:
        cp_context = _auto_intra_cp_context(
            k, v, cu_seqlens, CHUNK_SIZE, auto_cp, is_train,
            force_intra_cp=force_intra_cp,
        )
    return _chunk_gdr_fwd_impl(
        q, k, v, g, beta, scale, initial_state, cu_seqlens,
        output_final_state, output_h, auto_cp, state_v_first,
        enable_fwd_cp_cache, cp_context, is_train,
    )


def chunk_gated_delta_rule_bwd_qla(
    q, k, v, g, beta, A, do, dht=None, scale=None, initial_state=None,
    cu_seqlens=None, state_v_first=False, auto_cp=True, cp_cache=None, cp_context=None,
    force_intra_cp=False, is_train=True,
):
    if cp_context is None:
        cp_context = _auto_intra_cp_context(
            k, v, cu_seqlens, CHUNK_SIZE, auto_cp, is_train,
            force_intra_cp=force_intra_cp,
        )
    return _chunk_gdr_bwd_impl(
        q, k, v, g, beta, A, do, dht, scale, initial_state,
        cu_seqlens, state_v_first, auto_cp, cp_cache, cp_context,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(
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

    A = torch.rand(num_v_heads, device=DEVICE, dtype=torch.float32) * 16
    A[0] = 0
    A[-1] = 16
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


def assert_relative(actual, expected, name, rtol=RTOL):
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
