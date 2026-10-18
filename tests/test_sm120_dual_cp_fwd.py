# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""SM120 regressions supplementing the shared CP kernel and distributed tests."""
import pytest
import torch

from flash_qla.ops.gated_delta_rule.chunk import (
    CHUNK_SIZE, DUAL_CP_FWD_SUPPORTED, DUAL_CP_BWD_SUPPORTED,
    aggregate_card_state,
)
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context
from flash_qla.ops.gated_delta_rule.chunk.cp.preprocess import cp_preprocess_bwd
from gdr_common import (
    make_inputs, assert_relative, chunk_gated_delta_rule_fwd_qla,
    chunk_gated_delta_rule_fwd_ref,
)
from ref_cp import ref_scan

pytestmark = [pytest.mark.gpu, pytest.mark.sm120]


def _kv(t, state_v_first):
    return t.transpose(-1, -2).contiguous() if state_v_first else t


def _assert_close(actual, expected, tol=0.02):
    assert torch.isfinite(actual).all()
    error = (actual.double() - expected.double()).abs().max()
    bound = expected.abs().max() * tol + 1e-6
    assert error <= bound, f"max error {error.item():.6g} > {bound.item():.6g}"


@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True])
@pytest.mark.parametrize("h_dtype, m_dtype", [
    (torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16), (torch.float32, torch.bfloat16),
], ids=["fp32", "bf16", "fp16", "mixed"])
def test_card_state_dtypes_and_resets(state_v_first, use_h0, h_dtype, m_dtype):
    torch.manual_seed(931)
    boundaries = [0, 1, 4, 9]  # singleton, uneven splits, and a longer scan
    n, heads, dim = boundaries[-1], 4, 128
    ht = (torch.randn(n, heads, dim, dim, device="cuda") * 0.1).to(h_dtype)
    # Noncommuting matrices with persistent history make order/last-segment bugs visible.
    mt = (torch.eye(dim, device="cuda") * 0.7 +
          torch.randn_like(ht.float()) * 0.02).to(m_dtype)
    # Head 0 retains history; the others reset at the first, last, or middle segment.
    fallback = torch.ones((n, heads), device="cuda", dtype=torch.bool)
    for lo, hi in zip(boundaries, boundaries[1:]):
        fallback[lo, 1] = False
        fallback[hi - 1, 2] = False
        fallback[(lo + hi) // 2, 3] = False
    # A non-fallback M can be undefined in prepare_h. It must never affect h/M.
    mt[~fallback] = float("nan")
    seed = None
    if use_h0:
        seed = torch.randn(3, heads, dim, dim, device="cuda")
    # Communication payload dtype is intentional, including raw_h0 conversion.
    ref_seed = seed.to(h_dtype) if seed is not None else None
    seq_map = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    _, ref_h, ref_m = ref_scan(ht, mt, fallback, seq_map, raw_h0=ref_seed)
    kwargs = dict(
        raw_h0=_kv(seed, state_v_first) if seed is not None else None,
        state_v_first=state_v_first,
    )
    h, m = aggregate_card_state(_kv(ht, state_v_first), mt, fallback, seq_map, **kwargs)
    assert h.dtype == m.dtype == h_dtype
    _assert_close(_kv(h, state_v_first), ref_h)
    _assert_close(m, ref_m)
    for seq, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        reset_heads = ~fallback[lo:hi].all(dim=0)
        assert torch.count_nonzero(m[seq, reset_heads]) == 0


@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True])
@pytest.mark.parametrize("cache", [False, True], ids=["uncached", "cached"])
def test_pure_intra_forward(state_v_first, use_h0, cache):
    cu_list = [0, 1031, 4096]
    q, k, v, g, beta, _, h0, _, seed, _, cu, scale = make_inputs(
        1, cu_list[-1], 4, 4, True, cu_list, use_h0, state_v_first,
    )
    ctx = build_cp_context(
        cu, enable_intra=True, num_v_heads=4,
        chunk_size=CHUNK_SIZE, force_intra_cp=True,
    )
    assert ctx.is_intra and not ctx.is_inter
    assert ctx.intra_cp_cu_seqlens.numel() > cu.numel()
    _, _, o, _, final, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, seed, cu, cp_context=ctx,
        state_v_first=state_v_first, enable_fwd_cp_cache=cache,
    )
    _, ref_o, _, _, ref_final = chunk_gated_delta_rule_fwd_ref(
        q.double(), k.double(), v.double(), g.double(), beta.double(),
        scale=scale, initial_state=h0.double() if h0 is not None else None,
        cu_seqlens=cu, chunk_size=CHUNK_SIZE,
    )
    assert_relative(o, ref_o, "pure intra output")
    assert_relative(_kv(final, state_v_first), ref_final, "pure intra final state")


def test_dual_backward_rejected_before_work():
    from types import SimpleNamespace
    from arch_utils import supports

    assert DUAL_CP_FWD_SUPPORTED and not DUAL_CP_BWD_SUPPORTED
    assert supports("inter_intra_fwd") and not supports("inter_intra_bwd")
    with pytest.raises(NotImplementedError, match="Dual-CP.*Backward"):
        cp_preprocess_bwd(
            SimpleNamespace(is_inter=True, is_intra=True),
            q=None, k=None, v=None, a=None, g=None, beta=None, do=None, dht=None,
            scale=1.0, initial_state=None, state_v_first=False, cp_cache=None,
        )
