# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Single-GPU math and integration coverage for SM120 Dual-CP forward.

No distributed process group is emulated here. The affine oracle uses float64
homogeneous matrices; the prepare_h integration uses a token-by-token GDN oracle.
"""
import pytest
import torch

from flash_qla.ops.gated_delta_rule.chunk import (
    CHUNK_SIZE, DUAL_CP_FWD_SUPPORTED, DUAL_CP_BWD_SUPPORTED,
    aggregate_card_state, fused_gdr_h, get_warmup_chunks_bidi, kkt_solve,
)
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context
from flash_qla.ops.gated_delta_rule.chunk.cp.preprocess import cp_preprocess_bwd
from flash_qla.ops.utils import chunk_local_cumsum
from gdr_common import (
    make_inputs, assert_relative, chunk_gated_delta_rule_fwd_qla,
    chunk_gated_delta_rule_fwd_ref,
)

pytestmark = [pytest.mark.gpu, pytest.mark.sm120]


def _kv(t, state_v_first):
    return t.transpose(-1, -2).contiguous() if state_v_first else t


def _assert_close(actual, expected, tol=0.02):
    assert torch.isfinite(actual).all()
    error = (actual.double() - expected.double()).abs().max()
    bound = expected.abs().max() * tol + 1e-6
    assert error <= bound, f"max error {error.item():.6g} > {bound.item():.6g}"


def _affine_oracle(ht, mt, fallback, boundaries, seed):
    """Compose [M h; 0 I] in float64; no kernel indexing/layout conventions."""
    heads, k, v = ht.shape[1:]
    identity = torch.eye(k + v, dtype=torch.float64, device=ht.device)
    hs, ms = [], []
    for seq, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        product = identity.expand(heads, -1, -1).clone()
        for idx in range(lo, hi):
            affine = identity.expand(heads, -1, -1).clone()
            affine[:, :k, :k] = torch.where(
                fallback[idx, :, None, None], mt[idx].double(), 0,
            )
            affine[:, :k, k:] = ht[idx].double()
            product = affine @ product
        h = product[:, :k, k:]
        if seed is not None:
            h = h + product[:, :k, :k] @ seed[seq].double()
        hs.append(h)
        ms.append(product[:, :k, :k])
    return torch.stack(hs), torch.stack(ms)


@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("seed_kind", ["none", "zero", "nonzero"])
@pytest.mark.parametrize("pattern", ["all", "none", "mixed"])
@pytest.mark.parametrize("h_dtype, m_dtype", [
    (torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16), (torch.float32, torch.bfloat16),
], ids=["fp32", "bf16", "fp16", "mixed"])
def test_card_affine_oracle(state_v_first, seed_kind, pattern, h_dtype, m_dtype):
    torch.manual_seed(931)
    boundaries = [0, 1, 4, 9]  # singleton, uneven splits, and a longer scan
    n, heads, dim = boundaries[-1], 4, 128
    ht = (torch.randn(n, heads, dim, dim, device="cuda") * 0.1).to(h_dtype)
    # Noncommuting matrices with persistent history make order/last-segment bugs visible.
    mt = (torch.eye(dim, device="cuda") * 0.7 +
          torch.randn_like(ht.float()) * 0.02).to(m_dtype)
    fallback = torch.full((n, heads), pattern != "none", device="cuda", dtype=torch.bool)
    if pattern == "mixed":
        for lo, hi in zip(boundaries, boundaries[1:]):
            fallback[lo, 1] = False
            fallback[hi - 1, 2] = False
            fallback[(lo + hi) // 2, 3] = False
    # A non-fallback M can be undefined in prepare_h. It must never affect h/M.
    mt[~fallback] = float("nan")
    seed = None
    if seed_kind != "none":
        seed = torch.randn(3, heads, dim, dim, device="cuda")
        if seed_kind == "zero":
            seed.zero_()
    # Communication payload dtype is intentional, including raw_h0 conversion.
    ref_seed = seed.to(h_dtype) if seed is not None else None
    ref_h, ref_m = _affine_oracle(ht, mt, fallback, boundaries, ref_seed)
    seq_map = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    kwargs = dict(
        raw_h0=_kv(seed, state_v_first) if seed is not None else None,
        state_v_first=state_v_first,
    )
    h, m = aggregate_card_state(_kv(ht, state_v_first), mt, fallback, seq_map, **kwargs)
    assert h.dtype == m.dtype == h_dtype
    _assert_close(_kv(h, state_v_first), ref_h)
    _assert_close(m, ref_m)
    h_only, no_m = aggregate_card_state(
        _kv(ht, state_v_first), mt, fallback, seq_map, compute_m=False, **kwargs,
    )
    assert no_m is None
    torch.testing.assert_close(h_only, h, rtol=0, atol=0)
    for seq, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        reset_heads = ~fallback[lo:hi].all(dim=0)
        assert torch.count_nonzero(m[seq, reset_heads]) == 0


def _token_oracle(k, v, g, beta, boundaries, seed):
    """Original GDN recurrence, independent of chunk factorization and scans."""
    k, v, g, beta = (x[0].double() for x in (k, v, g, beta))
    heads, dim = k.shape[1:]
    hs, ms = [], []
    for seq, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        h = torch.zeros(heads, dim, dim, dtype=torch.float64, device=k.device)
        if seed is not None:
            h.copy_(seed[seq])
        m = torch.eye(dim, dtype=torch.float64, device=k.device).expand(heads, -1, -1).clone()
        for t in range(lo, hi):
            key = k[t, :, :, None]
            decay = g[t].exp()[:, None, None]
            h = decay * h
            residual = v[t, :, None, :] - key.transpose(-1, -2) @ h
            h = h + beta[t, :, None, None] * key @ residual
            m = decay * (m - beta[t, :, None, None] * key @ (key.transpose(-1, -2) @ m))
        hs.append(h)
        ms.append(m)
    return torch.stack(hs), torch.stack(ms)


@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True])
@pytest.mark.parametrize("ragged", [False, True], ids=["aligned", "ragged"])
def test_prepared_card_state(state_v_first, use_h0, ragged):
    # All cuts inside a raw sequence align to its chunk grid; only its last
    # segment may be short. The next raw sequence can begin at any token offset.
    cp_cu_list = [0, 7, 39, 84, 148, 161] if ragged else [0, 32, 64, 96, 160, 192]
    boundaries = [0, 1, 3, 5]
    raw_cu = [cp_cu_list[i] for i in boundaries]
    _, k, v, g, beta, _, h0, _, _, _, _, _ = make_inputs(
        1, cp_cu_list[-1], 2, 2, True, raw_cu, use_h0, False,
    )
    g = g * 0.001  # retain history so missing last-segment M is observable
    cp_cu = torch.tensor(cp_cu_list, dtype=torch.int32, device="cuda")
    seq_map = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    gf = chunk_local_cumsum(g, cu_seqlens=cp_cu, chunk_size=CHUNK_SIZE)
    a = kkt_solve(k=k, b=beta, cu_seqlens=cp_cu, chunk_size=CHUNK_SIZE)
    ht_fwd = torch.tensor([True, False, True, False, True], device="cuda")
    ht_bwd = torch.tensor([True, True, False, True, False], device="cuda")
    nw, _, fallback, _ = get_warmup_chunks_bidi(
        gf, cp_cu, ht_fwd, ht_bwd, chunk_size=CHUNK_SIZE,
        seq_map_r2c=seq_map, force_inter_boundaries=True,
    )
    _, ht, mt = fused_gdr_h(
        k, v, a, gf, beta, output_h=False,
        cu_seqlens=cp_cu, num_warmup_chunks=nw, state_v_first=state_v_first,
    )
    h, m = aggregate_card_state(
        ht, mt, fallback, seq_map,
        raw_h0=_kv(h0, state_v_first) if h0 is not None else None,
        state_v_first=state_v_first,
    )
    # The middle sequence's terminal mask discards history by design; the main
    # forward only needs its entrances. Boundary sequences must be exact affine
    # summaries for communication, including sub-chunk and ragged last segments.
    for seq in (0, 2):
        assert fallback[boundaries[seq]:boundaries[seq + 1]].all()
    ref_h, ref_m = _token_oracle(
        k, v, g, beta, raw_cu, h0.to(ht.dtype) if h0 is not None else None,
    )
    _assert_close(_kv(h, state_v_first)[[0, 2]], ref_h[[0, 2]])
    _assert_close(m[[0, 2]], ref_m[[0, 2]])
    # Also check every local summary against the effective masked affine map.
    masked_h, masked_m = _affine_oracle(
        _kv(ht, state_v_first), mt, fallback, boundaries,
        h0.to(ht.dtype) if h0 is not None else None,
    )
    _assert_close(_kv(h, state_v_first), masked_h)
    _assert_close(m, masked_m)


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
