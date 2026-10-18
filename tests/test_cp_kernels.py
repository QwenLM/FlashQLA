# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ref_cp import ref_warmup, ref_warmup_bidi, ref_scan, rel_max, THRESHOLD
from flash_qla.ops.gated_delta_rule.chunk import (
    CHUNK_SIZE,
    get_warmup_chunks,
    get_warmup_chunks_bidi,
    correct_initial_states,
    correct_terminal_states,
    aggregate_card_state,
)

DEVICE = "cuda"
K = V = 128                # the scan kernels assert this
SEED = 20260819

SCAN_RTOL = 2e-2


LAYOUTS = [
    pytest.param([1], id="one-seq-unsplit"),
    pytest.param([2, 2, 2], id="uniform-split"),
    pytest.param([1, 3, 2], id="mixed-split"),
    pytest.param([5], id="deep-split"),
]


def _build_layout(segments, *, ragged_tail):
    """-> (cu_seqlens, seq_map_r2c, ht_mask_fwd, ht_mask_bwd) for a synthetic split."""
    lengths = []
    seq_map = [0]
    for n_seg in segments:
        for j in range(n_seg):
            length = (j % 3 + 1) * CHUNK_SIZE
            if ragged_tail and j == n_seg - 1:
                length += 13
            lengths.append(length)
        seq_map.append(len(lengths))

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    n_cp = len(lengths)

    ht_fwd = torch.zeros(n_cp, dtype=torch.bool, device=DEVICE)
    ht_bwd = torch.zeros(n_cp, dtype=torch.bool, device=DEVICE)
    for i in range(len(segments)):
        ht_fwd[seq_map[i + 1] - 1] = True   # last cp segment of the raw seq
        ht_bwd[seq_map[i]] = True           # first cp segment of the raw seq

    return (
        torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        torch.tensor(seq_map, dtype=torch.int32, device=DEVICE),
        ht_fwd,
        ht_bwd,
    )


def _make_g(cu, num_heads, *, seed=SEED):
    """Per-token log decay, cumulative within each chunk -- the layout the warmup kernels
    expect (they sample one token per chunk and read its within-chunk cumulative sum).

    Heads 0 and -1 are pinned so both fallback branches are present by construction, in
    every direction and for any (even ragged) segment -- no separate test has to confirm
    the mix. Head 0's per-token decay is small enough that its accumulated warmup sum stays
    above THRESHOLD even over the longest segment (always fallback); the last head's is
    steep enough that a single sampled token already crosses THRESHOLD (never fallback).
    The middle heads are random, spread by ``logspace`` across the two extremes.
    """
    cu_list = cu.tolist()
    total = cu_list[-1]
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    base = -torch.rand(1, total, num_heads, generator=gen, device=DEVICE) * 0.5 - 0.01
    scale = torch.logspace(
        -2, 1.2, num_heads, device=DEVICE
    ) if num_heads > 1 else torch.ones(1, device=DEVICE)
    g = base * scale
    if num_heads > 1:
        max_seg = max(hi - lo for lo, hi in zip(cu_list[:-1], cu_list[1:]))
        g[0, :, 0] = 0.5 * THRESHOLD / max_seg   # summed over any segment: > THRESHOLD
        g[0, :, -1] = 2.0 * THRESHOLD            # one token alone: < THRESHOLD
    # cumulative within each chunk of each segment
    for lo, hi in zip(cu_list[:-1], cu_list[1:]):
        for c0 in range(lo, hi, CHUNK_SIZE):
            c1 = min(c0 + CHUNK_SIZE, hi)
            g[0, c0:c1] = torch.cumsum(g[0, c0:c1], dim=0)
    return g.float()


# ===========================================================================
# Synthetic scan inputs
# ===========================================================================
def _make_scan_inputs(seq_map, n_cp, H, *, state_v_first, fallback_pattern, seed=SEED):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (n_cp, H, V, K) if state_v_first else (n_cp, H, K, V)
    ht = torch.randn(shape, generator=gen, device=DEVICE, dtype=torch.float32) * 0.1
    # spectral norm well under 1 so the M product decays instead of blowing up
    mt = torch.randn(
        (n_cp, H, K, K), generator=gen, device=DEVICE, dtype=torch.float32
    ) * (0.5 / K ** 0.5)
    if fallback_pattern == "all":
        fallback = torch.ones(n_cp, H, dtype=torch.bool, device=DEVICE)
    elif fallback_pattern == "none":
        fallback = torch.zeros(n_cp, H, dtype=torch.bool, device=DEVICE)
    else:
        fallback = torch.rand(n_cp, H, generator=gen, device=DEVICE) < 0.7
    return ht, mt, fallback


FALLBACK_PATTERNS = ["all", "none", "random"]


# ===========================================================================
# get_warmup_chunks
# ===========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("ragged_tail", [False, True], ids=["aligned", "ragged"])
@pytest.mark.parametrize("reverse", [False, True], ids=["fwd", "reverse"])
@pytest.mark.parametrize("num_heads", [4, 16])
def test_get_warmup_chunks(segments, ragged_tail, reverse, num_heads):
    cu, _, ht_fwd, ht_bwd = _build_layout(segments, ragged_tail=ragged_tail)
    ht_mask = ht_bwd if reverse else ht_fwd
    g = _make_g(cu, num_heads)

    num_warmup, fallback = get_warmup_chunks(
        g=g, cu_seqlens=cu, ht_mask=ht_mask, chunk_size=CHUNK_SIZE, reverse=reverse,
    )
    exp_warmup, exp_fallback, valid = ref_warmup(g, cu, ht_mask, reverse=reverse)

    assert torch.equal(num_warmup, exp_warmup), (
        f"num_warmup mismatch\ngot:\n{num_warmup}\nwant:\n{exp_warmup}"
    )
    # `fallback_mask` rows under ht_mask are never written by this kernel, so only the
    # rest are meaningful; see ref_cp.ref_warmup.
    assert torch.equal(fallback[valid], exp_fallback[valid]), (
        f"fallback mismatch\ngot:\n{fallback}\nwant:\n{exp_fallback}\nvalid:\n{valid}"
    )


# ===========================================================================
# get_warmup_chunks_bidi
# ===========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("ragged_tail", [False, True], ids=["aligned", "ragged"])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("force_inter", [False, True], ids=["free", "force-boundaries"])
def test_get_warmup_chunks_bidi(segments, ragged_tail, num_heads, force_inter):
    """All four outputs against the reference. With ``force_inter`` on, the partitions of
    the first and last raw sequence are forced to full warmup inside the kernel (folding
    the old Python-side override into the tilelang scan), including the single-seq
    degeneracy where every partition is a boundary."""
    cu, seq_map, ht_fwd, ht_bwd = _build_layout(segments, ragged_tail=ragged_tail)
    g = _make_g(cu, num_heads)

    kw = dict(seq_map_r2c=seq_map, force_inter_boundaries=True) if force_inter else {}
    n_h, n_bwd, f_fwd, f_bwd = get_warmup_chunks_bidi(
        g=g, cu_seqlens=cu, ht_mask_fwd=ht_fwd, ht_mask_bwd=ht_bwd,
        chunk_size=CHUNK_SIZE, **kw,
    )
    r_h, r_bwd, r_f_fwd, r_f_bwd = ref_warmup_bidi(g, cu, ht_fwd, ht_bwd, **kw)

    assert torch.equal(n_h, r_h), f"num_warmup_h\ngot:\n{n_h}\nwant:\n{r_h}"
    assert torch.equal(n_bwd, r_bwd), f"num_warmup_bwd\ngot:\n{n_bwd}\nwant:\n{r_bwd}"
    assert torch.equal(f_fwd, r_f_fwd), f"fallback_fwd\ngot:\n{f_fwd}\nwant:\n{r_f_fwd}"
    assert torch.equal(f_bwd, r_f_bwd), f"fallback_bwd\ngot:\n{f_bwd}\nwant:\n{r_f_bwd}"


# ===========================================================================
# correct_initial_states / correct_terminal_states
# ===========================================================================
_FWD_BUF_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="prod-bf16"),
]
_BWD_BUF_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="fp32"),
    pytest.param(torch.float32, torch.bfloat16, id="prod-mixed"),
]

@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_h0", [False, True], ids=["zero_seed", "raw_h0"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize("h_dtype, m_dtype", _FWD_BUF_DTYPES)
def test_correct_initial_states(
    segments, state_v_first, use_raw_h0, fallback_pattern, h_dtype, m_dtype
):
    H = 4
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first, fallback_pattern=fallback_pattern,
    )
    ht, mt = ht.to(h_dtype), mt.to(m_dtype)
    raw_shape = (len(segments), H, V, K) if state_v_first else (len(segments), H, K, V)
    raw_h0 = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32) if use_raw_h0 else None

    out = correct_initial_states(
        raw_h0=raw_h0, ht_buffer=ht, mt_buffer=mt, fallback_mask=fallback,
        seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    ref, _, _ = ref_scan(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
    )
    err = rel_max(out, ref)
    assert out.dtype == torch.float32, "the main kernel's seed must stay fp32"
    assert err <= SCAN_RTOL, f"cp_h0 rel_max={err:.2e} > {SCAN_RTOL:g}"

    if fallback_pattern == "none":
        seq_map_list = seq_map.tolist()
        for bb in range(len(segments)):
            lo, hi = seq_map_list[bb], seq_map_list[bb + 1]
            seed = raw_h0[bb] if raw_h0 is not None else torch.zeros_like(out[lo])
            assert rel_max(out[lo], seed) <= SCAN_RTOL, f"seq {bb} seed"
            for idx in range(lo + 1, hi):
                assert rel_max(out[idx], ht[idx - 1]) <= SCAN_RTOL, f"seq {bb} segment {idx}"


@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_dht", [False, True], ids=["zero_seed", "raw_dht"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize("h_dtype, m_dtype", _BWD_BUF_DTYPES)
def test_correct_terminal_states(
    segments, state_v_first, use_raw_dht, fallback_pattern, h_dtype, m_dtype
):
    """The backward twin: same scan, walked in reverse, with M transposed."""
    H = 4
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    dht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first,
        fallback_pattern=fallback_pattern, seed=SEED + 1,
    )
    dht, mt = dht.to(h_dtype), mt.to(m_dtype)
    raw_shape = (len(segments), H, V, K) if state_v_first else (len(segments), H, K, V)
    raw_dht = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32) if use_raw_dht else None

    out = correct_terminal_states(
        raw_dht=raw_dht, dht_buffer=dht, mt_buffer=mt, fallback_mask=fallback,
        seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    ref, _, _ = ref_scan(
        dht, mt, fallback, seq_map, raw_h0=raw_dht,
        state_v_first=state_v_first, reverse=True, transpose_m=True,
    )
    err = rel_max(out, ref)
    assert out.dtype == torch.float32, "the main kernel's seed must stay fp32"
    assert err <= SCAN_RTOL, f"cp_dht rel_max={err:.2e} > {SCAN_RTOL:g}"


# ===========================================================================
# aggregate_card_state (inter+intra only)
# ===========================================================================
@pytest.mark.gpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize("use_raw_h0", [False, True], ids=["zero-seed", "raw-h0"])
@pytest.mark.parametrize(
    "reverse, transpose_m", [
        pytest.param(False, False, id="fwd"),
        pytest.param(True, True, id="bwd", marks=pytest.mark.needs_bwd),
    ],
)
def test_aggregate_card_state(
    segments, state_v_first, fallback_pattern, use_raw_h0, reverse, transpose_m
):
    H = 4
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first,
        fallback_pattern=fallback_pattern, seed=SEED + 2,
    )
    ht, mt = ht.bfloat16(), mt.bfloat16()
    raw_shape = (len(segments), H, V, K) if state_v_first else (len(segments), H, K, V)
    raw_h0 = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32) if use_raw_h0 else None

    h_card, m_card = aggregate_card_state(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m, compute_m=True,
    )
    _, ref_h, ref_m = ref_scan(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m,
    )
    err_h = rel_max(h_card, ref_h)
    assert err_h <= SCAN_RTOL, f"h_card rel_max={err_h:.2e} > {SCAN_RTOL:g}"
    err_m = rel_max(m_card, ref_m)
    assert err_m <= SCAN_RTOL, f"m_card rel_max={err_m:.2e} > {SCAN_RTOL:g}"

    h_no_m, m_no_m = aggregate_card_state(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m, compute_m=False,
    )
    assert m_no_m is None
    err_no_m = rel_max(h_no_m, h_card)
    assert err_no_m <= 1e-5, f"h_card must not depend on compute_m, rel_max={err_no_m:.2e}"


@pytest.mark.gpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_h0", [False, True], ids=["zero-seed", "raw-h0"])
def test_aggregate_card_state_affine_oracle(state_v_first, use_raw_h0):
    gen = torch.Generator(device=DEVICE).manual_seed(SEED + 3)
    bounds, H = [0, 1, 4, 9], 4  # singleton and unequal segment counts
    ht = (torch.randn(bounds[-1], H, K, V, generator=gen, device=DEVICE) * 0.1).bfloat16()
    # Noncommuting matrices retain enough history to expose order/last-segment errors.
    mt = (0.7 * torch.eye(K, device=DEVICE) +
          0.02 * torch.randn(bounds[-1], H, K, K, generator=gen, device=DEVICE)).bfloat16()
    fallback = torch.ones(bounds[-1], H, dtype=torch.bool, device=DEVICE)
    for lo, hi in zip(bounds, bounds[1:]):
        # Head 0 retains history; other heads reset at the first, last, or middle segment.
        fallback[lo, 1] = fallback[hi - 1, 2] = fallback[(lo + hi) // 2, 3] = False
    mt[~fallback] = float("nan")  # prepare_h may leave these transitions undefined
    seed = torch.randn(len(bounds) - 1, H, K, V, generator=gen, device=DEVICE) if use_raw_h0 else None

    # Independent affine composition: [M h; 0 I], without ref_scan's state recurrence.
    identity = torch.eye(K + V, dtype=torch.float64, device=DEVICE)
    expected_h, expected_m = [], []
    for seq, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        product = identity.expand(H, -1, -1).clone()
        for idx in range(lo, hi):
            affine = identity.expand(H, -1, -1).clone()
            affine[:, :K, :K] = torch.where(fallback[idx, :, None, None], mt[idx].double(), 0)
            affine[:, :K, K:] = ht[idx].double()
            product = affine @ product
        h = product[:, :K, K:]
        if seed is not None:
            # Match the documented communication payload conversion before composition.
            h = h + product[:, :K, :K] @ seed[seq].to(ht.dtype).double()
        expected_h.append(h)
        expected_m.append(product[:, :K, :K])

    def layout(t):
        return t.transpose(-1, -2).contiguous() if state_v_first else t

    seq_map = torch.tensor(bounds, dtype=torch.int32, device=DEVICE)
    h_card, m_card = aggregate_card_state(
        layout(ht), mt, fallback, seq_map,
        raw_h0=layout(seed) if seed is not None else None, state_v_first=state_v_first,
    )
    assert h_card.dtype == m_card.dtype == ht.dtype
    for name, actual, expected in (
        ("h", layout(h_card), torch.stack(expected_h)),
        ("M", m_card, torch.stack(expected_m)),
    ):
        assert torch.isfinite(actual).all(), name
        assert rel_max(actual, expected) <= SCAN_RTOL, name
    assert torch.count_nonzero(m_card[:, 1:]) == 0
