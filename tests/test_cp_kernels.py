# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ref_cp import ref_warmup, ref_warmup_bidi, ref_scan, rel_max, THRESHOLD
from arch_utils import ARCH
from flash_qla.ops.gated_delta_rule.chunk import (
    CHUNK_SIZE,
    _CP_SUPPORTS_SLOT_SCAN,
    get_warmup_chunks,
    get_warmup_chunks_bidi,
    correct_initial_states,
    correct_terminal_states,
    aggregate_card_state,
)

DEVICE = "cuda"
K = V = 128                # the scan kernels assert this
SEED = 20260819

SCAN_RTOL = 2e-5
LEGACY_SCAN_RTOL = 2e-2


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
def _make_scan_inputs(
    seq_map, n_cp, H, *, state_v_first, fallback_pattern, seed=SEED, near_identity=False,
):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (n_cp, H, V, K) if state_v_first else (n_cp, H, K, V)
    ht = torch.randn(shape, generator=gen, device=DEVICE, dtype=torch.float32) * 0.1
    # Short scans use contracting M; long scans retain history to expose accumulated error.
    mt = torch.randn(
        (n_cp, H, K, K), generator=gen, device=DEVICE, dtype=torch.float32
    ) * (0.0001 if near_identity else 0.5 / K ** 0.5)
    if near_identity:
        mt += torch.eye(K, device=DEVICE) * 0.999
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
# Vary batch/head parallelism and scan length through the automatic launcher.
# The expected result is independent of which precision or tile it selects.
_CORRECTION_SHAPES = [
    pytest.param([1], 1, SCAN_RTOL, id="single-unsplit"),
    pytest.param([5], 1, SCAN_RTOL, id="single-h1"),
    pytest.param([2, 2, 2], 8, SCAN_RTOL, id="uniform-h8"),
    pytest.param([1, 3, 2], 16, SCAN_RTOL, id="mixed-h16"),
    pytest.param([5], 128, SCAN_RTOL, id="single-h128"),
    pytest.param([64] + [1] * 47, 1, 5e-5, id="long64-b48"),
    pytest.param([256] + [1] * 47, 1, 2e-4, id="long256-b48"),
]

_FWD_BUF_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="legacy-bf16"),
]
_BWD_BUF_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="fp32"),
    pytest.param(torch.float32, torch.bfloat16, id="legacy-mixed"),
]

@pytest.mark.gpu
@pytest.mark.parametrize("segments, num_heads, fp32_rtol", _CORRECTION_SHAPES)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_h0", [False, True], ids=["zero_seed", "raw_h0"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize("h_dtype, m_dtype", _FWD_BUF_DTYPES)
def test_correct_initial_states(
    segments, num_heads, fp32_rtol, state_v_first, use_raw_h0, fallback_pattern, h_dtype, m_dtype,
):
    scan_rtol = (fp32_rtol if h_dtype == m_dtype == torch.float32
                 and ARCH in ("SM90", "SM100", "SM103", "SM120") else LEGACY_SCAN_RTOL)
    H = num_heads
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first, fallback_pattern=fallback_pattern,
        near_identity=max(segments) >= 64,
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
    assert err <= scan_rtol, f"cp_h0 rel_max={err:.2e} > {scan_rtol:g}"

    if fallback_pattern == "none":
        seq_map_list = seq_map.tolist()
        for bb in range(len(segments)):
            lo, hi = seq_map_list[bb], seq_map_list[bb + 1]
            seed = raw_h0[bb] if raw_h0 is not None else torch.zeros_like(out[lo])
            assert rel_max(out[lo], seed) <= scan_rtol, f"seq {bb} seed"
            for idx in range(lo + 1, hi):
                assert rel_max(out[idx], ht[idx - 1]) <= scan_rtol, f"seq {bb} segment {idx}"


@pytest.mark.gpu
@pytest.mark.parametrize("segments, num_heads, fp32_rtol", _CORRECTION_SHAPES)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_dht", [False, True], ids=["zero_seed", "raw_dht"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize("h_dtype, m_dtype", _BWD_BUF_DTYPES)
def test_correct_terminal_states(
    segments, num_heads, fp32_rtol, state_v_first, use_raw_dht, fallback_pattern, h_dtype, m_dtype,
):
    """The backward twin: same scan, walked in reverse, with M transposed."""
    scan_rtol = (fp32_rtol if h_dtype == m_dtype == torch.float32
                 and ARCH in ("SM90", "SM100", "SM103", "SM120") else LEGACY_SCAN_RTOL)
    H = num_heads
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    dht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first,
        fallback_pattern=fallback_pattern, seed=SEED + 1,
        near_identity=max(segments) >= 64,
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
    assert err <= scan_rtol, f"cp_dht rel_max={err:.2e} > {scan_rtol:g}"


# ===========================================================================
# aggregate_card_state (inter+intra only)
# ===========================================================================
_AGGREGATE_CASES = [
    pytest.param(
        *shape.values, fallback, use_raw, reverse, reverse, *dtypes.values,
        id=f"{shape.id}-{fallback}-{'raw-h0' if use_raw else 'zero-seed'}-"
           f"{'bwd' if reverse else 'fwd'}-{dtypes.id}",
    )
    for shape in _CORRECTION_SHAPES
    for fallback in FALLBACK_PATTERNS
    for use_raw in (False, True)
    for reverse in (False, True)
    for dtypes in _FWD_BUF_DTYPES
] + [
    pytest.param(
        [256], 4, 2e-4, "all", True, False, False,
        torch.float32, torch.float32, id="long256",
    ),
]


@pytest.mark.gpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize(
    "segments, num_heads, fp32_rtol, fallback_pattern, use_raw_h0, "
    "reverse, transpose_m, h_dtype, m_dtype",
    _AGGREGATE_CASES,
)
def test_aggregate_card_state(
    segments, num_heads, fp32_rtol, state_v_first, fallback_pattern, use_raw_h0,
    reverse, transpose_m, h_dtype, m_dtype,
):
    scan_rtol = (fp32_rtol if h_dtype == m_dtype == torch.float32
                 and ARCH in ("SM90", "SM100", "SM103", "SM120") else LEGACY_SCAN_RTOL)
    H = num_heads
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first,
        fallback_pattern=fallback_pattern, seed=SEED + 2,
        near_identity=max(segments) >= 64,
    )
    ht, mt = ht.to(h_dtype), mt.to(m_dtype)
    raw_shape = (len(segments), H, V, K) if state_v_first else (len(segments), H, K, V)
    raw_h0 = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32) if use_raw_h0 else None

    # Dual CP only aggregates dh in backward and reuses the forward M product.
    # Hopper additionally retains its generic reverse-M aggregation interface.
    compute_m = not reverse or ARCH == "SM90"
    h_card, m_card = aggregate_card_state(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m, compute_m=compute_m,
    )
    _, ref_h, ref_m = ref_scan(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m,
    )
    err_h = rel_max(h_card, ref_h)
    assert err_h <= scan_rtol, f"h_card rel_max={err_h:.2e} > {scan_rtol:g}"
    if not compute_m:
        assert m_card is None
        return
    err_m = rel_max(m_card, ref_m)
    assert err_m <= scan_rtol, f"m_card rel_max={err_m:.2e} > {scan_rtol:g}"

    h_no_m, m_no_m = aggregate_card_state(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m, compute_m=False,
    )
    assert m_no_m is None
    assert rel_max(h_no_m, ref_h) <= scan_rtol


@pytest.mark.gpu
@pytest.mark.parametrize(
    "is_zigzag, num_heads",
    [
        pytest.param(False, 1, id="boundary-h1"),
        pytest.param(False, 32, id="boundary-h32"),
        pytest.param(False, 64, id="boundary-h64"),
        pytest.param(False, 128, id="boundary-h128"),
        pytest.param(True, 1, id="zigzag-h1"),
        pytest.param(True, 16, id="zigzag-h16"),
        pytest.param(True, 32, id="zigzag-h32"),
        pytest.param(True, 64, id="zigzag-h64"),
    ],
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("reverse", [False, True], ids=["fwd", "bwd"])
@pytest.mark.parametrize("use_raw_state", [False, True], ids=["zero-seed", "raw-seed"])
@pytest.mark.parametrize("packed", [False, True], ids=["split-mixed", "packed-fp32"])
@pytest.mark.parametrize("fallback_pattern", ["all", "random"])
def test_correct_states_slots_precision(
    is_zigzag, num_heads, state_v_first, reverse, use_raw_state, packed, fallback_pattern,
):
    """Indirect reads and sparse/scattered writes against a logical-order fp64 scan.

    The h/M rows deliberately have unrelated addresses in the packed gather. The split
    mixed-dtype case exercises the fallback used by older state producers.
    """
    if not _CP_SUPPORTS_SLOT_SCAN:
        pytest.skip("slot scan not built for this architecture")
    H, n_cp = num_heads, 8
    seq_map = torch.tensor([0, 3, 8] if is_zigzag else [0, 8], dtype=torch.int32, device=DEVICE)
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first, fallback_pattern=fallback_pattern,
    )
    if packed:
        h_slot = torch.tensor([11, 0, 9, 6, 15, 2, 13, 4], dtype=torch.int32, device=DEVICE)
        m_slot = torch.tensor([10, 1, 8, 7, 14, 3, 12, 5], dtype=torch.int32, device=DEVICE)
        h_buffer = torch.empty((2 * n_cp, H, K, V), device=DEVICE)
        m_buffer = h_buffer
    else:
        h_slot = torch.tensor([5, 0, 4, 3, 7, 1, 6, 2], dtype=torch.int32, device=DEVICE)
        m_slot = h_slot
        h_buffer = torch.empty_like(ht)
        mt = mt.bfloat16()
        m_buffer = torch.empty_like(mt)
    h_buffer[h_slot.long()] = ht
    m_buffer[m_slot.long()] = mt
    if is_zigzag:
        out_slot = torch.tensor([-1, 3, 0, -1, 2, -1, 1, -1], dtype=torch.int32, device=DEVICE)
        n_seqs, output_rows = 2, 4
    else:
        out_slot = torch.full((n_cp,), -1, dtype=torch.int32, device=DEVICE)
        out_slot[0 if reverse else -1] = 0
        n_seqs, output_rows = 1, 1
    raw = torch.randn((n_seqs, H, K, V), device=DEVICE) * 0.1 if use_raw_state else None
    # A supplied mask is indexed by physical h rows, independently of m/out slots.
    mask = None
    if fallback_pattern != "all":
        mask = torch.ones(h_buffer.shape[:2], dtype=torch.bool, device=DEVICE)
        mask[h_slot.long()] = fallback
    wrapper = correct_terminal_states if reverse else correct_initial_states
    out = wrapper(
        raw, h_buffer, m_buffer, mask, seq_map, state_v_first=state_v_first,
        h_slot=h_slot, m_slot=m_slot, out_slot=out_slot, is_zigzag=is_zigzag,
    )
    ref, _, _ = ref_scan(
        ht, mt, fallback, seq_map, raw_h0=raw, state_v_first=state_v_first,
        reverse=reverse, transpose_m=reverse,
    )
    expected = torch.empty((output_rows, H, K, V), device=DEVICE, dtype=torch.float64)
    owned = out_slot >= 0
    expected[out_slot[owned].long()] = ref[owned]
    if not is_zigzag:
        expected = expected[0]
    assert out.shape == expected.shape
    assert out.dtype == torch.float32
    err = rel_max(out, expected)
    assert err <= SCAN_RTOL, f"slots rel_max={err:.2e} > {SCAN_RTOL:g}"


# ===========================================================================
# correct_{initial,terminal}_states_zigzag -- the fused inter fold
# ===========================================================================
# The fused kernel scans straight out of the all-gather buffer via slot maps instead of
# requiring a physically reindexed copy. It does the same arithmetic in the same order as
# `_zigzag_reindex` + the plain fold, so the two must agree BITWISE -- a tolerance here
# would hide exactly the indexing mistakes this test exists to catch.
_ZIGZAG_SHAPES = [
    pytest.param(2, 1, id="cp2-d1"),    # single sequence: the case the perf work targets
    pytest.param(4, 2, id="cp4-d2"),
    pytest.param(8, 3, id="cp8-d3"),
]


def _zigzag_maps(cp, D, rank, device):
    """Mirrors context.py:_calc_inter_cp_seqs_zigzag. Kept as an independent
    transcription so a bug there cannot cancel itself out in this test."""
    NC, n_local = 2 * cp, 2 * D
    stride = 2 * n_local
    c2g, h_slot, m_slot, split_slot, out_slot = [], [], [], [], []
    for d in range(D):
        for c in range(NC):
            rank_g, local_seg = (c, d) if c < cp else (NC - 1 - c, D + d)
            c2g.append(rank_g * n_local + local_seg)
            h_slot.append(rank_g * stride + local_seg)
            m_slot.append(rank_g * stride + n_local + local_seg)
            split_slot.append(rank_g * n_local + local_seg)
            out_slot.append(local_seg if rank_g == rank else -1)
    owned = [d * NC + rank for d in range(D)] + [d * NC + (NC - 1 - rank) for d in range(D)]

    def i32(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    return {
        "c2g": torch.tensor(c2g, dtype=torch.long, device=device),
        "owned": torch.tensor(owned, dtype=torch.long, device=device),
        "h_slot": i32(h_slot), "m_slot": i32(m_slot), "split_slot": i32(split_slot),
        "out_slot": i32(out_slot),
        "fold_seq_map": torch.arange(0, D * NC + 1, NC, dtype=torch.int32, device=device),
    }


@pytest.mark.gpu
@pytest.mark.parametrize("cp, D", _ZIGZAG_SHAPES)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_state", [False, True], ids=["zero_seed", "raw_seed"])
@pytest.mark.parametrize("reverse", [False, True], ids=["fwd", "bwd"])
@pytest.mark.parametrize(
    "h_dtype, m_dtype",
    [
        pytest.param(torch.bfloat16, torch.bfloat16, id="prod-fwd"),
        # the backward's real pairing: fp32 dh against the bf16 M cached in the forward.
        # dtypes differ, so h and M cannot ride one cat-packed tensor.
        pytest.param(torch.float32, torch.bfloat16, id="prod-bwd"),
    ],
)
def test_correct_states_zigzag(cp, D, state_v_first, use_raw_state, reverse,
                               h_dtype, m_dtype):
    if not _CP_SUPPORTS_SLOT_SCAN:
        pytest.skip("slot scan not built for this architecture")

    torch.manual_seed(SEED + cp * 100 + D)
    H, rank = 4, 1
    NC, n_local = 2 * cp, 2 * D
    maps = _zigzag_maps(cp, D, rank, DEVICE)
    state_shape = (V, K) if state_v_first else (K, V)

    # per-rank payloads, exactly as each card would hand them to the collective
    h_rank = torch.randn(cp, n_local, H, *state_shape, device=DEVICE).to(h_dtype)
    m_rank = torch.randn(cp, n_local, H, K, K, device=DEVICE).to(m_dtype)
    raw_shape = (D, H, *state_shape)
    raw = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32) if use_raw_state else None

    correct = correct_terminal_states if reverse else correct_initial_states

    # --- reference: reindex into doc-chunk order, fold, then pull out owned rows ---
    h_all = h_rank.reshape(cp * n_local, H, *state_shape)
    m_all = m_rank.reshape(cp * n_local, H, K, K)
    h_buf = h_all.index_select(0, maps["c2g"]).contiguous()
    m_buf = m_all.index_select(0, maps["c2g"]).contiguous()
    fb = torch.ones((D * NC, H), dtype=torch.bool, device=DEVICE)
    kw = dict(fallback_mask=fb, seq_map_r2c=maps["fold_seq_map"],
              state_v_first=state_v_first)
    if reverse:
        ref_all = correct(raw_dht=raw, dht_buffer=h_buf, mt_buffer=m_buf, **kw)
    else:
        ref_all = correct(raw_h0=raw, ht_buffer=h_buf, mt_buffer=m_buf, **kw)
    ref = ref_all.index_select(0, maps["owned"]).contiguous()

    # --- fused, two collectives: h and M gathered apart, one map serves both ---
    got_split = correct(
        raw, h_all, m_all, None, maps["fold_seq_map"], state_v_first=state_v_first,
        h_slot=maps["split_slot"], m_slot=maps["split_slot"],
        out_slot=maps["out_slot"], is_zigzag=True,
    )
    assert got_split.shape == ref.shape, (got_split.shape, ref.shape)
    assert torch.equal(got_split, ref), (
        f"split-gather fold differs from reindex+fold: "
        f"max |delta|={(got_split.float() - ref.float()).abs().max():.3e}"
    )

    # --- fused, one collective: cat([h, M]) per rank, two maps into one tensor ---
    if h_dtype != m_dtype:
        return   # cannot cat; the split path above is what production uses here
    packed = torch.cat([h_rank, m_rank], dim=1).reshape(cp * 2 * n_local, H, *state_shape)
    got_cat = correct(
        raw, packed, packed, None, maps["fold_seq_map"], state_v_first=state_v_first,
        h_slot=maps["h_slot"], m_slot=maps["m_slot"],
        out_slot=maps["out_slot"], is_zigzag=True,
    )
    assert torch.equal(got_cat, ref), (
        f"cat-packed fold differs from reindex+fold: "
        f"max |delta|={(got_cat.float() - ref.float()).abs().max():.3e}"
    )
