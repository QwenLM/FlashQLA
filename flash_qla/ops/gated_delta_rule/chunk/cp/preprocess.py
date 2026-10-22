# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .comm import all_gather_into_tensor, pack_hm, pack_states, unpack_hm

@dataclass
class CPCache:
    h0: torch.Tensor | None = None            # fwd-corrected initial state fed to the main kernel
    mt: torch.Tensor | None = None            # per-chunk M (is_intra)
    m_first: torch.Tensor | None = None       # per-seq first-seq M, for the bwd inter gather (is_inter)
    fallback_bwd: torch.Tensor | None = None      # pure-intra bwd fallback mask
    num_warmup_bwd: torch.Tensor | None = None    # pure-intra bwd warmup counts


def _assert_inter_intra_supported(is_inter, is_intra, aggregate_card_state):
    """Combined inter+intra CP needs ``aggregate_card_state``, which only the
    SM100/SM103 kernels provide so far. Fail loudly instead of on a ``None`` call."""
    if is_inter and is_intra and aggregate_card_state is None:
        raise NotImplementedError(
            "Combined inter+intra CP requires `aggregate_card_state`, which is not "
            "implemented for this architecture (SM100/SM103 only). Use pure inter-card "
            "CP (build_cp_context(..., enable_inter=True)) or pure intra-card CP instead."
        )


# ---------------------------------------------------------------------------
# shared inter correction: gather this card's boundary (h, M) with its neighbours and
# correct it into a per-seq card state (fwd) / terminal grad (bwd).
# ---------------------------------------------------------------------------
def _inter_correct_fwd(h_seq, m_seq, initial_state, cp_context, state_v_first):
    """Gather per-seq (h, M) over the ``pre`` cards sharing this card's first local
    sequence and correct it; returns the per-seq card_h0 with seq[0] corrected (seq[1:]
    keep ``initial_state``). On the first rank the card_h0 is just ``initial_state``."""
    from flash_qla.ops.gated_delta_rule.chunk import correct_initial_states

    Hv = h_seq.shape[1]
    hm = pack_hm(h_seq[-1], m_seq[-1])
    ag_hm, _ = all_gather_into_tensor(hm, group=cp_context.group)
    rank = dist.get_rank(group=cp_context.group)
    pre = cp_context.pre_num_ranks
    h_buf, m_buf = unpack_hm(ag_hm[rank - pre: rank + 1], h_seq[-1], m_seq[-1])

    if cp_context.is_first_rank:
        return initial_state

    seq_map, fb_mask = cp_context.get_fwd_scan_tensors(Hv)
    cp_h0_all = correct_initial_states(
        raw_h0=None, ht_buffer=h_buf, mt_buffer=m_buf,
        fallback_mask=fb_mask, seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    if initial_state is None:
        card_h0 = torch.zeros(h_seq.shape, dtype=torch.float32, device=h_seq.device)
    else:
        card_h0 = initial_state.clone()
    card_h0[0] = cp_h0_all[h_buf.shape[0] - 1]
    return card_h0


def _inter_correct_bwd(dh_seq, m_first, dht, cp_context, state_v_first):
    """Gather per-seq dh over the ``post`` cards sharing this card's last local
    sequence and correct it; returns the per-seq terminal grad with seq[-1] set from
    the following cards (or ``dht`` unchanged when there is nothing to correct)."""
    from flash_qla.ops.gated_delta_rule.chunk import correct_terminal_states

    Hv = dh_seq.shape[1]
    N = dh_seq.shape[0]
    hm = pack_hm(dh_seq[0], m_first)
    ag_hm, _ = all_gather_into_tensor(hm, group=cp_context.group)
    rank = dist.get_rank(group=cp_context.group)
    post = cp_context.post_num_ranks
    dh_buf, m_buf = unpack_hm(ag_hm[rank: rank + 1 + post], dh_seq[0], m_first)

    if dht is None and cp_context.is_last_rank:
        return None
    card_dht = torch.zeros(dh_seq.shape, dtype=torch.float32, device=dh_seq.device) if dht is None else dht.clone()
    if not cp_context.is_last_rank:
        seq_map, fb_mask = cp_context.get_bwd_scan_tensors(Hv)
        cp_dht_all = correct_terminal_states(
            raw_dht=None, dht_buffer=dh_buf, mt_buffer=m_buf,
            fallback_mask=fb_mask, seq_map_r2c=seq_map, state_v_first=state_v_first,
        )
        card_dht[N - 1] = cp_dht_all[0]
    return card_dht


# ---------------------------------------------------------------------------
# zigzag inter correction: full all-gather of per-owned-chunk (h, M), reindex into
# per-document chunk order, fold each document, extract this rank's owned chunks.
# ---------------------------------------------------------------------------
def _zigzag_reindex(packed_a, packed_b, cp_context):
    """All-gather the packed (a, b) per-owned-chunk pair, unpack, flatten to
    [world*n_local, ...] and index_select into per-document chunk order."""
    hm = pack_hm(packed_a, packed_b)
    ag_hm, _ = all_gather_into_tensor(hm, group=cp_context.group)
    a_all, b_all = unpack_hm(ag_hm, packed_a, packed_b)
    a_flat = a_all.reshape(-1, *packed_a.shape[1:])
    b_flat = b_all.reshape(-1, *packed_b.shape[1:])
    slot = cp_context.chunk_to_gathered_slot
    return a_flat.index_select(0, slot).contiguous(), b_flat.index_select(0, slot).contiguous()


def _zigzag_gather(packed_a, packed_b, cp_context):
    """All-gather (a, b) for the fused fold. No unpack, no reindex.

    Returns ``(a_buf, b_buf, a_slot, b_slot)``; the fold kernel reads both buffers
    through the slot maps, so whatever layout NCCL produced is left alone.

    Two shapes, picked by dtype:

    * **same dtype** (forward: both come from ``fused_gdr_h``, or both from the same
      ``aggregate_card_state``) -- row-stack into one tensor and issue a single
      collective. ``fold_h_slot`` / ``fold_m_slot`` address the two row ranges.
    * **different dtypes** (backward: ``dh`` is fp32 from ``fused_gdr_dh`` while ``M``
      is the bf16 one cached in the forward) -- one tensor cannot hold both, so gather
      them apart. Each buffer is then plain ``[world*n_local, ...]`` and the single
      ``fold_split_slot`` serves both.

    Either way the fold runs straight off the gathered buffer; the second collective is
    the only difference, and it costs far less than the copies it replaces.
    """
    if packed_a.dtype == packed_b.dtype:
        buf = pack_states(packed_a, packed_b)
        ag, _ = all_gather_into_tensor(buf, group=cp_context.group)
        flat = ag.reshape(-1, *buf.shape[1:])
        return flat, flat, cp_context.fold_h_slot, cp_context.fold_m_slot

    a_all, _ = all_gather_into_tensor(packed_a, group=cp_context.group)
    b_all, _ = all_gather_into_tensor(packed_b, group=cp_context.group)
    slot = cp_context.fold_split_slot
    return (a_all.reshape(-1, *packed_a.shape[1:]),
            b_all.reshape(-1, *packed_b.shape[1:]), slot, slot)


def _inter_correct_fwd_zigzag(ht, mt, initial_state, cp_context, state_v_first):
    from flash_qla.ops.gated_delta_rule.chunk import (
        correct_initial_states, correct_initial_states_zigzag,
    )

    if correct_initial_states_zigzag is None:      # sm120: no fused fold, two-step path
        H = ht.shape[1]
        h_buf, m_buf = _zigzag_reindex(ht, mt, cp_context)
        fb = torch.ones((h_buf.shape[0], H), dtype=torch.bool, device=ht.device)
        cp_h0_all = correct_initial_states(
            raw_h0=initial_state, ht_buffer=h_buf, mt_buffer=m_buf,
            fallback_mask=fb, seq_map_r2c=cp_context.fold_seq_map,
            state_v_first=state_v_first,
        )
        return cp_h0_all.index_select(0, cp_context.owned_fold_slot).contiguous()

    h_buf, m_buf, h_slot, m_slot = _zigzag_gather(ht, mt, cp_context)
    return correct_initial_states_zigzag(
        initial_state, h_buf, m_buf, cp_context.fold_seq_map,
        h_slot, m_slot, cp_context.fold_out_slot,
        n_local=ht.shape[0], state_v_first=state_v_first,
    )


def _inter_correct_bwd_zigzag(dh, mt, dht, cp_context, state_v_first):
    from flash_qla.ops.gated_delta_rule.chunk import (
        correct_terminal_states, correct_terminal_states_zigzag,
    )

    if correct_terminal_states_zigzag is None:     # sm120: no fused fold, two-step path
        H = dh.shape[1]
        dh_buf, m_buf = _zigzag_reindex(dh, mt, cp_context)
        fb = torch.ones((dh_buf.shape[0], H), dtype=torch.bool, device=dh.device)
        cp_dht_all = correct_terminal_states(
            raw_dht=dht, dht_buffer=dh_buf, mt_buffer=m_buf,
            fallback_mask=fb, seq_map_r2c=cp_context.fold_seq_map,
            state_v_first=state_v_first,
        )
        return cp_dht_all.index_select(0, cp_context.owned_fold_slot).contiguous()

    dh_buf, m_buf, h_slot, m_slot = _zigzag_gather(dh, mt, cp_context)
    return correct_terminal_states_zigzag(
        dht, dh_buf, m_buf, cp_context.fold_seq_map,
        h_slot, m_slot, cp_context.fold_out_slot,
        n_local=dh.shape[0], state_v_first=state_v_first,
    )


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def cp_preprocess_fwd(
    cp_context,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    enable_fwd_cp_cache: bool,
):
    from flash_qla.ops.gated_delta_rule.chunk import (
        fused_gdr_h, correct_initial_states, aggregate_card_state,
        get_warmup_chunks, get_warmup_chunks_bidi,
    )

    is_inter, is_intra = cp_context.is_inter, cp_context.is_intra
    if not is_inter and not is_intra:
        return initial_state, None
    _assert_inter_intra_supported(is_inter, is_intra, aggregate_card_state)

    Hv = v.shape[2]
    chunk_size = a.shape[-1]
    fallback = num_warmup_bwd = fallback_bwd = None

    # --- zigzag inter ---
    if cp_context.is_zigzag:
        if is_intra:
            # intra occupancy split on top of zigzag ownership. Warmup/fallback are set
            # conservatively full (no gate-decay pruning yet) -- correct but not optimal.
            cp_cu = cp_context.intra_cp_cu_seqlens
            seq_map_r2c = cp_context.seq_map_r2c
            num_warmup = cp_context.get_full_warmup(Hv, chunk_size)
            fb_part = torch.ones((cp_cu.numel() - 1, Hv), dtype=torch.bool, device=k.device)
            _, ht, mt = fused_gdr_h(
                k=k, v=v, a=a, g=g, b=beta, initial_state=None,
                output_final_state=True, output_h=False, cu_seqlens=cp_cu,
                num_warmup_chunks=num_warmup, state_v_first=state_v_first,
            )
            # collapse each owned chunk's partitions into one (h, M).
            h_seq, m_seq = aggregate_card_state(
                ht, mt, fb_part, seq_map_r2c, raw_h0=None,
                state_v_first=state_v_first, compute_m=True,
            )
            # inter zigzag fold -> per-owned-chunk entering state.
            card_h0 = _inter_correct_fwd_zigzag(h_seq, m_seq, initial_state, cp_context, state_v_first)
            # scan the corrected per-owned-chunk h0 back into each intra partition.
            cp_h0 = correct_initial_states(
                raw_h0=card_h0, ht_buffer=ht, mt_buffer=mt,
                fallback_mask=fb_part, seq_map_r2c=seq_map_r2c, state_v_first=state_v_first,
            )
            return cp_h0, CPCache(h0=cp_h0, mt=mt, m_first=m_seq)

        cu = cp_context.cu_seqlens
        num_warmup = cp_context.get_full_warmup(Hv, chunk_size)
        # stage 1: per-owned-chunk (he, M) from a zero in-state.
        _, ht, mt = fused_gdr_h(
            k=k, v=v, a=a, g=g, b=beta, initial_state=None,
            output_final_state=True, output_h=False, cu_seqlens=cu,
            num_warmup_chunks=num_warmup, state_v_first=state_v_first,
        )
        cp_h0 = _inter_correct_fwd_zigzag(ht, mt, initial_state, cp_context, state_v_first)
        # cache the forward per-chunk M for the backward reverse fold.
        return cp_h0, CPCache(h0=cp_h0, mt=mt)

    # --- Stage 1: prepare_h ---
    if is_intra:
        cp_cu = cp_context.intra_cp_cu_seqlens
        seq_map_r2c = cp_context.seq_map_r2c
        if enable_fwd_cp_cache or is_inter:
            num_warmup, num_warmup_bwd, fallback, fallback_bwd = get_warmup_chunks_bidi(
                g=g, cu_seqlens=cp_cu, ht_mask_fwd=cp_context.ht_mask,
                ht_mask_bwd=cp_context.ht_mask_bwd, chunk_size=chunk_size,
                seq_map_r2c=seq_map_r2c, force_inter_boundaries=is_inter,
            )
        else:
            num_warmup, fallback = get_warmup_chunks(
                g=g, cu_seqlens=cp_cu, ht_mask=cp_context.ht_mask, chunk_size=chunk_size,
            )
        _, ht, mt = fused_gdr_h(
            k=k, v=v, a=a, g=g, b=beta, initial_state=None,
            output_final_state=True, output_h=False, cu_seqlens=cp_cu,
            num_warmup_chunks=num_warmup, state_v_first=state_v_first,
        )
    else:
        cu = cp_context.cu_seqlens
        cu_cpu = cp_context.cu_seqlens_cpu.tolist()
        N = len(cu_cpu) - 1
        num_warmup = torch.zeros((N, Hv), dtype=cu.dtype, device=k.device)
        num_warmup[-1, :] = (cu_cpu[-1] - cu_cpu[-2] + chunk_size - 1) // chunk_size
        if N > 1:
            num_warmup[0, :] = (cu_cpu[1] - cu_cpu[0] + chunk_size - 1) // chunk_size
        _, ht, mt = fused_gdr_h(
            k=k, v=v, a=a, g=g, b=beta, initial_state=initial_state,
            output_final_state=True, output_h=False, cu_seqlens=cu,
            num_warmup_chunks=num_warmup, state_v_first=state_v_first,
        )

    # --- Stage 2: inter correction ---
    if is_inter:
        if is_intra:
            raw_h0 = initial_state
            if raw_h0 is not None and not cp_context.is_first_rank:
                raw_h0 = raw_h0.clone()
                raw_h0[0] = 0
            h_seq, m_seq = aggregate_card_state(
                ht, mt, fallback, seq_map_r2c, raw_h0=raw_h0,
                state_v_first=state_v_first, compute_m=True,
            )
        else:
            h_seq, m_seq = ht, mt
        card_h0 = _inter_correct_fwd(h_seq, m_seq, initial_state, cp_context, state_v_first)
    else:
        card_h0 = initial_state

    # --- Stage 3: intra correct ---
    if is_intra:
        cp_h0 = correct_initial_states(
            raw_h0=card_h0, ht_buffer=ht, mt_buffer=mt,
            fallback_mask=fallback, seq_map_r2c=seq_map_r2c, state_v_first=state_v_first,
        )
    else:
        cp_h0 = card_h0

    if is_intra and not is_inter and not enable_fwd_cp_cache:
        return cp_h0, None   # pure intra, uncached -> backward recomputes

    cp_cache = CPCache(
        h0=cp_h0,
        mt=mt if is_intra else None,
        m_first=m_seq[0] if is_inter else None,
        fallback_bwd=fallback_bwd,
        num_warmup_bwd=num_warmup_bwd,
    )
    return cp_h0, cp_cache


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------
def cp_preprocess_bwd(
    cp_context,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    cp_cache,
):
    from flash_qla.ops.gated_delta_rule.chunk import (
        fused_gdr_h, fused_gdr_dh, correct_initial_states, correct_terminal_states,
        aggregate_card_state, get_warmup_chunks_bidi,
    )

    is_inter, is_intra = cp_context.is_inter, cp_context.is_intra
    if not is_inter and not is_intra:
        return initial_state, dht
    _assert_inter_intra_supported(is_inter, is_intra, aggregate_card_state)

    if fused_gdr_dh is None:
        raise NotImplementedError(
            "Backward pass (CP) is not implemented for SM120 (Blackwell). "
            "Only forward pass is supported on this architecture."
        )

    Hv = v.shape[2]
    chunk_size = a.shape[-1]
    mt = fallback = None

    # --- zigzag inter ---
    if cp_context.is_zigzag:
        assert dht is None, (
            "zigzag CP does not support a terminal-state gradient yet; the per-segment "
            "final state is not the document-final state. Use output_final_state=False."
        )
        cp_h0 = cp_cache.h0
        if is_intra:
            cp_cu = cp_context.intra_cp_cu_seqlens
            seq_map_r2c = cp_context.seq_map_r2c
            mt = cp_cache.mt            # forward per-partition M (stage-3 reverse fold)
            m_seq = cp_cache.m_first    # forward per-owned-chunk M (inter reverse fold)
            num_warmup = cp_context.get_full_warmup(Hv, chunk_size)
            fb_part = torch.ones((cp_cu.numel() - 1, Hv), dtype=torch.bool, device=k.device)
            # stage 1: per-partition dh from a zero terminal grad.
            _, dh = fused_gdr_dh(
                q=q, k=k, a=a, g=g, b=beta, do=do, dht=None,
                output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cp_cu,
                num_warmup_chunks=num_warmup, state_v_first=state_v_first,
            )
            # collapse partitions -> per-owned-chunk dh (reverse).
            dh_seq, _ = aggregate_card_state(
                dh, mt, fb_part, seq_map_r2c, raw_h0=None,
                state_v_first=state_v_first, reverse=True, transpose_m=True, compute_m=False,
            )
            # inter reverse fold with the forward per-owned-chunk M.
            card_dht = _inter_correct_bwd_zigzag(dh_seq, m_seq, dht, cp_context, state_v_first)
            # scan the corrected terminal grad back into each partition.
            cp_dht = correct_terminal_states(
                raw_dht=card_dht, dht_buffer=dh, mt_buffer=mt,
                fallback_mask=fb_part, seq_map_r2c=seq_map_r2c, state_v_first=state_v_first,
            )
            return cp_h0, cp_dht

        cu = cp_context.cu_seqlens
        num_warmup = cp_context.get_full_warmup(Hv, chunk_size)
        # stage 1: per-owned-chunk dh from a zero terminal grad.
        _, dh = fused_gdr_dh(
            q=q, k=k, a=a, g=g, b=beta, do=do, dht=None,
            output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cu,
            num_warmup_chunks=num_warmup, state_v_first=state_v_first,
        )
        # reverse-fold using the forward per-chunk M (cached).
        cp_dht = _inter_correct_bwd_zigzag(dh, cp_cache.mt, dht, cp_context, state_v_first)
        return cp_h0, cp_dht

    # --- Stage 1: prepare_dh (recover the fwd-cached h0/mt first) ---
    if is_intra:
        cp_cu = cp_context.intra_cp_cu_seqlens
        seq_map_r2c = cp_context.seq_map_r2c
        # inter+intra has no recompute path -- the fwd cache is mandatory there.
        use_cache = is_inter or cp_cache is not None
        if use_cache and not is_inter:
            # The cache was sized by the forward partitioning; everything below indexes
            # it against this pass's. A mismatch reads past the end of h0 / mt (tilelang
            # does no bounds checking), so drop the stale cache and rebuild.
            n_part = cp_cu.numel() - 1
            if cp_cache.h0.shape[0] != n_part or cp_cache.mt.shape[0] != n_part:
                warnings.warn(
                    "forward CP cache does not match the backward partitioning "
                    f"(h0 {cp_cache.h0.shape[0]} rows, mt {cp_cache.mt.shape[0]}, "
                    f"pass has {n_part} CP partitions); ignoring the cache and "
                    "recomputing.",
                    RuntimeWarning, stacklevel=2,
                )
                use_cache = False
        if use_cache:
            cp_h0, mt = cp_cache.h0, cp_cache.mt
            num_warmup, fallback = cp_cache.num_warmup_bwd, cp_cache.fallback_bwd
        else:
            # pure intra, uncached -> recompute fwd h0/mt + bwd warmup.
            num_warmup_h, num_warmup, fallback_fwd, fallback = get_warmup_chunks_bidi(
                g=g, cu_seqlens=cp_cu, ht_mask_fwd=cp_context.ht_mask,
                ht_mask_bwd=cp_context.ht_mask_bwd, chunk_size=chunk_size,
            )
            _, ht_b, mt = fused_gdr_h(
                k=k, v=v, a=a, g=g, b=beta, initial_state=None,
                output_final_state=True, output_h=False, cu_seqlens=cp_cu,
                num_warmup_chunks=num_warmup_h, state_v_first=state_v_first,
            )
            cp_h0 = correct_initial_states(
                raw_h0=initial_state, ht_buffer=ht_b, mt_buffer=mt,
                fallback_mask=fallback_fwd, seq_map_r2c=seq_map_r2c, state_v_first=state_v_first,
            )
        _, dh = fused_gdr_dh(
            q=q, k=k, a=a, g=g, b=beta, do=do, dht=None,
            output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cp_cu,
            num_warmup_chunks=num_warmup, state_v_first=state_v_first,
        )
    else:
        cu = cp_context.cu_seqlens
        cu_cpu = cp_context.cu_seqlens_cpu.tolist()
        N = len(cu_cpu) - 1
        cp_h0 = cp_cache.h0
        num_warmup = torch.zeros((N, Hv), dtype=cu.dtype, device=k.device)
        num_warmup[0, :] = (cu_cpu[1] - cu_cpu[0] + chunk_size - 1) // chunk_size
        _, dh = fused_gdr_dh(
            q=q, k=k, a=a, g=g, b=beta, do=do, dht=dht,
            output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cu,
            num_warmup_chunks=num_warmup, state_v_first=state_v_first,
        )

    # --- Stage 2: inter correction ---
    if is_inter:
        if is_intra:
            raw_dht = dht
            if raw_dht is not None and not cp_context.is_last_rank:
                raw_dht = raw_dht.clone()
                raw_dht[-1] = 0
            # aggregate (reverse) -> per-seq dh; m_card is the fwd M product (cached).
            dh_seq, _ = aggregate_card_state(
                dh, mt, fallback, seq_map_r2c, raw_h0=raw_dht,
                state_v_first=state_v_first, reverse=True, transpose_m=True, compute_m=False,
            )
        else:
            dh_seq = dh
        card_dht = _inter_correct_bwd(dh_seq, cp_cache.m_first, dht, cp_context, state_v_first)
    else:
        card_dht = dht

    # --- Stage 3: intra correct ---
    if is_intra:
        cp_dht = correct_terminal_states(
            raw_dht=card_dht, dht_buffer=dh, mt_buffer=mt,
            fallback_mask=fallback, seq_map_r2c=seq_map_r2c, state_v_first=state_v_first,
        )
    else:
        cp_dht = card_dht

    return cp_h0, cp_dht
