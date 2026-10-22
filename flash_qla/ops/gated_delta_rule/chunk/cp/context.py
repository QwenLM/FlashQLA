# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import tilelang

from flash_qla.utils import tensor_cache
from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE

from .autocp import current_arch, is_calibrated
from .autocp import decide as autocp_decide

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

ARCH = current_arch()

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count


@dataclass
class FlashQLACPContext:
    # Two orthogonal flags select the CP mode:
    #   (is_inter, is_intra) = (T,F) pure inter | (F,T) pure intra |
    #   (T,T) inter+intra | (F,F) no CP. An inter+intra card whose intra split
    #   did not trigger degenerates to (T,F), i.e. behaves exactly like pure inter.
    is_inter: bool = False
    is_intra: bool = False

    # --- common ---
    cu_seqlens: torch.Tensor | None = None

    # --- inter-card ---
    group: "ProcessGroup | None" = None
    cu_seqlens_cpu: torch.Tensor | None = None 
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    is_first_rank: bool | None = None
    post_num_ranks: int | None = None
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None

    # --- inter-card zigzag layout ---
    # zigzag spreads a document's NC=2*cp chunks non-contiguously: rank r owns chunk r
    # and chunk NC-1-r of every doc. The fold then runs per document in chunk order over
    # the full all-gathered (h, M) buffer, reusing correct_initial_states. All tensors are
    # built in build_cp_context(zigzag=True); the framework only feeds each rank its chunks
    # in the agreed [early-halves d=0..D-1, late-halves d=0..D-1] layout.
    is_zigzag: bool = False
    num_docs: int = 0                                        # D documents in the pack
    num_chunks: int = 0                                      # NC = 2 * cp_size
    chunk_to_gathered_slot: torch.Tensor | None = None       # [D*NC] doc-chunk slot -> gather index
    fold_seq_map: torch.Tensor | None = None                 # [D+1] per-doc groups for the fold
    owned_fold_slot: torch.Tensor | None = None              # [n_local] local seg -> fold output slot
    # fused fold (see _calc_inter_cp_seqs_zigzag): address the gathered cat([h, M])
    # buffer directly, so no reindex is materialised before the scan.
    fold_h_slot: torch.Tensor | None = None                  # [D*NC] -> gathered h row
    fold_m_slot: torch.Tensor | None = None                  # [D*NC] -> gathered M row
    fold_out_slot: torch.Tensor | None = None                # [D*NC] -> output row, -1 = skip
    fold_split_slot: torch.Tensor | None = None              # [D*NC] -> row, h/M gathered apart

    # --- intra-card ---
    intra_cp_cu_seqlens: torch.Tensor | None = None
    intra_cp_cu_seqlens_cpu: torch.Tensor | None = None
    seq_map_r2c: torch.Tensor | None = None
    seq_map_c2r: torch.Tensor | None = None
    ht_mask: torch.Tensor | None = None
    ht_mask_bwd: torch.Tensor | None = None

    def copy_for_backward(self) -> "FlashQLACPContext":
        copied = FlashQLACPContext(
            is_inter=self.is_inter,
            is_intra=self.is_intra,
            group=self.group,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            cu_seqlens_cpu=self.cu_seqlens_cpu.clone() if self.cu_seqlens_cpu is not None else None,
            is_last_rank=self.is_last_rank,
            pre_num_ranks=self.pre_num_ranks,
            is_first_rank=self.is_first_rank,
            post_num_ranks=self.post_num_ranks,
            conv1d_kernel_size=self.conv1d_kernel_size,
            pre_num_conv_tokens=self.pre_num_conv_tokens,
            is_zigzag=self.is_zigzag,
            num_docs=self.num_docs,
            num_chunks=self.num_chunks,
        )
        # the fold reindex tensors are the same for fwd and bwd gathers; carry them.
        if self.is_zigzag:
            copied.chunk_to_gathered_slot = self.chunk_to_gathered_slot.clone()
            copied.fold_seq_map = self.fold_seq_map.clone()
            copied.owned_fold_slot = self.owned_fold_slot.clone()
            copied.fold_h_slot = self.fold_h_slot
            copied.fold_m_slot = self.fold_m_slot
            copied.fold_out_slot = self.fold_out_slot
            copied.fold_split_slot = self.fold_split_slot
            # shared, not cloned: it is the ``tensor_cache`` key behind
            # ``get_full_warmup``, so a fresh clone each step would miss every backward.
            copied.cu_seqlens_cpu = self.cu_seqlens_cpu
        # carry the intra-level partition tensors whenever the intra split is active.
        if self.is_intra:
            copied.intra_cp_cu_seqlens = (
                self.intra_cp_cu_seqlens.clone() if self.intra_cp_cu_seqlens is not None else None
            )
            copied.intra_cp_cu_seqlens_cpu = self.intra_cp_cu_seqlens_cpu
            copied.seq_map_r2c = self.seq_map_r2c.clone() if self.seq_map_r2c is not None else None
            copied.seq_map_c2r = self.seq_map_c2r.clone() if self.seq_map_c2r is not None else None
            copied.ht_mask = self.ht_mask.clone() if self.ht_mask is not None else None
            copied.ht_mask_bwd = self.ht_mask_bwd.clone() if self.ht_mask_bwd is not None else None
        return copied

    @property
    def num_seqs(self) -> int:
        return 0 if self.cu_seqlens is None else len(self.cu_seqlens) - 1

    def get_fwd_scan_tensors(self, num_v_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        dev_idx = self.cu_seqlens.device.index
        return (
            _create_scan_seq_map(self.pre_num_ranks, dev_idx),
            _create_scan_fb_mask(self.pre_num_ranks, num_v_heads, dev_idx),
        )

    def get_bwd_scan_tensors(self, num_v_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        dev_idx = self.cu_seqlens.device.index
        return (
            _create_scan_seq_map(self.post_num_ranks, dev_idx),
            _create_scan_fb_mask(self.post_num_ranks, num_v_heads, dev_idx),
        )

    def get_full_warmup(self, num_v_heads: int, chunk_size: int) -> torch.Tensor:
        """The zigzag path's ``[nseg, Hv]`` warmup-chunk count, over whichever
        segmentation the kernels are fed: the intra partitions when the intra split is
        active, else the owned chunks.

        Zigzag cannot use ``get_warmup_chunks_bidi``'s gate-decay pruning -- every owned
        chunk is a document-interior piece folded across cards, so all of them need the
        full per-segment M. The count is then a pure function of the segmentation, so
        ``_create_full_warmup`` caches it on the CPU segment tensor the context already
        holds. That keeps it off the per-step path, where it used to cost a blocking
        ``.cpu()`` D2H plus one kernel launch per segment, twice per training step.
        """
        if self.is_intra:
            cu, cu_cpu = self.intra_cp_cu_seqlens, self.intra_cp_cu_seqlens_cpu
        else:
            cu, cu_cpu = self.cu_seqlens, self.cu_seqlens_cpu
        if cu_cpu is None:
            cu_cpu = cu.cpu()
        return _create_full_warmup(
            cu_cpu, num_v_heads, chunk_size, cu.device.index, cu.dtype,
        )

def build_cp_context(
    cu_seqlens: torch.Tensor,
    group: "ProcessGroup | None" = None,
    enable_inter: bool = True,
    enable_intra: bool = True,
    num_v_heads: int | None = None,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    is_train: bool = True,
    force_intra_cp: bool = False,
    zigzag: bool = False,
) -> FlashQLACPContext:
    """Unified CP-context builder over the canonical ``(cu_seqlens, num_v_heads)``.
    ``enable_inter``/``enable_intra`` select the mode: (T,F) pure inter | (F,T) pure
    intra | (T,T) inter+intra | (F,F) no CP.

    - inter needs ``group``.
    - intra needs ``cu_seqlens`` (varlen) and ``num_v_heads``.
    - ``is_train`` picks the intra decision's cost regime. It keys off the *step*, not
      the pass: a training step's forward and backward must reach the same decision,
      otherwise the two passes disagree on the partitioning.
    - ``g`` lets the calibrated model see the per-head decay, so a gated/SWA head whose
      warmup covers only the recent chunks is costed as such. Only read on arches where
      :data:`AUTOCP_MODEL` holds; the heuristic ignores it. Opt-in, and not used by the
      chunk driver's automatic path: it joins the ``tensor_cache`` key, so a caller that
      passes a fresh ``g`` every step gets a miss every step -- one device sync per call
      to read the gate, and no CUDA graph capture (a miss under capture asserts).
    - ``force_intra_cp`` bypasses the intra decision and always splits, for tests and
      profiling that need the intra path on configurations it would skip.
    """

    if enable_inter and group is None:
        warnings.warn(
            "inter-card CP is enabled but no group is provided; disabling inter-card CP.",
            RuntimeWarning, stacklevel=2,
        )
        enable_inter = False
        
    if enable_intra and num_v_heads is None:
        warnings.warn(
            "intra-card CP is enabled but no num_v_heads is provided; disabling intra-card CP, which may lead to suboptimal performance.",
            RuntimeWarning, stacklevel=2,
        )
        enable_intra = False

    if zigzag:
        # zigzag is an inter-card layout. Short-conv halo (contiguous previous card) does
        # not hold under zigzag, so it is rejected for now.
        assert enable_inter and group is not None, "zigzag requires inter-card CP with a group"
        assert conv1d_kernel_size is None, "zigzag with short-conv halo is not implemented yet"
        inter_ctx = _calc_inter_cp_seqs_zigzag(
            cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group,
        )
        if not (enable_intra and num_v_heads is not None):
            return inter_ctx
        # intra splits each owned chunk (its own local segment) for occupancy; the inter
        # zigzag fold then runs at per-owned-chunk granularity on the aggregated states.
        intra_ctx = _calc_intra_cp_seqs(
            raw_cu_seqlens=inter_ctx.cu_seqlens, chunk_size=CHUNK_SIZE,
            num_v_heads=num_v_heads, g=g if AUTOCP_MODEL else None,
            is_train=is_train, force_intra_cp=force_intra_cp,
        )
        return FlashQLACPContext(
            is_inter=True,
            is_intra=intra_ctx.is_intra,
            is_zigzag=True,
            group=group,
            cu_seqlens=inter_ctx.cu_seqlens,
            cu_seqlens_cpu=inter_ctx.cu_seqlens_cpu,
            is_first_rank=inter_ctx.is_first_rank,
            is_last_rank=inter_ctx.is_last_rank,
            num_docs=inter_ctx.num_docs,
            num_chunks=inter_ctx.num_chunks,
            chunk_to_gathered_slot=inter_ctx.chunk_to_gathered_slot,
            fold_seq_map=inter_ctx.fold_seq_map,
            owned_fold_slot=inter_ctx.owned_fold_slot,
            fold_h_slot=inter_ctx.fold_h_slot,
            fold_m_slot=inter_ctx.fold_m_slot,
            fold_out_slot=inter_ctx.fold_out_slot,
            fold_split_slot=inter_ctx.fold_split_slot,
            intra_cp_cu_seqlens=intra_ctx.intra_cp_cu_seqlens,
            intra_cp_cu_seqlens_cpu=intra_ctx.intra_cp_cu_seqlens_cpu,
            seq_map_r2c=intra_ctx.seq_map_r2c,
            seq_map_c2r=intra_ctx.seq_map_c2r,
            ht_mask=intra_ctx.ht_mask,
            ht_mask_bwd=intra_ctx.ht_mask_bwd,
        )

    if enable_inter and enable_intra:
        assert group is not None and num_v_heads is not None
        # inter split -> local per-card cu_seqlens + rank topology; then intra split
        # over that local sequence; merge both field groups into one context.
        inter_ctx = _calc_inter_cp_seqs(
            cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group,
            conv1d_kernel_size=conv1d_kernel_size,
        )
        intra_ctx = _calc_intra_cp_seqs(
            raw_cu_seqlens=inter_ctx.cu_seqlens,
            chunk_size=CHUNK_SIZE,
            num_v_heads=num_v_heads,
            g=g if AUTOCP_MODEL else None,
            is_train=is_train,
            force_intra_cp=force_intra_cp,
        )
        return FlashQLACPContext(
            is_inter=True,
            is_intra=intra_ctx.is_intra,
            cu_seqlens=inter_ctx.cu_seqlens,
            group=inter_ctx.group,
            cu_seqlens_cpu=inter_ctx.cu_seqlens_cpu,
            is_last_rank=inter_ctx.is_last_rank,
            pre_num_ranks=inter_ctx.pre_num_ranks,
            is_first_rank=inter_ctx.is_first_rank,
            post_num_ranks=inter_ctx.post_num_ranks,
            conv1d_kernel_size=inter_ctx.conv1d_kernel_size,
            pre_num_conv_tokens=inter_ctx.pre_num_conv_tokens,
            intra_cp_cu_seqlens=intra_ctx.intra_cp_cu_seqlens,
            seq_map_r2c=intra_ctx.seq_map_r2c,
            seq_map_c2r=intra_ctx.seq_map_c2r,
            ht_mask=intra_ctx.ht_mask,
            ht_mask_bwd=intra_ctx.ht_mask_bwd,
        )

    if enable_inter:
        assert group is not None
        return _calc_inter_cp_seqs(
            cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group,
            conv1d_kernel_size=conv1d_kernel_size,
        )

    if enable_intra:
        assert (
            cu_seqlens is not None
            and num_v_heads is not None
        )
        return _calc_intra_cp_seqs(
            raw_cu_seqlens=cu_seqlens,
            chunk_size=CHUNK_SIZE,
            num_v_heads=num_v_heads,
            g=g if AUTOCP_MODEL else None,
            is_train=is_train,
            force_intra_cp=force_intra_cp,
        )

    return FlashQLACPContext(cu_seqlens=cu_seqlens)

# ---------------------------------------------------------------------------
# build inter-card context
# ---------------------------------------------------------------------------
@tensor_cache
def _calc_inter_cp_seqs(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: "dist.ProcessGroup | None" = None,
    conv1d_kernel_size: int | None = None,
) -> FlashQLACPContext:
    if world_size is None:
        assert group is not None
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    total_tokens = cu_seqlens_cpu[-1].item()
    assert total_tokens % world_size == 0, (
        f"inter-card CP requires total tokens ({total_tokens}) divisible by "
        f"world_size ({world_size}); pad/reshape the global sequence to a multiple of world_size."
    )
    part_len = total_tokens // world_size
    rank_start = part_len * rank
    rank_end = rank_start + part_len

    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side="right")
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side="left")
    subset_cu_seqlens = cu_seqlens_cpu[start_seq_idx: end_seq_idx + 1]

    local_cu_seqlens_cpu = (
        subset_cu_seqlens.clamp(min=rank_start, max=rank_end) - rank_start
    ).unique_consecutive().to(torch.int32)
    local_cu_seqlens_gpu = local_cu_seqlens_cpu.to(
        device=cu_seqlens.device, non_blocking=True)

    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()

    pre_num_conv_tokens = max(0, rank_start - first_seq_global_start)

    first_rank_of_first_seq = first_seq_global_start // part_len
    pre_num_ranks = rank - first_rank_of_first_seq
    is_first_rank = (rank == first_rank_of_first_seq)

    last_rank_of_last_seq = (last_seq_global_end - 1) // part_len
    post_num_ranks = last_rank_of_last_seq - rank
    is_last_rank = (rank == last_rank_of_last_seq)

    return FlashQLACPContext(
        is_inter=True,
        group=group,
        cu_seqlens=local_cu_seqlens_gpu,
        cu_seqlens_cpu=local_cu_seqlens_cpu,
        is_last_rank=is_last_rank,
        pre_num_ranks=pre_num_ranks,
        is_first_rank=is_first_rank,
        post_num_ranks=post_num_ranks,
        conv1d_kernel_size=conv1d_kernel_size,
        pre_num_conv_tokens=pre_num_conv_tokens,
    )

# ---------------------------------------------------------------------------
# build inter-card zigzag context
# ---------------------------------------------------------------------------
@tensor_cache
def _calc_inter_cp_seqs_zigzag(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
    group: "dist.ProcessGroup | None" = None,
) -> FlashQLACPContext:
    """Inter-card context for the zigzag layout: rank ``r`` owns chunk ``r`` and chunk
    ``NC-1-r`` (NC = 2*cp) of every document. Each owned chunk becomes its own local
    varlen segment, laid out as ``[early halves d=0..D-1, late halves d=0..D-1]`` -- the
    order the framework must feed this rank's tokens in. Builds the reindex that maps the
    all-gathered per-chunk ``(h, M)`` into per-document chunk order plus the extraction
    that reads each owned chunk's folded entering state back out."""
    if world_size is None:
        assert group is not None
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()
    cu_seqlens_cpu = cu_seqlens_cpu.to(dtype=torch.long)

    device = cu_seqlens.device
    cp = world_size
    NC = 2 * cp
    D = len(cu_seqlens_cpu) - 1
    doc_lens = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).tolist()
    for d, L in enumerate(doc_lens):
        assert L % NC == 0, (
            f"zigzag requires every doc length divisible by num_chunks={NC}; doc {d} "
            f"has length {L}. Pad the document to a multiple of 2*cp."
        )

    # local segments: early halves (chunk r) then late halves (chunk NC-1-r), each of
    # length L_d / NC.
    chunk_lens = [L // NC for L in doc_lens]
    local_lens = chunk_lens + chunk_lens
    local_cu = [0]
    for cl in local_lens:
        local_cu.append(local_cu[-1] + cl)
    local_cu_cpu = torch.tensor(local_cu, dtype=torch.int32)
    local_cu_gpu = local_cu_cpu.to(device=device, non_blocking=True)

    n_local = 2 * D
    # doc-chunk-order slot p = d*NC + c -> flat all-gather index rank_g*n_local + local_seg.
    c2g = torch.empty(D * NC, dtype=torch.long)
    for d in range(D):
        for c in range(NC):
            if c < cp:
                rank_g, local_seg = c, d              # early half on rank c
            else:
                rank_g, local_seg = NC - 1 - c, D + d  # late half on rank NC-1-c
            c2g[d * NC + c] = rank_g * n_local + local_seg

    fold_seq_map = torch.arange(0, D * NC + 1, NC, dtype=torch.int32)

    # local seg -> fold output slot (the doc-chunk position it owns).
    owned = torch.empty(n_local, dtype=torch.long)
    for d in range(D):
        owned[d] = d * NC + rank
        owned[D + d] = d * NC + (NC - 1 - rank)

    # --- fused-fold slot maps ---
    # The fused kernel scans straight out of the all-gather of `cat([h, M], dim=0)`, so
    # each rank contributes 2*n_local rows: h first, then M. These three maps are what
    # replace the physical reindex -- `h_slot`/`m_slot` address the two row ranges of the
    # gathered buffer, and `out_slot` says where (or whether) a scanned chunk is written.
    stride = 2 * n_local
    h_slot = torch.empty(D * NC, dtype=torch.int32)
    m_slot = torch.empty(D * NC, dtype=torch.int32)
    out_slot = torch.empty(D * NC, dtype=torch.int32)
    # h and M only share a dtype on the forward (both come from fused_gdr_h / the same
    # aggregate). On the backward dh is fp32 against a bf16 cached M, so they cannot ride
    # one cat-packed tensor and are gathered separately -- then each buffer is just
    # [world*n_local, ...] and one map serves both.
    split_slot = torch.empty(D * NC, dtype=torch.int32)
    for d in range(D):
        for c in range(NC):
            if c < cp:
                rank_g, local_seg = c, d
            else:
                rank_g, local_seg = NC - 1 - c, D + d
            p = d * NC + c
            h_slot[p] = rank_g * stride + local_seg
            m_slot[p] = rank_g * stride + n_local + local_seg
            split_slot[p] = rank_g * n_local + local_seg
            # every chunk is scanned, but only this rank's two per doc are stored.
            out_slot[p] = local_seg if rank_g == rank else -1

    return FlashQLACPContext(
        is_inter=True,
        is_zigzag=True,
        group=group,
        cu_seqlens=local_cu_gpu,
        cu_seqlens_cpu=local_cu_cpu,
        is_first_rank=(rank == 0),
        is_last_rank=(rank == world_size - 1),
        num_docs=D,
        num_chunks=NC,
        chunk_to_gathered_slot=c2g.to(device=device, non_blocking=True),
        fold_seq_map=fold_seq_map.to(device=device, non_blocking=True),
        owned_fold_slot=owned.to(device=device, non_blocking=True),
        fold_h_slot=h_slot.to(device=device, non_blocking=True),
        fold_m_slot=m_slot.to(device=device, non_blocking=True),
        fold_out_slot=out_slot.to(device=device, non_blocking=True),
        fold_split_slot=split_slot.to(device=device, non_blocking=True),
    )


# ---------------------------------------------------------------------------
# build intra-card context
# ---------------------------------------------------------------------------
AUTOCP_MODEL = is_calibrated(ARCH)


def _heuristic_intra_cp(num_chunks: list[int], H: int,
                        is_train: bool) -> tuple[bool, int]:
    # Latency model: T = a·L_cp + b·(B·H·Lc/P) / L_cp + c
    # Minimizing T yields the theoretical optimum: L_cp* ∝ √(B·H·Lc / P), where P = MULTI_PROCESSOR_COUNT, L_cp = max_local_chunks
    # Scaled by empirical factor (3) and aligned to the nearest power of 2 for optimal SM scheduling & memory alignment.

    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(H * sum(num_chunks) / MULTI_PROCESSOR_COUNT) * 3)
    )

    # Set min to 4 to ensure multi-stage pipelining in fused_gdr;
    max_local_chunks = max(max_local_chunks, 4)

    # Disable CP when sequences are too short or B * H naturally saturates SM occupancy.
    # CP has fixed overhead (warmup + correct_initial_states) that only pays off
    # when the longest sequence has enough chunks to amortize the cost.

    Be = sum(num_chunks) / max(num_chunks)

    if ARCH in ("sm90", "sm120", "sm121"):
        use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)
    elif ARCH in ("sm100", "sm103"):
        if is_train:
            use_cp = Be * H <= 56 and max(num_chunks) >= 16
        else:
            use_cp = (Be * H <= 56 and max(num_chunks) >= 256) or (
                Be * H <= 32 and max(num_chunks) >= 192
            )
    else:
        raise ValueError(f"no CP heuristic for {ARCH}")

    return use_cp, max_local_chunks


def _build_intra_cp_context(
    raw_cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_chunks: list[int],
    max_local_chunks: int,
) -> FlashQLACPContext:
    device = raw_cu_seqlens.device
    seqlen_dtype = raw_cu_seqlens.dtype
    raw_cu_seqlens_list = raw_cu_seqlens.tolist()

    cp_cu_seqlens = []
    ht_mask = []
    ht_mask_bwd = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    max_local_tokens = max_local_chunks * chunk_size
    for i, c in enumerate(num_chunks):
        s = raw_cu_seqlens_list[i]
        e = raw_cu_seqlens_list[i + 1]
        if c > max_local_chunks:
            first = True
            while s < e:
                cp_cu_seqlens.append(s)
                ht_mask.append(False)
                ht_mask_bwd.append(first)
                first = False
                seq_map_c2r.append(i)
                s += max_local_tokens
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(s)
            ht_mask.append(True)
            ht_mask_bwd.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu_seqlens_list[-1])

    # kept so ``get_full_warmup`` can read the segmentation without a D2H on the GPU
    # copy below; it is also the stable ``tensor_cache`` key for that warmup.
    cp_cu_seqlens_cpu = torch.tensor(cp_cu_seqlens, dtype=seqlen_dtype, device="cpu")
    cp_cu_seqlens = torch.tensor(
        cp_cu_seqlens, dtype=seqlen_dtype, device=device, requires_grad=False
    )
    seq_map_c2r = torch.tensor(seq_map_c2r, dtype=seqlen_dtype, device=device)
    seq_map_r2c = torch.tensor(
        seq_map_r2c, dtype=seqlen_dtype, device=device, requires_grad=False
    )
    ht_mask = torch.tensor(
        ht_mask, dtype=torch.bool, device=device, requires_grad=False
    )
    ht_mask_bwd = torch.tensor(
        ht_mask_bwd, dtype=torch.bool, device=device, requires_grad=False
    )

    return FlashQLACPContext(
        is_intra=True,
        cu_seqlens=raw_cu_seqlens,
        intra_cp_cu_seqlens=cp_cu_seqlens,
        intra_cp_cu_seqlens_cpu=cp_cu_seqlens_cpu,
        seq_map_r2c=seq_map_r2c,
        seq_map_c2r=seq_map_c2r,
        ht_mask=ht_mask,
        ht_mask_bwd=ht_mask_bwd,
    )


@tensor_cache
def _calc_intra_cp_seqs(
    raw_cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_v_heads: int,
    g: torch.Tensor | None = None,
    is_train: bool = False,
    force_intra_cp: bool = False,
) -> FlashQLACPContext:
    raw_cu_seqlens_list = raw_cu_seqlens.tolist()
    seqlens = [raw_cu_seqlens_list[i + 1] - raw_cu_seqlens_list[i]
               for i in range(len(raw_cu_seqlens_list) - 1)]
    num_chunks = [tilelang.cdiv(x, chunk_size) for x in seqlens]

    if AUTOCP_MODEL:
        use_cp, max_local_chunks = autocp_decide(
            num_chunks=num_chunks, num_v_heads=num_v_heads,
            P=MULTI_PROCESSOR_COUNT, chunk=chunk_size, is_train=is_train, g=g,
        )
    else:
        warnings.warn(
            "AUTOCP_MODEL is disabled; falling back to the heuristic intra-CP "
            "rule instead of the calibrated autocp model. Enable AUTOCP_MODEL "
            "for accurate CP decisions.",
            RuntimeWarning, stacklevel=2,
        )
        use_cp, max_local_chunks = _heuristic_intra_cp(
            num_chunks, num_v_heads, is_train)

    if force_intra_cp and not use_cp:
        use_cp = True
        if max_local_chunks is None:
            _, max_local_chunks = _heuristic_intra_cp(
                num_chunks, num_v_heads, is_train)

    if not use_cp:
        return FlashQLACPContext(is_intra=False, cu_seqlens=raw_cu_seqlens)

    return _build_intra_cp_context(
        raw_cu_seqlens, chunk_size, num_chunks, max_local_chunks)


# ---------------------------------------------------------------------------
# indexing tensors generation
# ---------------------------------------------------------------------------
@tensor_cache
def _create_cu_seqlens(
    batch_size: int,
    num_tokens: int,
    device_idx: int,
):
    return (
        torch.arange((batch_size + 1), dtype=torch.int32, device=f"cuda:{device_idx}")
        * num_tokens
    )


@tensor_cache
def _create_scan_seq_map(num_ranks: int, device_idx: int) -> torch.Tensor:
    seq_map = torch.zeros(2, dtype=torch.int32, device=f"cuda:{device_idx}")
    seq_map[1] = num_ranks + 1
    return seq_map


@tensor_cache
def _create_scan_fb_mask(num_ranks: int, num_v_heads: int, device_idx: int) -> torch.Tensor:
    return torch.ones((num_ranks + 1, num_v_heads), dtype=torch.bool, device=f"cuda:{device_idx}")


@tensor_cache
def _create_full_warmup(
    cu_seqlens_cpu: torch.Tensor,
    num_v_heads: int,
    chunk_size: int,
    device_idx: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[nseg, Hv]`` warmup-chunk count with no pruning -- see ``get_full_warmup``.

    ``cu_seqlens_cpu`` must be on the host: the whole point is to derive the counts
    without a D2H, then ship the result in one H2D.
    """
    lens = cu_seqlens_cpu.tolist()
    nw = torch.tensor(
        [(lens[i + 1] - lens[i] + chunk_size - 1) // chunk_size for i in range(len(lens) - 1)],
        dtype=dtype, device="cpu",
    )
    return nw.unsqueeze(1).expand(-1, num_v_heads).contiguous().to(f"cuda:{device_idx}")
