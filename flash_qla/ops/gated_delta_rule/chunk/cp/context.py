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

    # --- intra-card ---
    intra_cp_cu_seqlens: torch.Tensor | None = None
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
        )
        # carry the intra-level partition tensors whenever the intra split is active.
        if self.is_intra:
            copied.intra_cp_cu_seqlens = (
                self.intra_cp_cu_seqlens.clone() if self.intra_cp_cu_seqlens is not None else None
            )
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

def build_cp_context(
    cu_seqlens: torch.Tensor | None = None,
    *,
    enable_inter: bool = False,
    enable_intra: bool = False,
    group: "ProcessGroup | None" = None,
    num_v_heads: int | None = None,
    chunk_size: int | None = None,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    is_train: bool = False,
    force_intra_cp: bool = False,
) -> FlashQLACPContext:
    """Unified CP-context builder over the canonical ``(cu_seqlens, num_v_heads)``.
    ``enable_inter``/``enable_intra`` select the mode: (T,F) pure inter | (F,T) pure
    intra | (T,T) inter+intra | (F,F) no CP.

    - inter needs ``group``.
    - intra needs ``cu_seqlens`` (varlen), ``chunk_size`` and ``num_v_heads``.
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
    if enable_inter and enable_intra:
        assert group is not None and chunk_size is not None and num_v_heads is not None
        # inter split -> local per-card cu_seqlens + rank topology; then intra split
        # over that local sequence; merge both field groups into one context.
        inter_ctx = _calc_inter_cp_seqs(
            cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, group=group,
            conv1d_kernel_size=conv1d_kernel_size,
        )
        intra_ctx = _calc_intra_cp_seqs(
            raw_cu_seqlens=inter_ctx.cu_seqlens, chunk_size=chunk_size,
            num_v_heads=num_v_heads, g=g if AUTOCP_MODEL else None,
            is_train=is_train, force_intra_cp=force_intra_cp,
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
        assert cu_seqlens is not None and chunk_size is not None and num_v_heads is not None
        return _calc_intra_cp_seqs(
            raw_cu_seqlens=cu_seqlens, chunk_size=chunk_size,
            num_v_heads=num_v_heads, g=g if AUTOCP_MODEL else None,
            is_train=is_train, force_intra_cp=force_intra_cp,
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
    # Pin the source so nonblocking context preparation does not need to
    # stage a pageable host buffer before the device copy.
    local_cu_source = (
        local_cu_seqlens_cpu.pin_memory()
        if cu_seqlens.is_cuda else local_cu_seqlens_cpu
    )
    local_cu_seqlens_gpu = local_cu_source.to(
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