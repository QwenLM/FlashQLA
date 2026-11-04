# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

import torch
import tilelang

from .copy import copy_to_device, copy_to_host

if TYPE_CHECKING:
    from flash_qla.ops.gated_delta_rule.chunk.cp.context import FlashQLACPContext

_KEY_VALUE_TYPES = (int, float, bool, str, bytes, type(None))


@dataclass(frozen=True)
class PreparedVarlen:
    """Varlen metadata prepared for a specific chunk size."""

    chunk_size: int
    chunk_offsets: torch.Tensor
    total_chunks: int
    chunk_indices: torch.Tensor
    cp_context: FlashQLACPContext | None = None


def _same_key(a: Any, b: Any) -> bool:
    if a is b:
        return True
    return type(a) is type(b) and isinstance(a, _KEY_VALUE_TYPES) and a == b


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The dynamic cache is limited to a fixed size (default is 4). CUDA graph
    capture entries are marked static and are not evicted.
    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.
    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries = []
    cache_size = 4
    dynamic_cache_size = 0

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, dynamic_cache_size
        assert torch.cuda.is_available()
        is_capturing = torch.cuda.is_current_stream_capturing()

        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result, is_static = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(_same_key(a, b) for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and _same_key(v, last_kwargs[k])
                    for k, v in kwargs.items()
                ):
                    if is_capturing and not is_static:
                        cache_entries[i] = (
                            last_args,
                            last_kwargs,
                            last_result,
                            True,
                        )
                        dynamic_cache_size -= 1
                    elif not is_static:
                        cache_entries = (
                            cache_entries[:i]
                            + cache_entries[i + 1 :]
                            + [entry]
                        )
                    return last_result

        assert not is_capturing, (
            f"FLA tensor_cache miss during CUDA graph capture: {fn.__name__}"
        )
        result = fn(*args, **kwargs)

        if dynamic_cache_size >= cache_size:
            while cache_entries:
                entry = cache_entries[0]
                cache_entries = cache_entries[1:]
                if entry[3]:
                    cache_entries.append(entry)
                else:
                    dynamic_cache_size -= 1
                    break

        cache_entries.append((args, kwargs, result, False))
        dynamic_cache_size += 1
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def _fallback_prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    cu_seqlens_cpu = copy_to_host(cu_seqlens)
    indices = torch.cat(
        [
            torch.arange(n)
            for n in tilelang.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()
        ]
    )
    indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    return copy_to_device(indices, cu_seqlens.device, dtype=cu_seqlens.dtype)


@tensor_cache
def _fallback_prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    seqlens = torch.diff(cu_seqlens)
    num_chunks_per_seq = (seqlens + chunk_size - 1) // chunk_size
    chunk_offsets = torch.zeros_like(cu_seqlens)
    chunk_offsets[1:] = torch.cumsum(num_chunks_per_seq, dim=0)
    return chunk_offsets, copy_to_host(chunk_offsets[-1]).item()


def prepare_varlen_metadata(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    group: "Any | None" = None,
    num_v_heads: int | None = None,
    is_train: bool = True,
    conv1d_kernel_size: int | None = None,
    force_intra_cp: bool = False,
) -> torch.Tensor:
    """Attach chunk offsets, count, and indices using CPU lengths and the backend chunk size.

    When ``num_v_heads`` and/or ``group`` is given, the full :class:`FlashQLACPContext` is
    precomputed here (via the same builder ``build_cp_context`` delegates to) and attached
    as ``cp_context``. A later ``build_cp_context`` on this ``cu_seqlens`` then returns it
    directly, skipping ``_calc_inter_cp_seqs`` / ``_calc_intra_cp_seqs`` entirely. Pass the
    same CP arguments here as you would to ``build_cp_context`` so the stored context is the
    one you expect -- the fast path returns it verbatim regardless of later arguments.

    ``g`` is deliberately not accepted: a gate-aware intra decision reads the per-head decay
    and cannot be cached on the prepared fast path, so ``cp_context`` here is always built
    with ``g=None``. Accordingly ``build_cp_context`` asserts ``g is None`` when it reuses a
    prepared ``cp_context``.
    """
    from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
    from flash_qla.ops.gated_delta_rule.chunk.cp.context import _build_cp_context_impl

    lengths = cu_seqlens_cpu.diff()

    counts = (lengths + CHUNK_SIZE - 1) // CHUNK_SIZE
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    # Match the original indices.eq(0).cumsum()-1 semantics, including empty documents.
    positions = torch.cat([torch.arange(n) for n in counts.tolist()]) if counts.numel() else torch.empty(0, dtype=torch.long)
    indices = torch.stack((positions.eq(0).cumsum(0) - 1, positions), dim=1)

    cp_context = None
    if group is not None or num_v_heads is not None:
        cp_context = _build_cp_context_impl(
            cu_seqlens, group=group,
            enable_inter=group is not None, enable_intra=num_v_heads is not None,
            num_v_heads=num_v_heads, conv1d_kernel_size=conv1d_kernel_size,
            cu_seqlens_cpu=cu_seqlens_cpu, is_train=is_train,
            force_intra_cp=force_intra_cp,
        )

    cu_seqlens._flash_qla_prepared_varlen = PreparedVarlen(
        chunk_size=CHUNK_SIZE,
        chunk_offsets=copy_to_device(offsets, cu_seqlens.device, dtype=cu_seqlens.dtype, non_blocking=True),
        total_chunks=int(offsets[-1]),
        chunk_indices=copy_to_device(indices, cu_seqlens.device, dtype=cu_seqlens.dtype, non_blocking=True),
        cp_context=cp_context,
    )
    return cu_seqlens


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    prepared = getattr(cu_seqlens, "_flash_qla_prepared_varlen", None)
    if prepared is not None and prepared.chunk_size == chunk_size:
        return prepared.chunk_indices
    return _fallback_prepare_chunk_indices(cu_seqlens, chunk_size)


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int,
) -> tuple[torch.Tensor, int]:
    prepared = getattr(cu_seqlens, "_flash_qla_prepared_varlen", None)
    if prepared is not None and prepared.chunk_size == chunk_size:
        return prepared.chunk_offsets, prepared.total_chunks
    return _fallback_prepare_chunk_offsets(cu_seqlens, chunk_size)
