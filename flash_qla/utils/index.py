# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools
from typing import Any
from collections.abc import Callable

import torch
import tilelang

_KEY_VALUE_TYPES = (int, float, bool, str, bytes, type(None))


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
    indices = torch.cat(
        [
            torch.arange(n)
            for n in tilelang.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def _fallback_prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    seqlens = torch.diff(cu_seqlens)
    num_chunks_per_seq = (seqlens + chunk_size - 1) // chunk_size
    chunk_offsets = torch.zeros_like(cu_seqlens)
    chunk_offsets[1:] = torch.cumsum(num_chunks_per_seq, dim=0)
    return chunk_offsets, chunk_offsets[-1].item()


def prepare_varlen_metadata(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    chunk_sizes: tuple[int, ...] = (16, 64),
) -> torch.Tensor:
    """Prepare immutable per-input chunk metadata before pipeline scheduling.

    Explicit ownership avoids the four-entry global cache eviction at MB8+.
    The CPU offsets must be the original source of the device tensor.
    Offsets and their aliases must remain read-only until metadata is released.
    """
    lengths = cu_seqlens_cpu.diff()

    def upload(host: torch.Tensor) -> torch.Tensor:
        if cu_seqlens.is_cuda:
            host = host.pin_memory()
        return host.to(device=cu_seqlens.device, dtype=cu_seqlens.dtype, non_blocking=True)

    entries = {}
    for size in chunk_sizes:
        counts = (lengths + size - 1) // size
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        # Match the original indices.eq(0).cumsum()-1 semantics, including empty documents.
        positions = torch.cat([torch.arange(n) for n in counts.tolist()]) if counts.numel() else torch.empty(0, dtype=torch.long)
        indices = torch.stack((positions.eq(0).cumsum(0) - 1, positions), dim=1)
        entries[size] = (upload(offsets), int(offsets[-1]), upload(indices))
    cu_seqlens._flash_qla_prepared_varlen = entries
    return cu_seqlens


def _prepared_varlen_entry(
    cu_seqlens: torch.Tensor, chunk_size: int,
) -> tuple[torch.Tensor, int, torch.Tensor] | None:
    prepared = getattr(cu_seqlens, "_flash_qla_prepared_varlen", None)
    if prepared is None:
        return None
    return prepared.get(chunk_size)


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    prepared = _prepared_varlen_entry(cu_seqlens, chunk_size)
    if prepared is not None:
        return prepared[2]
    return _fallback_prepare_chunk_indices(cu_seqlens, chunk_size)


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int,
) -> tuple[torch.Tensor, int]:
    prepared = _prepared_varlen_entry(cu_seqlens, chunk_size)
    if prepared is not None:
        return prepared[:2]
    return _fallback_prepare_chunk_offsets(cu_seqlens, chunk_size)
