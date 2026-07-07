# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools
from typing import Any
from collections.abc import Callable

import torch
import tilelang


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    cache = []
    cache_size = 4

    def _equal(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return a is b
        return a == b

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache, cache_size
        for (cached_args, cached_kwargs, cached_result) in cache:
            if all(_equal(a, b) for a, b in zip(args, cached_args, strict=False)) and \
                    all(k in cached_kwargs and _equal(v, cached_kwargs[k]) for k, v in kwargs.items()):
                return cached_result
        result = fn(*args, **kwargs)
        cache.insert(0, (args, kwargs, result))
        if len(cache) > cache_size:
            cache = cache[:cache_size]
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
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
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    seqlens = torch.diff(cu_seqlens)
    num_chunks_per_seq = (seqlens + chunk_size - 1) // chunk_size
    chunk_offsets = torch.zeros_like(cu_seqlens)
    chunk_offsets[1:] = torch.cumsum(num_chunks_per_seq, dim=0)
    return chunk_offsets, chunk_offsets[-1].item()
