# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def all_gather_into_tensor(
    inp: torch.Tensor,
    out: torch.Tensor | None = None,
    group: "ProcessGroup | None" = None,
    async_op: bool = False,
):
    world_size = dist.get_world_size(group=group)
    if out is None:
        out = torch.empty(world_size, *inp.shape, device=inp.device, dtype=inp.dtype)
    handle = dist.all_gather_into_tensor(out, inp, group=group, async_op=async_op)
    return out, handle


def pack_states(S_ext: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Pack (h, M) for the zigzag fold by stacking rows, not bytes.

    At K == V == 128 the two states have identical shape, and under CP they carry the
    same dtype, so a plain ``cat`` along dim 0 gives one contiguous ``[2n, H, 128, 128]``
    payload. One collective, and -- unlike ``pack_hm`` -- the gathered result needs no
    unpacking: ``[world, 2n, ...]`` reshapes to ``[world*2n, ...]`` for free, and the
    fold kernel addresses h and M as two row ranges via its slot maps.

    Asserts rather than falling back: if this invariant ever breaks we want to hear
    about it, not to silently return to the five-pass path.
    """
    assert S_ext.shape == M.shape, (S_ext.shape, M.shape)
    assert S_ext.dtype == M.dtype, (S_ext.dtype, M.dtype)
    return torch.cat([S_ext, M], dim=0)


def pack_hm(S_ext: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    assert S_ext.shape[:1] == M.shape[:1], (S_ext.shape, M.shape)
    return torch.cat([
        S_ext.reshape(-1).view(torch.uint8),
        M.reshape(-1).view(torch.uint8),
    ])


def unpack_hm(hm: torch.Tensor, S_ext_like: torch.Tensor, M_like: torch.Tensor):
    n_cards = hm.shape[0]
    split = S_ext_like.numel() * S_ext_like.element_size()
    total = split + M_like.numel() * M_like.element_size()
    assert hm.shape[1:] == (total,), (hm.shape, total)
    S_ext = hm[:, :split].contiguous().view(S_ext_like.dtype)
    M = hm[:, split:].contiguous().view(M_like.dtype)
    return (S_ext.reshape(n_cards, *S_ext_like.shape),
            M.reshape(n_cards, *M_like.shape))
