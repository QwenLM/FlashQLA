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
    """Pack equally shaped (h, M) states as two contiguous row ranges.

    The gathered ``[world, 2n, ...]`` tensor flattens to ``[world*2n, ...]`` as a
    view; slot maps address h and M directly in either inter-card layout. Callers
    gather different shapes or dtypes separately.
    """
    assert S_ext.shape == M.shape, (S_ext.shape, M.shape)
    assert S_ext.dtype == M.dtype, (S_ext.dtype, M.dtype)
    # TODO: Have prepare_h / aggregate_card_state write gather-ready hm storage:
    # all owned chunks for zigzag, only the boundary chunk for contiguous CP.
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
