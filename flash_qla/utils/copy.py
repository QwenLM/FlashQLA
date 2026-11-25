# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch


def copy_to_host(src: torch.Tensor) -> torch.Tensor:
    """使用原生 torch 同步取回 CPU 元数据。"""
    return src.cpu()


def copy_to_device(
    src: torch.Tensor,
    device: torch.device | str | int = "cuda",
    *,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    """使用原生 torch 将元数据拷到指定设备。"""
    device = torch.device(device)
    return src.to(device=device, dtype=dtype, non_blocking=non_blocking)
