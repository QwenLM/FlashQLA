# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import builtins
import functools
import os

import torch


@functools.lru_cache
def use_sm_memcpy() -> bool:
    return os.environ.get("FLA_USE_SM_MEMCPY", "0") == "1"


def builtin_has_sm_memcpy() -> bool:
    reg = getattr(builtins, "__flashmoe_registry__", None)
    return (
        isinstance(reg, dict)
        and reg.get("copy_to_host") is not None
        and reg.get("copy_to_device") is not None
    )


def copy_to_host(src: torch.Tensor) -> torch.Tensor:
    """同步取回 CPU 元数据；registry 随作用域变化，不缓存其内容。"""
    if src.is_cuda and src.numel() and use_sm_memcpy() and builtin_has_sm_memcpy():
        with torch.cuda.device(src.device):
            return builtins.__flashmoe_registry__["copy_to_host"](src.contiguous())
    return src.cpu()


def copy_to_device(
    src: torch.Tensor,
    device: torch.device | str | int = "cuda",
    *,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    """将 CPU 元数据拷到指定设备；CPU/设备内拷贝仍使用原生 torch。"""
    device = torch.device(device)
    if (
        src.device.type == "cpu"
        and device.type == "cuda"
        and src.numel()
        and use_sm_memcpy()
        and builtin_has_sm_memcpy()
    ):
        src = src.to(dtype=dtype or src.dtype).contiguous()
        # FlashMoE 的接口不接受 device，须在目标 GPU 的上下文中调用。
        with torch.cuda.device(device):
            return builtins.__flashmoe_registry__["copy_to_device"](src)
    return src.to(device=device, dtype=dtype, non_blocking=non_blocking)
