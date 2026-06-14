# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch


def _raise_blackwell_auto_cp_unsupported():
    raise RuntimeError(
        "FlashQLA auto_cp is not supported on SM100+ Blackwell GPUs. "
        "Run the Blackwell fused GDN path with auto_cp disabled."
    )


def get_warmup_chunks(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ht_mask: torch.Tensor,
    chunk_size: int,
    threshold: float,
):
    _raise_blackwell_auto_cp_unsupported()


def correct_initial_states(
    raw_h0: torch.Tensor,
    ht_buffer: torch.Tensor,
    mt_buffer: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
):
    _raise_blackwell_auto_cp_unsupported()
