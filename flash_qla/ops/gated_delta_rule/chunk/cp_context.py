# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import tilelang

from flash_qla.utils import tensor_cache

_target_compute_version = tilelang.contrib.nvcc.get_target_compute_version()
if _target_compute_version == "9.0":
    from .hopper import get_warmup_chunks, fused_gdr_h, correct_initial_states

    _chunk_backend = "hopper"
elif _target_compute_version == "10.0":
    from .blackwell import get_warmup_chunks, fused_gdr_h, correct_initial_states

    _chunk_backend = "blackwell"
else:
    raise ValueError("FlashQLA now supports sm90/sm100 only.")


MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count


def _check_backend_matches_device(x: torch.Tensor):
    if not x.is_cuda:
        return
    major = torch.cuda.get_device_capability(x.device)[0]
    if major >= 10 and _chunk_backend != "blackwell":
        raise RuntimeError(
            "FlashQLA was imported with the SM90/Hopper TileLang target, but the "
            "input tensor is on an SM100+ Blackwell GPU. Re-import FlashQLA with "
            "TileLang target 10.0 enabled."
        )
    if major == 9 and _chunk_backend != "hopper":
        raise RuntimeError(
            "FlashQLA was imported with the SM100/Blackwell TileLang target, but "
            "the input tensor is on an SM90 Hopper GPU. Re-import FlashQLA with "
            "TileLang target 9.0 enabled."
        )


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
def _calc_cp_seqs(
    raw_cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_v_heads: int,
):
    # TODO: tilelang kernel
    device = raw_cu_seqlens.device
    seqlen_dtype = raw_cu_seqlens.dtype
    raw_cu_seqlens = raw_cu_seqlens.tolist()
    raw_batch_size = len(raw_cu_seqlens) - 1
    seqlens = [raw_cu_seqlens[i + 1] - raw_cu_seqlens[i] for i in range(raw_batch_size)]
    num_chunks = [tilelang.cdiv(x, chunk_size) for x in seqlens]

    # autocp
    H = num_v_heads
    # Latency model: T = a·L_cp + b·(B·H·Lc/P) / L_cp + c
    # Minimizing T yields the theoretical optimum: L_cp* ∝ √(B·H·Lc / P), where P = MULTI_PROCESSOR_COUNT, L_cp = max_local_chunks
    # Scaled by empirical factor (3) and aligned to the nearest power of 2 for optimal SM scheduling & memory alignment.

    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(H * sum(num_chunks) / MULTI_PROCESSOR_COUNT) * 3)
    )

    # Set min to 4 to ensure multi-stage pipelining in fused_gdr;
    max_local_chunks = max(max_local_chunks, 4)

    use_cp = False
    cp_cu_seqlens = []
    ht_mask = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    max_local_tokens = max_local_chunks * chunk_size
    for i, c in enumerate(num_chunks):
        s = raw_cu_seqlens[i]
        e = raw_cu_seqlens[i + 1]
        if c > max_local_chunks:
            while s < e:
                cp_cu_seqlens.append(s)
                ht_mask.append(False)
                seq_map_c2r.append(i)
                s += max_local_tokens
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(s)
            ht_mask.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu_seqlens[-1])

    # Disable CP when B * H naturally saturates SM occupancy.
    # For varlen inputs, use `total_chunks / max_seq_chunks` as effective B,
    # since CP helps accelerate highly uneven sequence lengths.

    Be = sum(num_chunks) / max(num_chunks)
    use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)

    if use_cp:
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
    else:
        cp_cu_seqlens, seq_map_r2c, ht_mask = None, None, None

    return use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask


def intra_card_cp_preprocess(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor,
    raw_cu_seqlens: torch.Tensor,
    warmup_threshold: float = -10.0,
):
    _check_backend_matches_device(k)
    batch_size, num_tokens, num_k_heads, k_head_dim = k.shape
    _, _, num_v_heads, v_head_dim = v.shape
    chunk_size = a.shape[-1]
    device = k.device

    if batch_size > 1:
        return raw_h0, raw_cu_seqlens, None, None

    if raw_cu_seqlens is None:
        raw_cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, device.index)

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask = _calc_cp_seqs(
        raw_cu_seqlens,
        chunk_size,
        num_v_heads,
    )

    if not use_cp:
        return raw_h0, raw_cu_seqlens, None, None

    num_warmup_chunks, fallback_mask = get_warmup_chunks(
        g=g,
        cu_seqlens=cp_cu_seqlens,
        ht_mask=ht_mask,
        chunk_size=chunk_size,
        threshold=warmup_threshold,
    )  # [cp_batch_size, num_v_heads]
    _, ht, mt = fused_gdr_h(
        k=k,
        v=v,
        a=a,
        g=g,
        b=b,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_chunks,
    )  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]
    cp_h0 = correct_initial_states(
        raw_h0=raw_h0,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c,
    )

    return cp_h0, cp_cu_seqlens, seq_map_c2r, raw_cu_seqlens
