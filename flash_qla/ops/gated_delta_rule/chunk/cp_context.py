# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import tilelang

from flash_qla.utils import tensor_cache

ARCH = None
if tilelang.contrib.nvcc.get_target_compute_version() == "9.0":
    from .hopper import get_warmup_chunks, get_warmup_chunks_bidi, fused_gdr_h, correct_initial_states, correct_terminal_states
    from .hopper.cp_bwd import fused_gdr_dh_ws as fused_gdr_dh
    ARCH = "SM90"
elif tilelang.contrib.nvcc.get_target_compute_version() == "10.0":
    from .blackwell import get_warmup_chunks, get_warmup_chunks_bidi, fused_gdr_h, correct_initial_states, correct_terminal_states
    from .blackwell.cp_bwd import fused_gdr_dh_ws as fused_gdr_dh
    ARCH = "SM100"
elif tilelang.contrib.nvcc.get_target_compute_version() == "10.3":
    from .blackwell import get_warmup_chunks, get_warmup_chunks_bidi, fused_gdr_h, correct_initial_states, correct_terminal_states
    from .blackwell.cp_bwd import fused_gdr_dh_ws as fused_gdr_dh
    ARCH = "SM103"
elif tilelang.contrib.nvcc.get_target_compute_version() == "12.0":
    from .blackwell_sm120 import get_warmup_chunks, get_warmup_chunks_bidi, fused_gdr_h, correct_initial_states, correct_terminal_states
    fused_gdr_dh = None
    ARCH = "SM120"
else:
    raise ValueError(f"FlashQLA now support sm90, sm100 and sm103 only. Found compute version: {tilelang.contrib.nvcc.get_target_compute_version()}")


MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count


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
    is_bwd: bool = False,
):
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
    ht_mask_bwd = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    max_local_tokens = max_local_chunks * chunk_size
    for i, c in enumerate(num_chunks):
        s = raw_cu_seqlens[i]
        e = raw_cu_seqlens[i + 1]
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
    cp_cu_seqlens.append(raw_cu_seqlens[-1])

    # Disable CP when sequences are too short or B * H naturally saturates SM occupancy.
    # CP has fixed overhead (warmup + correct_initial_states) that only pays off
    # when the longest sequence has enough chunks to amortize the cost.

    Be = sum(num_chunks) / max(num_chunks)

    if ARCH == "SM90" or ARCH == "SM120":
        use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)
    elif ARCH in ["SM100", "SM103"]:
        # SM100 uses separate thresholds for fwd and bwd:
        # - bwd kernel does more work per chunk (higher arithmetic intensity), so GPU
        #   under-utilization appears at fewer chunks (>=64 vs >=256 for fwd). It also
        #   runs prepare_dh (fused_gdr_dh) which itself benefits from CP parallelism,
        #   further lowering the break-even point.
        # - fwd has two tiers: moderate head count (Be*H<=56) needs very long sequences
        #   (>=256 chunks) to justify CP overhead; very low head count (Be*H<=32) allows
        #   slightly shorter sequences (>=192 chunks).
        if is_bwd:
            use_cp = Be * H <= 56 and max(num_chunks) >= 16
        else:
            use_cp = (Be * H <= 56 and max(num_chunks) >= 256) or (
                Be * H <= 32 and max(num_chunks) >= 192
            )
    else:
        raise ValueError(f"FlashQLA now support sm90, sm100 and sm103 only. Found compute version: {tilelang.contrib.nvcc.get_target_compute_version()}")

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
        ht_mask_bwd = torch.tensor(
            ht_mask_bwd, dtype=torch.bool, device=device, requires_grad=False
        )
    else:
        cp_cu_seqlens, seq_map_r2c, ht_mask, ht_mask_bwd = None, None, None, None

    return use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask, ht_mask_bwd


def intra_card_cp_preprocess(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor,
    raw_cu_seqlens: torch.Tensor,
    warmup_threshold: float = -10.0,
    state_v_first: bool = False,
    enable_fwd_cp_cache: bool = False,
):
    batch_size, num_tokens, num_k_heads, k_head_dim = k.shape
    _, _, num_v_heads, v_head_dim = v.shape
    chunk_size = a.shape[-1]
    device = k.device

    if batch_size > 1:
        return raw_h0, raw_cu_seqlens, None, None, None

    if raw_cu_seqlens is None:
        raw_cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, device.index)

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask, ht_mask_bwd = _calc_cp_seqs(
        raw_cu_seqlens=raw_cu_seqlens,
        chunk_size=chunk_size,
        num_v_heads=num_v_heads,
        is_bwd=False,
    )

    if not use_cp:
        return raw_h0, raw_cu_seqlens, None, None, None

    if enable_fwd_cp_cache:
        num_warmup_chunks, num_warmup_chunks_bwd, fallback_mask, fallback_mask_bwd = get_warmup_chunks_bidi(
            g=g,
            cu_seqlens=cp_cu_seqlens,
            ht_mask_fwd=ht_mask,
            ht_mask_bwd=ht_mask_bwd,
            chunk_size=chunk_size,
            threshold=warmup_threshold,
        )
    else:
        num_warmup_chunks, fallback_mask = get_warmup_chunks(
            g=g,
            cu_seqlens=cp_cu_seqlens,
            ht_mask=ht_mask,
            chunk_size=chunk_size,
            threshold=warmup_threshold,
        )  # [cp_batch_size, num_v_heads]
        num_warmup_chunks_bwd, fallback_mask_bwd = None, None

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
        state_v_first=state_v_first,
    )  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]

    cp_h0 = correct_initial_states(
        raw_h0=raw_h0,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c,
        state_v_first=state_v_first,
    )

    if enable_fwd_cp_cache:
        cp_cache = (cp_h0, mt, fallback_mask_bwd, num_warmup_chunks_bwd)
    else:
        cp_cache = None
    return cp_h0, cp_cu_seqlens, seq_map_c2r, raw_cu_seqlens, cp_cache


def intra_card_cp_preprocess_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor,
    q: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    raw_cu_seqlens: torch.Tensor,
    state_v_first: bool = False,
    cp_cache: tuple | None = None,
):
    batch_size, num_tokens, num_k_heads, _ = k.shape
    _, _, num_v_heads, _ = v.shape
    chunk_size = a.shape[-1]
    device = k.device

    if fused_gdr_dh is None:
        raise NotImplementedError(
            "Backward pass (CP) is not implemented for SM120 (Blackwell). "
            "Only forward pass is supported on this architecture."
        )

    if batch_size > 1:
        return raw_h0, raw_cu_seqlens, dht, raw_cu_seqlens, None, False

    if raw_cu_seqlens is None:
        raw_cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, device.index)

    use_cp, cp_cu_seqlens, seq_map_r2c, _, ht_mask, ht_mask_bwd = _calc_cp_seqs(
        raw_cu_seqlens=raw_cu_seqlens,
        chunk_size=chunk_size,
        num_v_heads=num_v_heads,
        is_bwd=True,
    )

    if not use_cp:
        return raw_h0, raw_cu_seqlens, dht, raw_cu_seqlens, None, False

    if cp_cache is not None:
        # Use cached forward CP artifacts
        cp_h0, mt_buffer, fallback_bwd, num_warmup_bwd = cp_cache
    else:
        num_warmup_h, num_warmup_bwd, fallback_fwd, fallback_bwd = get_warmup_chunks_bidi(
            g=g, cu_seqlens=cp_cu_seqlens,
            ht_mask_fwd=ht_mask, ht_mask_bwd=ht_mask_bwd,
            chunk_size=chunk_size,
        )

        _, ht_buffer, mt_buffer = fused_gdr_h(
            k=k, v=v, a=a, g=g, b=b,
            initial_state=None,
            output_final_state=True,
            output_h=False,
            cu_seqlens=cp_cu_seqlens,
            num_warmup_chunks=num_warmup_h,
            state_v_first=state_v_first,
        )

        cp_h0 = correct_initial_states(
            raw_h0=raw_h0,
            ht_buffer=ht_buffer,
            mt_buffer=mt_buffer,
            fallback_mask=fallback_fwd,
            seq_map_r2c=seq_map_r2c,
            state_v_first=state_v_first,
        )

    _, dht_buffer = fused_gdr_dh(
        q=q, k=k, a=a, g=g, b=b, do=do,
        dht=None,
        output_dh0=True,
        output_dh=False,
        scale=scale,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_bwd,
        state_v_first=state_v_first,
    )

    cp_dht = correct_terminal_states(
        raw_dht=dht,
        dht_buffer=dht_buffer,
        mt_buffer=mt_buffer.float(),
        fallback_mask=fallback_bwd,
        seq_map_r2c=seq_map_r2c,
        state_v_first=state_v_first,
    )

    return cp_h0, cp_cu_seqlens, cp_dht, cp_cu_seqlens, seq_map_r2c, True
