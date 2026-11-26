# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T


@tilelang.jit()
def tilelang_get_warmup_chunks(
    num_heads,
    chunk_size,
    threshold,
    accum_dtype,
    g_dtype,
    mask_dtype,
    seqlen_dtype,
    reverse: bool = False,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_threads = tilelang.cdiv(num_heads, 32) * 32

    @T.prim_func
    def tilelang_get_warmup_chunks_kernel(
        g: T.Tensor([1, num_tokens, num_heads], dtype=g_dtype),
        ht_mask: T.Tensor([batch_size], dtype=mask_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        num_warmup_chunks: T.Tensor([batch_size, num_heads], dtype=seqlen_dtype),
        fallback_mask: T.Tensor([batch_size, num_heads], dtype=mask_dtype),
    ):
        with T.Kernel(batch_size, threads=num_threads) as (bb,):
            if ht_mask[bb]:
                for i_h in T.Parallel(num_heads):
                    num_warmup_chunks[bb, i_h] = 0
            else:
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")
                num_iters = T.alloc_var("int32")
                seq_start_idx = cu_seqlens[bb]
                seq_end_idx = cu_seqlens[bb + 1]
                num_iters = (seq_end_idx - seq_start_idx) // chunk_size

                g_fragment = T.alloc_fragment((num_heads), dtype=accum_dtype)
                g_cumsum = T.alloc_fragment((num_heads), dtype=accum_dtype)
                n_fragment = T.alloc_fragment((num_heads), dtype=seqlen_dtype)
                f_fragment = T.alloc_fragment((num_heads), dtype=mask_dtype)
                T.clear(g_cumsum)
                T.fill(n_fragment, num_iters)
                T.fill(f_fragment, True)

                for i_s in T.serial(num_iters):
                    for i_h in T.Parallel(num_heads):
                        if reverse:
                            g_fragment[i_h] = g[
                                0,
                                seq_start_idx + (i_s + 1) * chunk_size - 1,
                                i_h,
                            ]
                        else:
                            g_fragment[i_h] = g[
                                0,
                                seq_end_idx - i_s * chunk_size - 1,
                                i_h,
                            ]
                    for i_h in T.Parallel(num_heads):
                        g_cumsum[i_h] += g_fragment[i_h]
                    for i_h in T.Parallel(num_heads):
                        if g_cumsum[i_h] < threshold and n_fragment[i_h] == num_iters:
                            n_fragment[i_h] = i_s + 1
                            f_fragment[i_h] = False

                for i_h in T.Parallel(num_heads):
                    num_warmup_chunks[bb, i_h] = n_fragment[i_h]
                for i_h in T.Parallel(num_heads):
                    fallback_mask[bb, i_h] = f_fragment[i_h]

    return tilelang_get_warmup_chunks_kernel


def get_warmup_chunks(
    g: torch.Tensor,  # [1, num_total_tokens, num_v_heads]
    cu_seqlens: torch.Tensor,  # [cp_real_batch_size + 1]
    ht_mask: torch.Tensor,  # [cp_real_batch_size]
    chunk_size: int = 32,
    threshold: float = -10.0,
    reverse: bool = False,
):
    batch_size, num_tokens, num_heads = g.shape
    real_batch_size = ht_mask.shape[0]
    assert cu_seqlens.shape[0] == real_batch_size + 1
    assert batch_size == 1
    assert chunk_size == 32

    tilelang_get_warmup_chunks_kernel = tilelang_get_warmup_chunks(
        num_heads=num_heads,
        chunk_size=chunk_size,
        threshold=threshold,
        accum_dtype="float32",
        g_dtype=g.dtype,
        mask_dtype=ht_mask.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        reverse=reverse,
    )
    num_warmup_chunks = torch.empty(
        [real_batch_size, num_heads], dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    fallback_mask = torch.empty(
        [real_batch_size, num_heads], dtype=ht_mask.dtype, device=cu_seqlens.device
    )
    tilelang_get_warmup_chunks_kernel(
        g, ht_mask, cu_seqlens, num_warmup_chunks, fallback_mask
    )

    return num_warmup_chunks, fallback_mask


@tilelang.jit()
def tilelang_get_warmup_chunks_bidi(
    num_heads,
    chunk_size,
    threshold,
    accum_dtype,
    g_dtype,
    mask_dtype,
    seqlen_dtype,
    force_inter_boundaries: bool = False,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    raw_batch_size = T.dynamic("raw_batch_size")
    num_threads = tilelang.cdiv(num_heads, 32) * 32

    @T.prim_func
    def tilelang_get_warmup_chunks_bidi_kernel(
        g: T.Tensor([1, num_tokens, num_heads], dtype=g_dtype),
        ht_mask_fwd: T.Tensor([batch_size], dtype=mask_dtype),
        ht_mask_bwd: T.Tensor([batch_size], dtype=mask_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
        num_warmup_h: T.Tensor([batch_size, num_heads], dtype=seqlen_dtype),
        num_warmup_bwd_out: T.Tensor([batch_size, num_heads], dtype=seqlen_dtype),
        fallback_fwd: T.Tensor([batch_size, num_heads], dtype=mask_dtype),
        fallback_bwd: T.Tensor([batch_size, num_heads], dtype=mask_dtype),
    ):
        with T.Kernel(batch_size, threads=num_threads) as (bb,):
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            num_iters = T.alloc_var("int32")
            seq_start_idx = cu_seqlens[bb]
            seq_end_idx = cu_seqlens[bb + 1]
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, chunk_size)

            bwd_g_idx = T.alloc_var("int32")

            g_fragment_fwd = T.alloc_fragment((num_heads), dtype=accum_dtype)
            g_fragment_bwd = T.alloc_fragment((num_heads), dtype=accum_dtype)
            g_cumsum_fwd = T.alloc_fragment((num_heads), dtype=accum_dtype)
            g_cumsum_bwd = T.alloc_fragment((num_heads), dtype=accum_dtype)
            n_fwd = T.alloc_fragment((num_heads), dtype=seqlen_dtype)
            n_bwd = T.alloc_fragment((num_heads), dtype=seqlen_dtype)
            f_fwd = T.alloc_fragment((num_heads), dtype=mask_dtype)
            f_bwd = T.alloc_fragment((num_heads), dtype=mask_dtype)

            T.clear(g_cumsum_fwd)
            T.clear(g_cumsum_bwd)

            if ht_mask_fwd[bb]:
                T.fill(n_fwd, 0)
                T.fill(f_fwd, False)
            else:
                T.fill(n_fwd, num_iters)
                T.fill(f_fwd, True)

            if ht_mask_bwd[bb]:
                T.fill(n_bwd, 0)
                T.fill(f_bwd, False)
            else:
                T.fill(n_bwd, num_iters)
                T.fill(f_bwd, True)

            for i_s in T.serial(num_iters):
                for i_h in T.Parallel(num_heads):
                    g_fragment_fwd[i_h] = g[
                        0,
                        seq_end_idx - i_s * chunk_size - 1,
                        i_h,
                    ]
                for i_h in T.Parallel(num_heads):
                    g_cumsum_fwd[i_h] += g_fragment_fwd[i_h]
                for i_h in T.Parallel(num_heads):
                    if g_cumsum_fwd[i_h] < threshold and n_fwd[i_h] == num_iters:
                        n_fwd[i_h] = i_s + 1
                        f_fwd[i_h] = False

                bwd_g_idx = seq_start_idx + (i_s + 1) * chunk_size - 1
                if bwd_g_idx >= seq_end_idx:
                    bwd_g_idx = seq_end_idx - 1
                for i_h in T.Parallel(num_heads):
                    g_fragment_bwd[i_h] = g[
                        0,
                        bwd_g_idx,
                        i_h,
                    ]
                for i_h in T.Parallel(num_heads):
                    g_cumsum_bwd[i_h] += g_fragment_bwd[i_h]
                for i_h in T.Parallel(num_heads):
                    if g_cumsum_bwd[i_h] < threshold and n_bwd[i_h] == num_iters:
                        n_bwd[i_h] = i_s + 1
                        f_bwd[i_h] = False

            if force_inter_boundaries:
                first_end = T.alloc_var("int32")
                last_start = T.alloc_var("int32")
                first_end = seq_map_r2c[1]
                last_start = seq_map_r2c[raw_batch_size - 1]
                if bb < first_end or bb >= last_start:
                    for i_h in T.Parallel(num_heads):
                        n_fwd[i_h] = num_iters
                        n_bwd[i_h] = num_iters
                        f_fwd[i_h] = True
                        f_bwd[i_h] = True

            for i_h in T.Parallel(num_heads):
                if n_fwd[i_h] > n_bwd[i_h]:
                    num_warmup_h[bb, i_h] = n_fwd[i_h]
                else:
                    num_warmup_h[bb, i_h] = n_bwd[i_h]
            for i_h in T.Parallel(num_heads):
                num_warmup_bwd_out[bb, i_h] = n_bwd[i_h]
            for i_h in T.Parallel(num_heads):
                fallback_fwd[bb, i_h] = f_fwd[i_h]
            for i_h in T.Parallel(num_heads):
                fallback_bwd[bb, i_h] = f_bwd[i_h]

    return tilelang_get_warmup_chunks_bidi_kernel


def get_warmup_chunks_bidi(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ht_mask_fwd: torch.Tensor,
    ht_mask_bwd: torch.Tensor,
    chunk_size: int = 32,
    threshold: float = -10.0,
    seq_map_r2c: torch.Tensor = None,
    force_inter_boundaries: bool = False,
):
    batch_size, num_tokens, num_heads = g.shape
    real_batch_size = ht_mask_fwd.shape[0]
    assert cu_seqlens.shape[0] == real_batch_size + 1
    assert batch_size == 1
    assert chunk_size == 32
    if force_inter_boundaries:
        assert seq_map_r2c is not None, "force_inter_boundaries requires seq_map_r2c"
    seq_map = seq_map_r2c if seq_map_r2c is not None else cu_seqlens.new_zeros(2)

    tilelang_kernel = tilelang_get_warmup_chunks_bidi(
        num_heads=num_heads,
        chunk_size=chunk_size,
        threshold=threshold,
        accum_dtype="float32",
        g_dtype=g.dtype,
        mask_dtype=ht_mask_fwd.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        force_inter_boundaries=force_inter_boundaries,
    )
    num_warmup_h = torch.empty(
        [real_batch_size, num_heads], dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    num_warmup_bwd = torch.empty(
        [real_batch_size, num_heads], dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    fallback_fwd = torch.empty(
        [real_batch_size, num_heads], dtype=ht_mask_fwd.dtype, device=cu_seqlens.device
    )
    fallback_bwd = torch.empty(
        [real_batch_size, num_heads], dtype=ht_mask_fwd.dtype, device=cu_seqlens.device
    )
    tilelang_kernel(
        g, ht_mask_fwd, ht_mask_bwd, cu_seqlens, seq_map,
        num_warmup_h, num_warmup_bwd, fallback_fwd, fallback_bwd,
    )

    return num_warmup_h, num_warmup_bwd, fallback_fwd, fallback_bwd


def select_correction_config(
    H, raw_batch_size, DV, device=None, *, num_sms=None,
):
    """Select precision and DV width for SM120 (RTX 5090, 99 KiB SMEM, 1:1 TF32:FP32).

    On GeForce Blackwell (SM120), dense TF32 Tensor Core throughput matches FP32 CUDA Core
    throughput (~105 TFLOPS, 1:1 vs ~14:1 on SM100), so K-tiled double-buffered TMA pure-FP32
    beats TF32x3 across all grid sizes while delivering ~4x higher precision (~6.9e-7 vs 2.5e-6).
    AutoCP supplies num_sms to use the kernel policy without a device query.
    """
    if min(H, raw_batch_size, DV) <= 0:
        raise ValueError("Correction dimensions must be positive")
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")
    n_core = H * raw_batch_size * ((DV + 127) // 128)
    if n_core * 8 <= num_sms:
        return "fp32", 16
    if n_core * 4 <= num_sms:
        return "fp32", 32
    return "fp32", 64


def _fp32_frag_layout(DK, block_DV, state_v_first):
    """128-thread vectorized (8-col) strided-row FP32 fragment layout for block_DV in {16, 32, 64}."""
    if not state_v_first:
        tn = 8
        tm = {16: 2, 32: 4, 64: 8}[block_DV]
        n_col_tiles = block_DV // tn
        n_row_tiles = DK // tm
        return T.Fragment(
            (DK, block_DV),
            forward_thread_fn=lambda r, c: (r % n_row_tiles) * n_col_tiles + (c // tn),
            forward_index_fn=lambda r, c: (r // n_row_tiles) * tn + (c % tn),
        )
    tn = 8
    n_col_tiles = DK // tn  # 16
    n_row_tiles = 128 // n_col_tiles  # 8
    return T.Fragment(
        (block_DV, DK),
        forward_thread_fn=lambda r, c: (c // tn) * n_row_tiles + (r % n_row_tiles),
        forward_index_fn=lambda r, c: (r // n_row_tiles) * tn + (c % tn),
    )


def _m_transpose_source_layout(rows, cols):
    """Assign adjacent threads to contiguous M[col, row] source elements."""
    return T.Fragment(
        (rows, cols),
        forward_thread_fn=lambda row, col: (col * rows + row) % 128,
        forward_index_fn=lambda row, col: (col * rows + row) // 128,
    )


@tilelang.jit()
def tilelang_correct_h0(
    H,
    DK,
    DV,
    res_dtype,
    accum_dtype,
    h_buffer_dtype,
    m_buffer_dtype,
    seqlen_dtype,
    mask_dtype,
    use_raw_h0,
    state_v_first,
    reverse: bool = False,
    transpose_m: bool = False,
    store_intra_h0: bool = True,
    store_inter_h: bool = False,
    use_slots: bool = False,
    precision: str = "fp32",
    block_DV: int | None = None,
):
    cp_batch_size = T.dynamic("cp_batch_size")
    raw_batch_size = T.dynamic("raw_batch_size")
    out_batch_size = T.dynamic("out_batch_size")
    slot_size = T.dynamic("slot_size")
    out_state_shape = (
        (out_batch_size, H, DV, DK)
        if state_v_first
        else (out_batch_size, H, DK, DV)
    )
    state_shape = (
        (cp_batch_size, H, DV, DK)
        if state_v_first
        else (cp_batch_size, H, DK, DV)
    )
    raw_state_shape = (
        (raw_batch_size, H, DV, DK)
        if state_v_first
        else (raw_batch_size, H, DK, DV)
    )

    if precision != "fp32":
        raise ValueError(f"Unsupported SM120 correction precision: {precision}")
    if block_DV is None:
        block_DV = 16
    block_DV = min(block_DV, DV)

    tile_shape = (block_DV, DK) if state_v_first else (DK, block_DV)

    # SM120 has 99 KiB shared memory per SM (vs 227 KiB on SM100) and 1:1 dense TF32:FP32 peak:
    # 256-thread warp-specialized TMA scan with K-tiled double-buffered m_stage (m_stages=2)
    # and direct coalesced ht_buffer loads into h_fragment (40-48 KiB SMEM -> >=2 CTAs/SM):
    #   * dv16 (bk=32, s=2): 32 + 8 = 40 KiB -> 2 CTAs/SM + 128B TMA swizzle
    #   * dv32 (bk=16 fwd / 32 bwd, s=2): 32 KiB (fwd) / 48 KiB (bwd) -> 2-3 CTAs/SM
    #   * dv64 (bk=16, s=2): 16 + 32 = 48 KiB -> 2 CTAs/SM + 8x8 FP32 register tile
    if block_DV <= 16:
        block_K = 32
    elif block_DV <= 32:
        block_K = 32 if transpose_m else 16
    else:
        block_K = 16
    m_stages = 2

    THREADS = 256
    CONSUMER_NREG = 240
    PRODUCER_NREG = 24
    BAR_CONSUMER = 3

    @T.macro
    def scan_body(
        bb, bh, bv, seq_start_idx, seq_end_idx, num_iters,
        raw_h0, ht_buffer, mt_buffer, fallback_mask,
        cp_h0, h_card,
        h_slot, m_slot, out_slot,
    ):
        lo = bv * block_DV
        num_kb = DK // block_K
        skip_last_gemm = store_intra_h0 and (not store_inter_h) and (not use_slots)

        hp_stage = T.alloc_shared((1,) + tile_shape, dtype=accum_dtype)
        m_shape = (m_stages, block_K, DK) if transpose_m else (m_stages, DK, block_K)
        m_stage = T.alloc_shared(m_shape, dtype=m_buffer_dtype)
        h_fragment = T.alloc_fragment(tile_shape, dtype=accum_dtype)

        if DK == 128 and block_DV in (16, 32, 64):
            T.annotate_layout({
                hp_stage: T.Layout((1,) + tile_shape, lambda s, r, c: (s, r, c)),
                h_fragment: _fp32_frag_layout(DK, block_DV, state_v_first),
            })

        # 3-role warp-specialized pipeline (256 threads):
        #   - Consumer math warps (tx   0..127, 240 regs): pure-FP32 MAC over K tiles
        #     using bank-conflict-free 8-col `_fp32_frag_layout`, plus direct coalesced
        #     reads of `ht_buffer` (no separate h_stage SMEM needed).
        #   - Producer load warp  (tx 128..159, 24 regs): TMA-prefetches K tiles of
        #     `mt_buffer` into 2-stage `m_stage`, skipping the entire segment when
        #     `fallback_mask == 0`. Handshakes with math via `m_ready` (32) / `m_free` (128).
        #   - Producer store warp (tx 160..191, 24 regs): asynchronously writes `hp_stage`
        #     to `cp_h0` via `hp_ready` (128) / `hp_free` (32).
        #   - In standard intra-card entry-state scan (`skip_last_gemm=True`), the last
        #     segment's exit state has no downstream consumer, so `gemm_iters = num_iters - 1`
        #     skips the redundant final 128x128 M load and multiply.
        m_ready = T.alloc_barrier(arrive_count=[32] * m_stages)
        m_free = T.alloc_barrier(arrive_count=[128] * m_stages)
        hp_ready = T.alloc_barrier(arrive_count=[128])
        hp_free = T.alloc_barrier(arrive_count=[32])

        tx = T.get_thread_binding()

        if tx < 128:
            T.set_max_nreg(CONSUMER_NREG, 1)

            if use_raw_h0:
                if state_v_first:
                    T.copy(raw_h0[bb, bh, lo:lo + block_DV, 0:DK], h_fragment)
                else:
                    T.copy(raw_h0[bb, bh, 0:DK, lo:lo + block_DV], h_fragment)
            else:
                T.clear(h_fragment)

            m_step = T.alloc_var("int32")
            m_step = 0
            gemm_iters = T.alloc_var("int32")
            gemm_iters = T.max(num_iters - 1, 0) if skip_last_gemm else num_iters

            for i_s in T.serial(gemm_iters):
                p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                h_idx = T.Cast("int32", h_slot[p]) if use_slots else p
                do_fb = fallback_mask[h_idx, bh]

                T.sync_threads(BAR_CONSUMER, 128)
                if store_intra_h0:
                    T.barrier_wait(hp_free[0], (i_s + 1) % 2)
                    T.copy(h_fragment, hp_stage[0, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(hp_ready[0])
                else:
                    T.copy(h_fragment, hp_stage[0, :, :])
                    T.fence_proxy_async()

                T.sync_threads(BAR_CONSUMER, 128)
                if state_v_first:
                    T.copy(ht_buffer[h_idx, bh, lo:lo + block_DV, 0:DK], h_fragment)
                else:
                    T.copy(ht_buffer[h_idx, bh, 0:DK, lo:lo + block_DV], h_fragment)

                if do_fb:
                    for kb in T.serial(num_kb):
                        ms = m_step % m_stages
                        mp = (m_step // m_stages) % 2
                        T.barrier_wait(m_ready[ms], mp)
                        for kk in T.serial(block_K):
                            if state_v_first:
                                for i, j in T.Parallel(block_DV, DK):
                                    av = T.cast(hp_stage[0, i, kb * block_K + kk], "float32")
                                    bv_m = T.cast(
                                        m_stage[ms, kk, j] if transpose_m else m_stage[ms, j, kk],
                                        "float32",
                                    )
                                    h_fragment[i, j] += av * bv_m
                            else:
                                for i, j in T.Parallel(DK, block_DV):
                                    av = T.cast(
                                        m_stage[ms, kk, i] if transpose_m else m_stage[ms, i, kk],
                                        "float32",
                                    )
                                    bv_h = T.cast(hp_stage[0, kb * block_K + kk, j], "float32")
                                    h_fragment[i, j] += av * bv_h
                        T.barrier_arrive(m_free[ms])
                        m_step += 1

            if skip_last_gemm:
                if num_iters > 0:
                    T.sync_threads(BAR_CONSUMER, 128)
                    T.barrier_wait(hp_free[0], (gemm_iters + 1) % 2)
                    T.copy(h_fragment, hp_stage[0, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(hp_ready[0])

            if store_inter_h:
                if state_v_first:
                    T.copy(h_fragment, h_card[bb, bh, lo:lo + block_DV, 0:DK])
                else:
                    T.copy(h_fragment, h_card[bb, bh, 0:DK, lo:lo + block_DV])

        else:
            T.set_max_nreg(PRODUCER_NREG, 0)

            if tx < 160:
                m_step = T.alloc_var("int32")
                m_step = 0
                gemm_iters = T.alloc_var("int32")
                gemm_iters = T.max(num_iters - 1, 0) if skip_last_gemm else num_iters
                for i_s in T.serial(gemm_iters):
                    p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                    h_idx = T.Cast("int32", h_slot[p]) if use_slots else p
                    m_idx = T.Cast("int32", m_slot[p]) if use_slots else p
                    if fallback_mask[h_idx, bh]:
                        for kb in T.serial(num_kb):
                            ms = m_step % m_stages
                            mp = (m_step // m_stages + 1) % 2
                            T.barrier_wait(m_free[ms], mp)
                            if transpose_m:
                                T.tma_copy(mt_buffer[m_idx, bh, kb * block_K:(kb + 1) * block_K, 0:DK],
                                           m_stage[ms, :, :], barrier=m_ready[ms])
                            else:
                                T.tma_copy(mt_buffer[m_idx, bh, 0:DK, kb * block_K:(kb + 1) * block_K],
                                           m_stage[ms, :, :], barrier=m_ready[ms])
                            T.barrier_arrive(m_ready[ms])
                            m_step += 1

            elif tx < 192:
                if store_intra_h0:
                    for i_s in T.serial(num_iters):
                        p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                        w_idx = T.Cast("int32", out_slot[p]) if use_slots else p
                        T.barrier_wait(hp_ready[0], i_s % 2)
                        if w_idx >= 0:
                            if state_v_first:
                                T.copy(hp_stage[0, :, :],
                                       cp_h0[w_idx, bh, lo:lo + block_DV, 0:DK])
                            else:
                                T.copy(hp_stage[0, :, :],
                                       cp_h0[w_idx, bh, 0:DK, lo:lo + block_DV])
                        T.barrier_arrive(hp_free[0])

    def _grid_prologue(bbhv, seq_map_r2c):
        bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
        bb, bh = bbh // H, bbh % H
        seq_start_idx = T.Cast("int32", seq_map_r2c[bb])
        seq_end_idx = T.Cast("int32", seq_map_r2c[bb + 1])
        num_iters = seq_end_idx - seq_start_idx
        return bb, bh, bv, seq_start_idx, seq_end_idx, num_iters

    grid = T.ceildiv(DV, block_DV) * H * raw_batch_size

    if use_slots:
        if use_raw_h0:
            @T.prim_func
            def tilelang_correct_h0_kernel(
                raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
                ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
                mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
                fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
                seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
                h_slot: T.Tensor([slot_size], dtype=seqlen_dtype),
                m_slot: T.Tensor([slot_size], dtype=seqlen_dtype),
                out_slot: T.Tensor([slot_size], dtype=seqlen_dtype),
                cp_h0: T.Tensor(out_state_shape, dtype=res_dtype),
            ):
                with T.Kernel(grid, threads=THREADS) as (bbhv,):
                    bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                    scan_body(bb, bh, bv, s0, s1, ni, raw_h0, ht_buffer, mt_buffer,
                              fallback_mask, cp_h0, ht_buffer,
                              h_slot, m_slot, out_slot)
        else:
            @T.prim_func
            def tilelang_correct_h0_kernel(
                ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
                mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
                fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
                seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
                h_slot: T.Tensor([slot_size], dtype=seqlen_dtype),
                m_slot: T.Tensor([slot_size], dtype=seqlen_dtype),
                out_slot: T.Tensor([slot_size], dtype=seqlen_dtype),
                cp_h0: T.Tensor(out_state_shape, dtype=res_dtype),
            ):
                with T.Kernel(grid, threads=THREADS) as (bbhv,):
                    bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                    scan_body(bb, bh, bv, s0, s1, ni, cp_h0, ht_buffer, mt_buffer,
                              fallback_mask, cp_h0, ht_buffer,
                              h_slot, m_slot, out_slot)

    elif store_inter_h:
        @T.prim_func
        def tilelang_correct_h0_kernel(
            raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
            ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            h_card: T.Tensor(raw_state_shape, dtype=res_dtype),
        ):
            with T.Kernel(grid, threads=THREADS) as (bbhv,):
                bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                scan_body(bb, bh, bv, s0, s1, ni, raw_h0, ht_buffer, mt_buffer,
                          fallback_mask, ht_buffer, h_card,
                          seq_map_r2c, seq_map_r2c, seq_map_r2c)

    elif use_raw_h0:
        @T.prim_func
        def tilelang_correct_h0_kernel(
            raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
            ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor(state_shape, dtype=res_dtype),
        ):
            with T.Kernel(grid, threads=THREADS) as (bbhv,):
                bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                scan_body(bb, bh, bv, s0, s1, ni, raw_h0, ht_buffer, mt_buffer,
                          fallback_mask, cp_h0, ht_buffer,
                          seq_map_r2c, seq_map_r2c, seq_map_r2c)

    else:
        @T.prim_func
        def tilelang_correct_h0_kernel(
            ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor(state_shape, dtype=res_dtype),
        ):
            with T.Kernel(grid, threads=THREADS) as (bbhv,):
                bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                scan_body(bb, bh, bv, s0, s1, ni, cp_h0, ht_buffer, mt_buffer,
                          fallback_mask, cp_h0, ht_buffer,
                          seq_map_r2c, seq_map_r2c, seq_map_r2c)

    return tilelang_correct_h0_kernel


def _correct_states(
    raw_state: torch.Tensor | None,
    h_buffer: torch.Tensor,
    m_buffer: torch.Tensor,
    fallback_mask: torch.Tensor | None,
    seq_map_r2c: torch.Tensor,
    state_v_first: bool,
    reverse: bool,
    *,
    h_slot: torch.Tensor | None,
    m_slot: torch.Tensor | None,
    out_slot: torch.Tensor | None,
    is_zigzag: bool,
):
    n_rows, num_heads, dim_2, dim_3 = h_buffer.shape
    if state_v_first:
        v_head_dim, k_head_dim = dim_2, dim_3
    else:
        k_head_dim, v_head_dim = dim_2, dim_3
    assert k_head_dim == v_head_dim == 128
    assert m_buffer.shape == (n_rows, num_heads, k_head_dim, k_head_dim)

    slots = (h_slot, m_slot, out_slot)
    use_slots = any(slot is not None for slot in slots)
    if use_slots:
        if not all(slot is not None for slot in slots):
            raise ValueError("h_slot, m_slot and out_slot must be supplied together")
        for slot in slots:
            assert slot.ndim == 1 and slot.shape == h_slot.shape
            assert slot.dtype == seq_map_r2c.dtype and slot.device == h_buffer.device
        output_rows = 2 * (seq_map_r2c.numel() - 1) if is_zigzag else 1
    else:
        output_rows = n_rows

    if fallback_mask is None:
        fallback_mask = torch.ones(
            (n_rows, num_heads), dtype=torch.bool, device=h_buffer.device,
        )
    else:
        assert fallback_mask.shape == (n_rows, num_heads)

    use_raw_h0 = raw_state is not None
    res_dtype = raw_state.dtype if use_raw_h0 else torch.float32
    precision, block_DV = select_correction_config(
        num_heads, seq_map_r2c.numel() - 1, v_head_dim, h_buffer.device,
    )
    kernel = tilelang_correct_h0(
        precision=precision,
        block_DV=block_DV,
        H=num_heads,
        DK=k_head_dim,
        DV=v_head_dim,
        res_dtype=res_dtype,
        accum_dtype="float32",
        h_buffer_dtype=h_buffer.dtype,
        m_buffer_dtype=m_buffer.dtype,
        seqlen_dtype=seq_map_r2c.dtype,
        mask_dtype=fallback_mask.dtype,
        use_raw_h0=use_raw_h0,
        state_v_first=state_v_first,
        reverse=reverse,
        transpose_m=reverse,
        use_slots=use_slots,
    )
    out = torch.empty(
        (output_rows, num_heads, dim_2, dim_3),
        dtype=res_dtype, device=h_buffer.device,
    )
    args = [h_buffer, m_buffer, fallback_mask, seq_map_r2c]
    if use_slots:
        args.extend(slots)
    args.append(out)
    if use_raw_h0:
        args.insert(0, raw_state)
    kernel(*args)
    return out[0] if use_slots and not is_zigzag else out


def correct_initial_states(
    raw_h0: torch.Tensor | None,
    ht_buffer: torch.Tensor,
    mt_buffer: torch.Tensor,
    fallback_mask: torch.Tensor | None,
    seq_map_r2c: torch.Tensor,
    state_v_first: bool = False,
    *,
    h_slot: torch.Tensor | None = None,
    m_slot: torch.Tensor | None = None,
    out_slot: torch.Tensor | None = None,
    is_zigzag: bool = False,
):
    return _correct_states(
        raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
        state_v_first=state_v_first, reverse=False,
        h_slot=h_slot, m_slot=m_slot, out_slot=out_slot, is_zigzag=is_zigzag,
    )


def correct_terminal_states(
    raw_dht: torch.Tensor | None,
    dht_buffer: torch.Tensor,
    mt_buffer: torch.Tensor,
    fallback_mask: torch.Tensor | None,
    seq_map_r2c: torch.Tensor,
    state_v_first: bool = False,
    *,
    h_slot: torch.Tensor | None = None,
    m_slot: torch.Tensor | None = None,
    out_slot: torch.Tensor | None = None,
    is_zigzag: bool = False,
):
    return _correct_states(
        raw_dht, dht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
        state_v_first=state_v_first, reverse=True,
        h_slot=h_slot, m_slot=m_slot, out_slot=out_slot, is_zigzag=is_zigzag,
    )


# ---------------------------------------------------------------------------
# card-level (h, M) aggregation for inter+intra CP (SM120 99 KiB-safe)
# ---------------------------------------------------------------------------
@tilelang.jit()
def tilelang_aggregate_hm(
    H, h_dtype, m_dtype, seqlen_dtype, mask_dtype, state_v_first, use_raw_h0,
):
    """Scan independent columns of [h, M] with FP32 state and TF32x3 products.

    Uses block_K=32 (56 KiB total SMEM) to fit comfortably inside SM120's 99 KiB budget.
    """
    cp_batch_size = T.dynamic("cp_batch_size")
    raw_batch_size = T.dynamic("raw_batch_size")
    DK, block_DV, block_K = 128, 16, 32
    state_shape = (raw_batch_size, H, DK, DK)
    buffer_shape = (cp_batch_size, H, DK, DK)
    prev_layout = T.Layout((DK, 2 * block_DV), lambda row, col: (row, col))
    state_load_layout = _m_transpose_source_layout(2 * block_DV, block_K)

    @T.prim_func
    def tilelang_aggregate_hm_kernel(
        raw_h0: T.Tensor(state_shape, h_dtype),
        ht_buffer: T.Tensor(buffer_shape, h_dtype),
        mt_buffer: T.Tensor(buffer_shape, m_dtype),
        fallback_mask: T.Tensor((cp_batch_size, H), mask_dtype),
        seq_map: T.Tensor((raw_batch_size + 1,), seqlen_dtype),
        h_card: T.Tensor(state_shape, h_dtype),
        m_card: T.Tensor(state_shape, h_dtype),
    ):
        with T.Kernel(raw_batch_size * H, DK // block_DV, threads=128) as (bbh, block_col):
            bb, bh = bbh // H, bbh % H
            lo = block_col * block_DV
            start = T.cast(seq_map[bb], "int32")
            end = T.cast(seq_map[bb + 1], "int32")
            prev = T.alloc_shared((DK, 2 * block_DV), "float32")
            acc = T.alloc_fragment((DK, 2 * block_DV), "float32")
            a_hi = T.alloc_shared((DK, block_K), T.tfloat32)
            a_lo = T.alloc_shared((DK, block_K), T.tfloat32)
            b_hi = T.alloc_shared((2 * block_DV, block_K), T.tfloat32)
            b_lo = T.alloc_shared((2 * block_DV, block_K), T.tfloat32)
            T.annotate_layout({prev: prev_layout})

            for i, j in T.Parallel(DK, 2 * block_DV):
                if j < block_DV:
                    if use_raw_h0:
                        acc[i, j] = (raw_h0[bb, bh, lo + j, i] if state_v_first
                                     else raw_h0[bb, bh, i, lo + j])
                    else:
                        acc[i, j] = 0.0
                else:
                    acc[i, j] = T.if_then_else(i == lo + j - block_DV, 1.0, 0.0)

            for p in T.serial(start, end):
                T.copy(acc, prev)
                for i, j in T.Parallel(DK, 2 * block_DV):
                    if j < block_DV:
                        acc[i, j] = (ht_buffer[p, bh, lo + j, i] if state_v_first
                                     else ht_buffer[p, bh, i, lo + j])
                    else:
                        acc[i, j] = 0.0
                if fallback_mask[p, bh]:
                    if p == start and not use_raw_h0:
                        for i, j in T.Parallel(DK, block_DV):
                            acc[i, block_DV + j] = T.cast(mt_buffer[p, bh, i, lo + j], "float32")
                    else:
                        for kb in T.serial(DK // block_K):
                            for i, k in T.Parallel(DK, block_K):
                                av = T.cast(mt_buffer[p, bh, i, kb * block_K + k], "float32")
                                av_hi = T.cast(av, T.tfloat32)
                                a_hi[i, k] = av_hi
                                a_lo[i, k] = T.cast(av - T.cast(av_hi, "float32"), T.tfloat32)
                            for j, k in T.Parallel(2 * block_DV, block_K, loop_layout=state_load_layout):
                                bv = prev[kb * block_K + k, j]
                                bv_hi = T.cast(bv, T.tfloat32)
                                b_hi[j, k] = bv_hi
                                b_lo[j, k] = T.cast(bv - T.cast(bv_hi, "float32"), T.tfloat32)
                            T.gemm(a_hi, b_lo, acc, transpose_B=True, clear_accum=False)
                            T.gemm(a_lo, b_hi, acc, transpose_B=True, clear_accum=False)
                            T.gemm(a_hi, b_hi, acc, transpose_B=True, clear_accum=False)

            for i, j in T.Parallel(DK, block_DV):
                if state_v_first:
                    h_card[bb, bh, lo + j, i] = acc[i, j]
                else:
                    h_card[bb, bh, i, lo + j] = acc[i, j]
            for i, j in T.Parallel(DK, block_DV):
                m_card[bb, bh, i, lo + j] = acc[i, block_DV + j]

    return tilelang_aggregate_hm_kernel


def aggregate_card_state(
    ht_buffer: torch.Tensor,
    mt_buffer: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    raw_h0: torch.Tensor | None = None,
    state_v_first: bool = False,
    reverse: bool = False,
    transpose_m: bool = False,
    compute_m: bool = True,
):
    """Aggregate h/dh; computing M is supported only for a forward scan."""
    if compute_m and (reverse or transpose_m):
        raise ValueError(
            "Blackwell SM120 M aggregation requires reverse=False and transpose_m=False"
        )

    raw_batch_size = seq_map_r2c.shape[0] - 1
    _, num_heads, dim_2, dim_3 = ht_buffer.shape
    if state_v_first:
        v_head_dim, k_head_dim = dim_2, dim_3
    else:
        k_head_dim, v_head_dim = dim_2, dim_3
    assert k_head_dim == v_head_dim == 128

    res_dtype = ht_buffer.dtype
    use_raw_h0 = raw_h0 is not None
    if use_raw_h0:
        raw_h0 = raw_h0.to(res_dtype)

    raw_state_shape = (
        (raw_batch_size, num_heads, v_head_dim, k_head_dim)
        if state_v_first
        else (raw_batch_size, num_heads, k_head_dim, v_head_dim)
    )
    if compute_m:
        state_kernel = tilelang_aggregate_hm(
            num_heads, ht_buffer.dtype, mt_buffer.dtype, seq_map_r2c.dtype,
            fallback_mask.dtype, state_v_first, use_raw_h0,
        )
        h_card = torch.empty(raw_state_shape, dtype=res_dtype, device=ht_buffer.device)
        m_card = torch.empty(
            (raw_batch_size, num_heads, k_head_dim, k_head_dim),
            dtype=res_dtype, device=mt_buffer.device,
        )
        if not use_raw_h0:
            raw_h0 = torch.empty(raw_state_shape, dtype=res_dtype, device=ht_buffer.device)
        state_kernel(raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, h_card, m_card)
        return h_card, m_card

    precision, block_DV = select_correction_config(
        num_heads, raw_batch_size, v_head_dim, ht_buffer.device,
    )
    state_kernel = tilelang_correct_h0(
        precision=precision,
        block_DV=block_DV,
        H=num_heads,
        DK=k_head_dim,
        DV=v_head_dim,
        res_dtype=res_dtype,
        accum_dtype="float32",
        h_buffer_dtype=ht_buffer.dtype,
        m_buffer_dtype=mt_buffer.dtype,
        seqlen_dtype=seq_map_r2c.dtype,
        mask_dtype=fallback_mask.dtype,
        use_raw_h0=use_raw_h0,
        state_v_first=state_v_first,
        reverse=reverse,
        transpose_m=transpose_m,
        store_intra_h0=False,
        store_inter_h=True,
    )

    h_card = torch.empty(raw_state_shape, dtype=res_dtype, device=ht_buffer.device)
    if not use_raw_h0:
        raw_h0 = torch.empty(raw_state_shape, dtype=res_dtype, device=ht_buffer.device)
    state_kernel(raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, h_card)
    return h_card, None
