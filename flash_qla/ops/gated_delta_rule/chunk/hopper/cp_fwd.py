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
    chunk_size: int = 64,
    threshold: float = -10.0,
    reverse: bool = False,
):
    batch_size, num_tokens, num_heads = g.shape
    real_batch_size = ht_mask.shape[0]
    assert cu_seqlens.shape[0] == real_batch_size + 1
    assert batch_size == 1
    assert chunk_size == 64

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
    chunk_size: int = 64,
    threshold: float = -10.0,
    seq_map_r2c: torch.Tensor = None,
    force_inter_boundaries: bool = False,
):
    batch_size, num_tokens, num_heads = g.shape
    real_batch_size = ht_mask_fwd.shape[0]
    assert cu_seqlens.shape[0] == real_batch_size + 1
    assert batch_size == 1
    assert chunk_size == 64
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
    compute_m: bool = False,
    use_zigzag: bool = False,
    block_DV: int | None = None,
    num_stages: int | None = None,
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
    
    wide_M = m_buffer_dtype.itemsize > 2
    if block_DV is None:
        block_DV = 64 if compute_m or (store_intra_h0 and wide_M) else 128
        block_DV = min(block_DV, DV)
    if num_stages is None:
        num_stages = 1 if store_intra_h0 or wide_M else 2

    tile_shape = (block_DV, DK) if state_v_first else (DK, block_DV)

    WS_THREADS = 256          # 128 math + 32 load + 32 store + 64 parked
    CONSUMER_NREG = 240
    PRODUCER_NREG = 24

    BAR_CONSUMER = 3

    @T.macro
    def scan_body(
        bb, bh, bv, seq_start_idx, seq_end_idx, num_iters,
        raw_h0, ht_buffer, mt_buffer, fallback_mask,
        cp_h0, h_card, m_card,
        h_slot, m_slot, out_slot,
    ):
        lo = bv * block_DV

        # Ring buffers: filled by the load warp, drained by the math warpgroup.
        h_stage = T.alloc_shared((num_stages,) + tile_shape, dtype=h_buffer_dtype)
        m_stage = T.alloc_shared((num_stages, DK, DK), dtype=m_buffer_dtype)
        hp_stage = T.alloc_shared((num_stages,) + tile_shape, dtype=res_dtype)

        hd_shared = T.alloc_shared(tile_shape, dtype=m_buffer_dtype)
        mrun_shared = T.alloc_shared((DK, DK), dtype=m_buffer_dtype)

        h_fragment = T.alloc_fragment(tile_shape, dtype=accum_dtype)
        mrun_frag = T.alloc_fragment((DK, DK), dtype=accum_dtype)

        data_is_ready = T.alloc_barrier(arrive_count=[32] * num_stages)    # load -> math
        data_is_free = T.alloc_barrier(arrive_count=[128] * num_stages)    # math -> load
        hprev_ready = T.alloc_barrier(arrive_count=[128] * num_stages)     # math -> store
        hprev_free = T.alloc_barrier(arrive_count=[32] * num_stages)       # store -> math

        tx = T.get_thread_binding()

        if tx < 128:
            # ==================== math warpgroup ====================
            T.set_max_nreg(CONSUMER_NREG, 1)

            if use_raw_h0:
                if state_v_first:
                    T.copy(raw_h0[bb, bh, lo:lo + block_DV, 0:DK], h_fragment)
                else:
                    T.copy(raw_h0[bb, bh, 0:DK, lo:lo + block_DV], h_fragment)
            else:
                T.clear(h_fragment)

            if compute_m and bv == 0:
                for i, j in T.Parallel(DK, DK):   # M_running := identity
                    if i == j:
                        mrun_frag[i, j] = 1.0
                    else:
                        mrun_frag[i, j] = 0.0
                T.copy(mrun_frag, mrun_shared)

            for i_s in T.serial(num_iters):
                # `p` is the position in the scan; under zigzag the buffers are the raw
                # all-gather output, so the row to touch comes from a slot map rather
                # than from `p` itself. Non-zigzag binds the same PrimExpr, which is why
                # the emitted TIR is identical to before this indirection existed.
                p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                h_idx = T.Cast("int32", h_slot[p]) if use_zigzag else p
                s = i_s % num_stages

                if store_intra_h0:
                    T.barrier_wait(hprev_free[s], (i_s // num_stages + 1) % 2)
                    T.copy(h_fragment, hp_stage[s, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(hprev_ready[s])

                T.sync_threads(BAR_CONSUMER, 128)
                if fallback_mask[h_idx, bh]:
                    T.copy(h_fragment, hd_shared)     # save h_prev as a gemm operand
                    T.fence_proxy_async()
                T.sync_threads(BAR_CONSUMER, 128)

                T.barrier_wait(data_is_ready[s], (i_s // num_stages + 0) % 2)
                T.copy(h_stage[s, :, :], h_fragment)          # h_fragment = h[h_idx]
                if fallback_mask[h_idx, bh]:
                    if state_v_first:
                        if transpose_m:
                            T.gemm(hd_shared, m_stage[s, :, :], h_fragment,
                                   clear_accum=False)
                        else:
                            T.gemm(hd_shared, m_stage[s, :, :], h_fragment,
                                   transpose_B=True, clear_accum=False)
                    else:
                        if transpose_m:
                            T.gemm(m_stage[s, :, :], hd_shared, h_fragment,
                                   transpose_A=True, clear_accum=False)
                        else:
                            T.gemm(m_stage[s, :, :], hd_shared, h_fragment,
                                   clear_accum=False)

                if compute_m and bv == 0:
                    if fallback_mask[h_idx, bh]:
                        T.clear(mrun_frag)
                        T.gemm(m_stage[s, :, :], mrun_shared, mrun_frag,
                               clear_accum=True)
                        T.sync_threads(BAR_CONSUMER, 128)   # WAR on mrun_shared
                        T.copy(mrun_frag, mrun_shared)
                        T.fence_proxy_async()
                    else:
                        T.clear(mrun_shared)
                    T.sync_threads(BAR_CONSUMER, 128)   # the next iteration reads it

                T.barrier_arrive(data_is_free[s])

            if store_inter_h:   # post-last-chunk exiting state = card-level h
                if state_v_first:
                    T.copy(h_fragment, h_card[bb, bh, lo:lo + block_DV, 0:DK])
                else:
                    T.copy(h_fragment, h_card[bb, bh, 0:DK, lo:lo + block_DV])
            if compute_m and bv == 0:
                T.copy(mrun_shared, m_card[bb, bh, 0:DK, 0:DK])

        else:
            T.set_max_nreg(PRODUCER_NREG, 0)

            if tx < 160:
                # ==================== load warp ====================
                for i_s in T.serial(num_iters):
                    p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                    # h and M are separate rows of one cat-packed buffer under zigzag,
                    # hence two maps rather than one.
                    h_idx = T.Cast("int32", h_slot[p]) if use_zigzag else p
                    m_idx = T.Cast("int32", m_slot[p]) if use_zigzag else p
                    s = i_s % num_stages
                    T.barrier_wait(data_is_free[s], (i_s // num_stages + 1) % 2)
                    if state_v_first:
                        T.tma_copy(ht_buffer[h_idx, bh, lo:lo + block_DV, 0:DK],
                                   h_stage[s, :, :], barrier=data_is_ready[s])
                    else:
                        T.tma_copy(ht_buffer[h_idx, bh, 0:DK, lo:lo + block_DV],
                                   h_stage[s, :, :], barrier=data_is_ready[s])

                    if fallback_mask[h_idx, bh]:
                        T.tma_copy(mt_buffer[m_idx, bh, 0:DK, 0:DK],
                                   m_stage[s, :, :], barrier=data_is_ready[s])
                    T.barrier_arrive(data_is_ready[s])

            elif tx < 192:
                # ==================== store warp ====================
                if store_intra_h0:
                    for i_s in T.serial(num_iters):
                        p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                        # Under zigzag every chunk is scanned (the running product needs
                        # them all) but only the ones this rank owns are written out;
                        # out_slot is -1 for the rest. That is what removes the trailing
                        # index_select and shrinks the output to n_local rows.
                        w_idx = T.Cast("int32", out_slot[p]) if use_zigzag else p
                        s = i_s % num_stages
                        T.barrier_wait(hprev_ready[s], (i_s // num_stages + 0) % 2)

                        if w_idx >= 0:
                            if state_v_first:
                                T.copy(hp_stage[s, :, :],
                                       cp_h0[w_idx, bh, lo:lo + block_DV, 0:DK])
                            else:
                                T.copy(hp_stage[s, :, :],
                                       cp_h0[w_idx, bh, 0:DK, lo:lo + block_DV])
                        T.barrier_arrive(hprev_free[s])

    def _grid_prologue(bbhv, seq_map_r2c):
        bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
        bb, bh = bbh // H, bbh % H
        seq_start_idx = T.Cast("int32", seq_map_r2c[bb])
        seq_end_idx = T.Cast("int32", seq_map_r2c[bb + 1])
        num_iters = seq_end_idx - seq_start_idx
        return bb, bh, bv, seq_start_idx, seq_end_idx, num_iters

    grid = T.ceildiv(DV, block_DV) * H * raw_batch_size

    if use_zigzag:
        # zigzag inter fold. ht_buffer/mt_buffer are the raw all-gather output and the
        # doc-chunk order lives in the slot maps, so nothing is physically reordered
        # before the scan. h and M are two row ranges of ONE cat-packed tensor, which is
        # why the caller binds the same tensor to both -- h_slot and m_slot carry the
        # offset. Only this rank's chunks are written (out_slot is -1 elsewhere), so
        # cp_h0 has n_local rows rather than the gathered buffer's.
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
                with T.Kernel(grid, threads=WS_THREADS) as (bbhv,):
                    bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                    scan_body(bb, bh, bv, s0, s1, ni, raw_h0, ht_buffer, mt_buffer,
                              fallback_mask, cp_h0, ht_buffer, mt_buffer,
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
                with T.Kernel(grid, threads=WS_THREADS) as (bbhv,):
                    bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                    # cp_h0 doubles as the unused raw_h0 slot (same dtype, never read).
                    scan_body(bb, bh, bv, s0, s1, ni, cp_h0, ht_buffer, mt_buffer,
                              fallback_mask, cp_h0, ht_buffer, mt_buffer,
                              h_slot, m_slot, out_slot)

    elif store_inter_h:
        if compute_m:
            @T.prim_func
            def tilelang_correct_h0_kernel(
                raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
                ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
                mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
                fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
                seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
                h_card: T.Tensor(raw_state_shape, dtype=res_dtype),
                m_card: T.Tensor([raw_batch_size, H, DK, DK], dtype=res_dtype),
            ):
                with T.Kernel(grid, threads=WS_THREADS) as (bbhv,):
                    bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                    scan_body(bb, bh, bv, s0, s1, ni, raw_h0, ht_buffer, mt_buffer,
                              fallback_mask, ht_buffer, h_card, m_card,
                              seq_map_r2c, seq_map_r2c, seq_map_r2c)
        else:
            @T.prim_func
            def tilelang_correct_h0_kernel(
                raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
                ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
                mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
                fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
                seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
                h_card: T.Tensor(raw_state_shape, dtype=res_dtype),
            ):
                with T.Kernel(grid, threads=WS_THREADS) as (bbhv,):
                    bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                    scan_body(bb, bh, bv, s0, s1, ni, raw_h0, ht_buffer, mt_buffer,
                              fallback_mask, ht_buffer, h_card, mt_buffer,
                              seq_map_r2c, seq_map_r2c, seq_map_r2c)

    elif use_raw_h0:
        # correct mode with a raw_h0 seed. h_card / m_card slots use placeholders.
        @T.prim_func
        def tilelang_correct_h0_kernel(
            raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
            ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor(state_shape, dtype=res_dtype),
        ):
            with T.Kernel(grid, threads=WS_THREADS) as (bbhv,):
                bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                scan_body(bb, bh, bv, s0, s1, ni, raw_h0, ht_buffer, mt_buffer,
                          fallback_mask, cp_h0, ht_buffer, mt_buffer,
                          seq_map_r2c, seq_map_r2c, seq_map_r2c)

    else:
        # correct mode, zero seed.
        @T.prim_func
        def tilelang_correct_h0_kernel(
            ht_buffer: T.Tensor(state_shape, dtype=h_buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=m_buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor(state_shape, dtype=res_dtype),
        ):
            with T.Kernel(grid, threads=WS_THREADS) as (bbhv,):
                bb, bh, bv, s0, s1, ni = _grid_prologue(bbhv, seq_map_r2c)
                # cp_h0 doubles as the unused raw_h0 slot (same dtype, never read).
                scan_body(bb, bh, bv, s0, s1, ni, cp_h0, ht_buffer, mt_buffer,
                          fallback_mask, cp_h0, ht_buffer, mt_buffer,
                          seq_map_r2c, seq_map_r2c, seq_map_r2c)

    return tilelang_correct_h0_kernel


def correct_initial_states(
    raw_h0: torch.Tensor
    | None,  # [raw_batch_size, num_v_heads, k_head_dim, v_head_dim]
    ht_buffer: torch.Tensor,  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]
    mt_buffer: torch.Tensor,  # [cp_batch_size, num_v_heads, k_head_dim, k_head_dim]
    fallback_mask: torch.Tensor,  # [cp_batch_size, num_v_heads]
    seq_map_r2c: torch.Tensor,  # [raw_batch_size + 1]
    state_v_first: bool = False,
):
    cp_batch_size = fallback_mask.shape[0]
    _, num_heads, dim_2, dim_3 = ht_buffer.shape
    if state_v_first:
        v_head_dim, k_head_dim = dim_2, dim_3
    else:
        k_head_dim, v_head_dim = dim_2, dim_3
    assert k_head_dim == v_head_dim == 128

    if raw_h0 is None:
        res_dtype = torch.float32
        use_raw_h0 = False
    else:
        res_dtype = raw_h0.dtype
        use_raw_h0 = True

    tilelang_correct_h0_kernel = tilelang_correct_h0(
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
    )
    cp_h0 = torch.empty(
        (cp_batch_size, num_heads, v_head_dim, k_head_dim)
        if state_v_first
        else (cp_batch_size, num_heads, k_head_dim, v_head_dim),
        dtype=res_dtype,
        device=ht_buffer.device,
    )
    if use_raw_h0:
        tilelang_correct_h0_kernel(
            raw_h0,
            ht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c,
            cp_h0,
        )
    else:
        tilelang_correct_h0_kernel(
            ht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c,
            cp_h0,
        )

    return cp_h0


def correct_terminal_states(
    raw_dht: torch.Tensor
    | None,  # [raw_batch_size, num_v_heads, k_head_dim, v_head_dim]
    dht_buffer: torch.Tensor,  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]
    mt_buffer: torch.Tensor,  # [cp_batch_size, num_v_heads, k_head_dim, k_head_dim]
    fallback_mask: torch.Tensor,  # [cp_batch_size, num_v_heads]
    seq_map_r2c: torch.Tensor,  # [raw_batch_size + 1]
    state_v_first: bool = False,
):
    cp_batch_size = fallback_mask.shape[0]
    _, num_heads, dim_2, dim_3 = dht_buffer.shape
    if state_v_first:
        v_head_dim, k_head_dim = dim_2, dim_3
    else:
        k_head_dim, v_head_dim = dim_2, dim_3
    assert k_head_dim == v_head_dim == 128

    if raw_dht is None:
        res_dtype = torch.float32
        use_raw_h0 = False
    else:
        res_dtype = raw_dht.dtype
        use_raw_h0 = True

    tilelang_correct_ht_kernel = tilelang_correct_h0(
        H=num_heads,
        DK=k_head_dim,
        DV=v_head_dim,
        res_dtype=res_dtype,
        accum_dtype="float32",
        h_buffer_dtype=dht_buffer.dtype,
        m_buffer_dtype=mt_buffer.dtype,
        seqlen_dtype=seq_map_r2c.dtype,
        mask_dtype=fallback_mask.dtype,
        use_raw_h0=use_raw_h0,
        state_v_first=state_v_first,
        reverse=True,
        transpose_m=True,
    )
    cp_dht = torch.empty(
        (cp_batch_size, num_heads, v_head_dim, k_head_dim)
        if state_v_first
        else (cp_batch_size, num_heads, k_head_dim, v_head_dim),
        dtype=res_dtype,
        device=dht_buffer.device,
    )
    if use_raw_h0:
        tilelang_correct_ht_kernel(
            raw_dht,
            dht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c,
            cp_dht,
        )
    else:
        tilelang_correct_ht_kernel(
            dht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c,
            cp_dht,
        )

    return cp_dht


# ---------------------------------------------------------------------------
# zigzag inter fold: scan straight out of the all-gather buffer
# ---------------------------------------------------------------------------
def _correct_states_zigzag(
    raw_state: torch.Tensor | None,  # [num_docs, H, K, V] (or [.., V, K])
    h_buffer: torch.Tensor,          # all-gathered h rows
    m_buffer: torch.Tensor,          # all-gathered M rows; may be the same tensor as h_buffer
    fold_seq_map: torch.Tensor,      # [num_docs + 1]
    h_slot: torch.Tensor,            # [num_docs * num_chunks] -> row of h_buffer
    m_slot: torch.Tensor,            # [num_docs * num_chunks] -> row of m_buffer
    out_slot: torch.Tensor,          # likewise -> output row, or -1 to skip
    n_local: int,
    state_v_first: bool,
    reverse: bool,
    transpose_m: bool,
):
    """Shared body of the two zigzag entry points.

    The caller passes the raw all-gather output and the slot maps that address it, so no
    reindex is materialised. When h and M share a dtype they ride one cat-packed tensor
    and ``h_buffer is m_buffer``, the two maps differing by the n_local row offset; when
    they do not (backward's fp32 dh against a bf16 M) they are separate gathers and both
    maps are the same. The kernel does not care which -- it just follows the maps.
    """
    n_rows, num_heads, dim_2, dim_3 = h_buffer.shape
    if state_v_first:
        v_head_dim, k_head_dim = dim_2, dim_3
    else:
        k_head_dim, v_head_dim = dim_2, dim_3
    assert k_head_dim == v_head_dim == 128, (
        "the zigzag fold addresses h and M by row, which needs K == V == 128"
    )
    assert m_buffer.shape[0] == n_rows, (h_buffer.shape, m_buffer.shape)

    use_raw_h0 = raw_state is not None
    res_dtype = raw_state.dtype if use_raw_h0 else torch.float32

    # every gathered chunk is a document-interior piece, so the fold never falls back.
    fallback_mask = torch.ones(
        (n_rows, num_heads), dtype=torch.bool, device=h_buffer.device
    )

    kernel = tilelang_correct_h0(
        H=num_heads,
        DK=k_head_dim,
        DV=v_head_dim,
        res_dtype=res_dtype,
        accum_dtype="float32",
        h_buffer_dtype=h_buffer.dtype,
        m_buffer_dtype=m_buffer.dtype,
        seqlen_dtype=h_slot.dtype,
        mask_dtype=fallback_mask.dtype,
        use_raw_h0=use_raw_h0,
        state_v_first=state_v_first,
        reverse=reverse,
        transpose_m=transpose_m,
        use_zigzag=True,
    )
    out = torch.empty(
        (n_local, num_heads, v_head_dim, k_head_dim)
        if state_v_first
        else (n_local, num_heads, k_head_dim, v_head_dim),
        dtype=res_dtype,
        device=h_buffer.device,
    )
    args = [h_buffer, m_buffer, fallback_mask, fold_seq_map,
            h_slot, m_slot, out_slot, out]
    if use_raw_h0:
        args.insert(0, raw_state)
    kernel(*args)
    return out


def correct_initial_states_zigzag(
    raw_h0: torch.Tensor | None,
    h_buffer: torch.Tensor,
    m_buffer: torch.Tensor,
    fold_seq_map: torch.Tensor,
    h_slot: torch.Tensor,
    m_slot: torch.Tensor,
    out_slot: torch.Tensor,
    n_local: int,
    state_v_first: bool = False,
):
    return _correct_states_zigzag(
        raw_h0, h_buffer, m_buffer, fold_seq_map, h_slot, m_slot, out_slot, n_local,
        state_v_first=state_v_first, reverse=False, transpose_m=False,
    )


def correct_terminal_states_zigzag(
    raw_dht: torch.Tensor | None,
    h_buffer: torch.Tensor,
    m_buffer: torch.Tensor,
    fold_seq_map: torch.Tensor,
    h_slot: torch.Tensor,
    m_slot: torch.Tensor,
    out_slot: torch.Tensor,
    n_local: int,
    state_v_first: bool = False,
):
    return _correct_states_zigzag(
        raw_dht, h_buffer, m_buffer, fold_seq_map, h_slot, m_slot, out_slot, n_local,
        state_v_first=state_v_first, reverse=True, transpose_m=True,
    )


# ---------------------------------------------------------------------------
# card-level (h, M) aggregation for inter+intra CP
# ---------------------------------------------------------------------------
def aggregate_card_state(
    ht_buffer: torch.Tensor,   # [cp_batch_size, H, K, V] (or [.., V, K])
    mt_buffer: torch.Tensor,   # [cp_batch_size, H, K, K]
    fallback_mask: torch.Tensor,  # [cp_batch_size, H]
    seq_map_r2c: torch.Tensor,  # [raw_batch_size + 1]
    raw_h0: torch.Tensor | None = None,  # [raw_batch_size, H, K, V] (or [.., V, K])
    state_v_first: bool = False,
    reverse: bool = False,
    transpose_m: bool = False,
    compute_m: bool = True,
):
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

    state_kernel = tilelang_correct_h0(
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
        compute_m=compute_m,
    )
    raw_state_shape = (
        (raw_batch_size, num_heads, v_head_dim, k_head_dim)
        if state_v_first
        else (raw_batch_size, num_heads, k_head_dim, v_head_dim)
    )
    h_card = torch.empty(raw_state_shape, dtype=res_dtype, device=ht_buffer.device)
    if not use_raw_h0:
        raw_h0 = torch.empty(raw_state_shape, dtype=res_dtype, device=ht_buffer.device)
    if compute_m:
        m_card = torch.empty(
            (raw_batch_size, num_heads, k_head_dim, k_head_dim),
            dtype=res_dtype,
            device=mt_buffer.device,
        )
        state_kernel(raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, h_card, m_card)
        return h_card, m_card

    state_kernel(raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c, h_card)
    return h_card, None
