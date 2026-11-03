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


def select_correction_config(
    H, raw_batch_size, DV, device=None, *, num_sms=None,
    compute_m=False,
):
    """Select precision and DV width from the independent grid for both layouts.

    AutoCP supplies num_sms to use the kernel policy without a device query.
    """
    if min(H, raw_batch_size, DV) <= 0:
        raise ValueError("Correction dimensions must be positive")
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")
    n_core = H * raw_batch_size * ((DV + 127) // 128)
    if not compute_m:
        if n_core * 8 <= num_sms:
            return "fp32", 16
        if n_core * 4 <= num_sms:
            return "tf32x3", 32
        if n_core * 2 <= num_sms:
            return "tf32x3", 64
        return "tf32x3", 128

    # Running-M aggregation uses its own crossover and at most DV64.
    if n_core * 4 < num_sms:
        return "fp32", 16
    return "tf32x3", 64


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
    compute_m: bool = False,
    use_slots: bool = False,
    precision: str = "tf32x3",
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

    if precision not in ("tf32x3", "fp32"):
        raise ValueError(f"Unsupported correction precision: {precision}")
    if block_DV is None:
        block_DV = 128 if precision == "tf32x3" else 16
    # Running-M needs another persistent DK x DK fp32 operand. Limit the
    # aggregate tile so the pipeline and TF32 scratch fit Hopper's shared memory.
    if compute_m:
        block_DV = min(block_DV, 64)
    block_DV = min(block_DV, DV)
    if num_stages is None:
        num_stages = 1

    tile_shape = (block_DV, DK) if state_v_first else (DK, block_DV)
    # TF32 MMA uses a V-by-K accumulator for both external state layouts.
    # Keep the TMA stages in the external layout; only the math view changes.
    transpose_h = precision == "tf32x3" and not state_v_first and not compute_m and store_intra_h0
    fragment_shape = (block_DV, DK) if transpose_h else tile_shape
    # Scalar FP32 wants adjacent lanes to access adjacent V values. VK needs
    # a separate K-by-V compute view so its output stage can retain native TMA.
    separate_h = precision == "fp32" and DK == 128 and block_DV == 16 and state_v_first and not compute_m
    fp32_vec = 8 if reverse else 1
    # A 64-column scan can fit a 64-wide K tile; the 128-column scan and
    # running-M chain need smaller conversion scratch to fit shared memory.
    block_K = 64 if not compute_m and block_DV <= 64 else 16
    scratch_M = DK if compute_m or not (state_v_first or transpose_h) else block_DV
    scratch_N = DK if compute_m or state_v_first or transpose_h else block_DV

    WS_THREADS = 256          # 128 math + 32 load + 32 store + 64 parked
    CONSUMER_NREG = 240
    PRODUCER_NREG = 24

    BAR_CONSUMER = 3

    @T.macro
    def precise_gemm(
        A, B, C, scratch, sa, sb,
        M: int, N: int, trans_A: bool, trans_B: bool, b_is_m: bool,
    ):
        if precision == "fp32":
            for kk in T.serial(DK):
                for i, j in T.Parallel(M, N):
                    av = T.cast(A[sa, kk, i] if trans_A else A[sa, i, kk], "float32")
                    bv = T.cast(B[sb, j, kk] if trans_B else B[sb, kk, j], "float32")
                    C[i, j] += av * bv
        else:
            A_hi, A_lo, B_hi, B_lo = scratch
            # Both operands are K-major, as required by Hopper TF32 WGMMA.
            # TileLang 0.1.12 does not support TF32 in its TCGEN5 dispatcher;
            # on Blackwell T.gemm with a fragment emits TF32 MMA.sync instead.
            # K tiling bounds scratch space independently of the state width.
            for kb in T.serial(T.ceildiv(DK, block_K)):
                # Only row-major M needs this source-contiguous mapping. The
                # running h and M states have compiler-selected shared layouts.
                for i, k in T.Parallel(
                    M, block_K,
                    loop_layout=_m_transpose_source_layout(M, block_K) if trans_A else None,
                ):
                    av = T.cast(A[sa, kb * block_K + k, i] if trans_A else A[sa, i, kb * block_K + k], "float32")
                    # Keep hi/lo local: rereading hi from shared scratch inserts
                    # unnecessary synchronization inside the conversion loop.
                    av_hi = T.cast(av, T.tfloat32)
                    av_lo = T.cast(av - T.cast(av_hi, "float32"), T.tfloat32)
                    A_hi[i, k] = av_hi
                    A_lo[i, k] = av_lo
                for j, k in T.Parallel(
                    N, block_K,
                    loop_layout=(
                        _m_transpose_source_layout(N, block_K)
                        if b_is_m and not trans_B else None
                    ),
                ):
                    bv = T.cast(B[sb, j, kb * block_K + k] if trans_B else B[sb, kb * block_K + k, j], "float32")
                    bv_hi = T.cast(bv, T.tfloat32)
                    bv_lo = T.cast(bv - T.cast(bv_hi, "float32"), T.tfloat32)
                    B_hi[j, k] = bv_hi
                    B_lo[j, k] = bv_lo
                T.sync_threads(BAR_CONSUMER, 128)
                # Add the smaller cross terms before the high product.
                T.gemm(A_hi[0:M, :], B_lo[0:N, :], C, transpose_B=True, clear_accum=False)
                T.gemm(A_lo[0:M, :], B_hi[0:N, :], C, transpose_B=True, clear_accum=False)
                T.gemm(A_hi[0:M, :], B_hi[0:N, :], C, transpose_B=True, clear_accum=False)
                T.sync_threads(BAR_CONSUMER, 128)

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
        hp_stage = T.alloc_shared((num_stages,) + tile_shape, dtype=accum_dtype)
        if transpose_h:
            hp_compute = T.reshape(hp_stage, (num_stages, block_DV, DK))
        if separate_h:
            h_compute = T.alloc_shared((1,) + tile_shape, dtype=accum_dtype)

        if not store_intra_h0:
            hd_shared = T.alloc_shared((1,) + tile_shape, dtype=accum_dtype)
        mrun_shared = T.alloc_shared((1, DK, DK), dtype=accum_dtype)

        if precision == "tf32x3":
            # Reuse conversion scratch for h and running-M products.
            gemm_a_hi = T.alloc_shared((scratch_M, block_K), T.tfloat32)
            gemm_a_lo = T.alloc_shared((scratch_M, block_K), T.tfloat32)
            gemm_b_hi = T.alloc_shared((scratch_N, block_K), T.tfloat32)
            gemm_b_lo = T.alloc_shared((scratch_N, block_K), T.tfloat32)
            gemm_scratch = (gemm_a_hi, gemm_a_lo, gemm_b_hi, gemm_b_lo)
        else:
            gemm_scratch = ()

        h_fragment = T.alloc_fragment(fragment_shape, dtype=accum_dtype)
        mrun_frag = T.alloc_fragment((DK, DK), dtype=accum_dtype)
        if separate_h:
            T.annotate_layout({
                h_compute: T.Layout((1, block_DV, DK), lambda s, row, col: (s, col, row)),
                h_fragment: T.Fragment(tile_shape,
                    forward_thread_fn=lambda row, col: ((col * block_DV + row) // fp32_vec) % 128,
                    forward_index_fn=lambda row, col: ((col * block_DV + row) // (fp32_vec * 128)) * fp32_vec + row % fp32_vec),
            })
        if (precision == "fp32" and DK == 128 and block_DV == 16
                and reverse and transpose_m and not state_v_first
                and not compute_m and store_intra_h0):
            # M is transposed in backward. A plain K-by-V compute stage and
            # vector8 fragments avoid the inferred layout's shared bank conflicts.
            T.annotate_layout({
                hp_stage: T.Layout((num_stages, DK, block_DV), lambda s, row, col: (s, row, col)),
                h_fragment: T.Fragment(tile_shape,
                    forward_thread_fn=lambda row, col: ((row * block_DV + col) // 8) % 128,
                    forward_index_fn=lambda row, col: ((row * block_DV + col) // 1024) * 8 + col % 8),
            })
        if (precision == "fp32" and DK == 128 and block_DV == 16
                and not reverse and not transpose_m and not state_v_first
                and not compute_m and store_intra_h0):
            # Keep the scan's accumulator layout independent of the raw-h0
            # copy. Its inferred vector8 layout makes each warp read more
            # distinct M rows at the same shared-memory bank. Vector4 also
            # preserves paired FP32 FMA instructions on Blackwell.
            T.annotate_layout({h_fragment: T.Fragment(
                tile_shape,
                forward_thread_fn=lambda i, j: ((i * block_DV + j) // 4) % 128,
                forward_index_fn=lambda i, j: ((i * block_DV + j) // 512) * 4 + j % 4,
            )})

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
                    if transpose_h:
                        for vi, ki in T.Parallel(block_DV, DK):
                            h_fragment[vi, ki] = raw_h0[bb, bh, ki, lo + vi]
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
                T.copy(mrun_frag, mrun_shared[0, :, :])

            for i_s in T.serial(num_iters):
                # Slot maps translate logical scan positions to physical input rows.
                p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                h_idx = T.Cast("int32", h_slot[p]) if use_slots else p
                s = i_s % num_stages

                if separate_h:
                    T.sync_threads(BAR_CONSUMER, 128)
                    if fallback_mask[h_idx, bh]:
                        T.copy(h_fragment, h_compute[0, :, :])

                if store_intra_h0:
                    T.barrier_wait(hprev_free[s], (i_s // num_stages + 1) % 2)
                    if transpose_h:
                        for vi, ki in T.Parallel(block_DV, DK):
                            hp_stage[s, ki, vi] = h_fragment[vi, ki]
                    else:
                        T.copy(h_fragment, hp_stage[s, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(hprev_ready[s])

                T.sync_threads(BAR_CONSUMER, 128)
                if not store_intra_h0:
                    if fallback_mask[h_idx, bh]:
                        T.copy(h_fragment, hd_shared[0, :, :])
                        T.fence_proxy_async()
                T.sync_threads(BAR_CONSUMER, 128)

                T.barrier_wait(data_is_ready[s], (i_s // num_stages + 0) % 2)
                if transpose_h:
                    # The store warp has finished reading the native KV state.
                    # Reuse those bytes for V-by-K math, without another buffer.
                    if fallback_mask[h_idx, bh]:
                        T.barrier_wait(hprev_free[s], (i_s // num_stages) % 2)
                        T.copy(h_fragment, hp_compute[s, :, :])
                        T.sync_threads(BAR_CONSUMER, 128)
                    for vi, ki in T.Parallel(block_DV, DK):
                        h_fragment[vi, ki] = h_stage[s, ki, vi]
                else:
                    T.copy(h_stage[s, :, :], h_fragment)      # h_fragment = h[h_idx]
                if fallback_mask[h_idx, bh]:
                    if transpose_h:
                        precise_gemm(hp_compute, m_stage, h_fragment,
                                     gemm_scratch, s, s,
                                     block_DV, DK, False, not transpose_m, True)
                    elif separate_h:
                        precise_gemm(h_compute, m_stage, h_fragment,
                                     gemm_scratch, 0, s,
                                     block_DV, DK, False, not transpose_m, True)
                    elif store_intra_h0:
                        if state_v_first:
                            precise_gemm(hp_stage, m_stage, h_fragment,
                                         gemm_scratch, s, s,
                                         block_DV, DK, False, not transpose_m, True)
                        else:
                            precise_gemm(m_stage, hp_stage, h_fragment,
                                         gemm_scratch, s, s,
                                         DK, block_DV, transpose_m, False, False)
                    else:
                        if state_v_first:
                            precise_gemm(hd_shared, m_stage, h_fragment,
                                         gemm_scratch, 0, s,
                                         block_DV, DK, False, not transpose_m, True)
                        else:
                            precise_gemm(m_stage, hd_shared, h_fragment,
                                         gemm_scratch, s, 0,
                                         DK, block_DV, transpose_m, False, False)

                if compute_m and bv == 0:
                    if fallback_mask[h_idx, bh]:
                        T.clear(mrun_frag)
                        precise_gemm(m_stage, mrun_shared, mrun_frag,
                                     gemm_scratch, s, 0,
                                     DK, DK, False, False, False)
                        T.sync_threads(BAR_CONSUMER, 128)   # WAR on mrun_shared
                        T.copy(mrun_frag, mrun_shared[0, :, :])
                        T.fence_proxy_async()
                    else:
                        T.clear(mrun_shared)
                    T.sync_threads(BAR_CONSUMER, 128)   # the next iteration reads it

                T.barrier_arrive(data_is_free[s])

            if store_inter_h:   # post-last-chunk exiting state = card-level h
                if state_v_first:
                    T.copy(h_fragment, h_card[bb, bh, lo:lo + block_DV, 0:DK])
                else:
                    if transpose_h:
                        for vi, ki in T.Parallel(block_DV, DK):
                            h_card[bb, bh, ki, lo + vi] = h_fragment[vi, ki]
                    else:
                        T.copy(h_fragment, h_card[bb, bh, 0:DK, lo:lo + block_DV])
            if compute_m and bv == 0:
                T.copy(mrun_shared[0, :, :], m_card[bb, bh, 0:DK, 0:DK])

        else:
            T.set_max_nreg(PRODUCER_NREG, 0)

            if tx < 160:
                # ==================== load warp ====================
                for i_s in T.serial(num_iters):
                    p = seq_end_idx - 1 - i_s if reverse else seq_start_idx + i_s
                    # Separate maps also support h and M sharing one packed buffer.
                    h_idx = T.Cast("int32", h_slot[p]) if use_slots else p
                    m_idx = T.Cast("int32", m_slot[p]) if use_slots else p
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
                        # Scan every position; negative output slots skip the write.
                        w_idx = T.Cast("int32", out_slot[p]) if use_slots else p
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

    if use_slots:
        # Read gathered buffers through slots without reindexing. Input rows may
        # share one packed tensor; output slots select and order the requested states.
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
        # Zigzag owns two chunks per document; ordinary inter-CP needs one boundary.
        output_rows = 2 * (seq_map_r2c.numel() - 1) if is_zigzag else 1
    else:
        output_rows = n_rows

    # Masks address physical h rows, including when h and M share a packed buffer.
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
        precision=precision, block_DV=block_DV,
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
    raw_h0: torch.Tensor | None,  # [raw_batch_size, H, K, V] (or [.., V, K])
    ht_buffer: torch.Tensor,  # [cp_batch_size, H, K, V] (or [.., V, K])
    mt_buffer: torch.Tensor,  # [cp_batch_size, H, K, K]
    fallback_mask: torch.Tensor | None,  # [cp_batch_size, H]; None means all True
    seq_map_r2c: torch.Tensor,  # [raw_batch_size + 1]
    state_v_first: bool = False,
    *,
    h_slot: torch.Tensor | None = None,
    m_slot: torch.Tensor | None = None,
    out_slot: torch.Tensor | None = None,
    is_zigzag: bool = False,
):
    """Forward correction in sequence order, optionally through slot maps.

    Supply all three maps together for indirect addressing. seq_map_r2c then
    partitions logical scan positions; h_slot/m_slot map each position to a
    physical input row, and out_slot maps it to an output row (-1 skips the write).
    Slot scans return one boundary state without a batch dimension, or a batch
    of two states per document when is_zigzag=True. Without slots, every input
    row has an output row and the batch dimension is retained.
    Maps share seq_map_r2c's dtype. fallback_mask always addresses physical h rows.
    Every output row must be mapped exactly once.
    """
    return _correct_states(
        raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
        state_v_first=state_v_first, reverse=False,
        h_slot=h_slot, m_slot=m_slot, out_slot=out_slot, is_zigzag=is_zigzag,
    )


def correct_terminal_states(
    raw_dht: torch.Tensor | None,  # [raw_batch_size, H, K, V] (or [.., V, K])
    dht_buffer: torch.Tensor,  # [cp_batch_size, H, K, V] (or [.., V, K])
    mt_buffer: torch.Tensor,  # [cp_batch_size, H, K, K]
    fallback_mask: torch.Tensor | None,  # [cp_batch_size, H]; None means all True
    seq_map_r2c: torch.Tensor,  # [raw_batch_size + 1]
    state_v_first: bool = False,
    *,
    h_slot: torch.Tensor | None = None,
    m_slot: torch.Tensor | None = None,
    out_slot: torch.Tensor | None = None,
    is_zigzag: bool = False,
):
    """Reverse correction with transposed M; slots follow correct_initial_states."""
    return _correct_states(
        raw_dht, dht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
        state_v_first=state_v_first, reverse=True,
        h_slot=h_slot, m_slot=m_slot, out_slot=out_slot, is_zigzag=is_zigzag,
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

    precision, block_DV = select_correction_config(
        num_heads, raw_batch_size, v_head_dim, ht_buffer.device,
        compute_m=compute_m,
    )
    state_kernel = tilelang_correct_h0(
        precision=precision, block_DV=block_DV,
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
