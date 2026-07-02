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
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_threads = tilelang.cdiv(num_heads, 32) * 32

    @T.prim_func
    def tilelang_get_warmup_chunks_bidi_kernel(
        g: T.Tensor([1, num_tokens, num_heads], dtype=g_dtype),
        ht_mask_fwd: T.Tensor([batch_size], dtype=mask_dtype),
        ht_mask_bwd: T.Tensor([batch_size], dtype=mask_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
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
):
    batch_size, num_tokens, num_heads = g.shape
    real_batch_size = ht_mask_fwd.shape[0]
    assert cu_seqlens.shape[0] == real_batch_size + 1
    assert batch_size == 1
    assert chunk_size == 32

    tilelang_kernel = tilelang_get_warmup_chunks_bidi(
        num_heads=num_heads,
        chunk_size=chunk_size,
        threshold=threshold,
        accum_dtype="float32",
        g_dtype=g.dtype,
        mask_dtype=ht_mask_fwd.dtype,
        seqlen_dtype=cu_seqlens.dtype,
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
        g, ht_mask_fwd, ht_mask_bwd, cu_seqlens,
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
    buffer_dtype,
    seqlen_dtype,
    mask_dtype,
    use_raw_h0,
    state_v_first,
    reverse: bool = False,
    transpose_m: bool = False,
    block_DV: int = 32,
):
    cp_batch_size = T.dynamic("cp_batch_size")
    raw_batch_size = T.dynamic("raw_batch_size")
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

    @T.macro
    def kernel_body(
        bb,
        bh,
        bv,
        seq_start_idx,
        seq_end_idx,
        num_iters,
        ht_buffer,
        mt_buffer,
        fallback_mask,
        seq_map_r2c,
        cp_h0,
        h_fragment,
    ):
        h_shared = T.alloc_shared(
            (block_DV, DK) if state_v_first else (DK, block_DV),
            dtype=buffer_dtype,
        )
        hd_shared = T.alloc_shared(
            (block_DV, DK) if state_v_first else (DK, block_DV),
            dtype=buffer_dtype,
        )
        m_shared = T.alloc_shared((DK, DK), dtype=buffer_dtype)

        DV_start = bv * block_DV
        DV_end = (bv + 1) * block_DV

        for i_s in T.Pipelined(num_iters - 1, num_stages=2):
            idx = seq_start_idx + num_iters - 1 - i_s if reverse else seq_start_idx + i_s
            if state_v_first:
                T.copy(
                    h_fragment,
                    cp_h0[idx, bh, DV_start:DV_end, 0:DK],
                )
            else:
                T.copy(
                    h_fragment,
                    cp_h0[idx, bh, 0:DK, DV_start:DV_end],
                )
            if state_v_first:
                T.copy(
                    ht_buffer[idx, bh, DV_start:DV_end, 0:DK],
                    h_shared,
                )
            else:
                T.copy(
                    ht_buffer[idx, bh, 0:DK, DV_start:DV_end],
                    h_shared,
                )
            # TODO: manually WASP
            T.copy(mt_buffer[idx, bh, 0:DK, 0:DK], m_shared)
            if fallback_mask[idx, bh]:
                T.copy(h_fragment, hd_shared)
            T.copy(h_shared, h_fragment)
            if fallback_mask[idx, bh]:
                if state_v_first:
                    if transpose_m:
                        T.gemm(hd_shared, m_shared, h_fragment, clear_accum=False)
                    else:
                        T.gemm(hd_shared, m_shared, h_fragment, transpose_B=True, clear_accum=False)
                else:
                    if transpose_m:
                        T.gemm(m_shared, hd_shared, h_fragment, transpose_A=True, clear_accum=False)
                    else:
                        T.gemm(m_shared, hd_shared, h_fragment, clear_accum=False)

        last_idx = seq_start_idx if reverse else seq_start_idx + num_iters - 1
        if state_v_first:
            T.copy(
                h_fragment,
                cp_h0[last_idx, bh, DV_start:DV_end, 0:DK],
            )
        else:
            T.copy(
                h_fragment,
                cp_h0[last_idx, bh, 0:DK, DV_start:DV_end],
            )

    if use_raw_h0:

        @T.prim_func
        def tilelang_correct_h0_kernel(
            raw_h0: T.Tensor(raw_state_shape, dtype=res_dtype),
            ht_buffer: T.Tensor(state_shape, dtype=buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor(state_shape, dtype=res_dtype),
        ):
            with T.Kernel(
                T.ceildiv(DV, block_DV) * H * raw_batch_size, threads=128
            ) as (bbhv,):
                bbh, bv = (
                    bbhv // T.ceildiv(DV, block_DV),
                    bbhv % T.ceildiv(DV, block_DV),
                )
                bb, bh = bbh // H, bbh % H

                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")
                num_iters = T.alloc_var("int32")
                seq_start_idx = seq_map_r2c[bb]
                seq_end_idx = seq_map_r2c[bb + 1]
                num_iters = seq_end_idx - seq_start_idx

                h_fragment = T.alloc_fragment(
                    (block_DV, DK) if state_v_first else (DK, block_DV),
                    dtype=accum_dtype,
                )
                if state_v_first:
                    T.copy(
                        raw_h0[bb, bh, bv * block_DV : (bv + 1) * block_DV, 0:DK],
                        h_fragment,
                    )
                else:
                    T.copy(
                        raw_h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                        h_fragment,
                    )

                kernel_body(
                    bb,
                    bh,
                    bv,
                    seq_start_idx,
                    seq_end_idx,
                    num_iters,
                    ht_buffer,
                    mt_buffer,
                    fallback_mask,
                    seq_map_r2c,
                    cp_h0,
                    h_fragment,
                )

    else:

        @T.prim_func
        def tilelang_correct_h0_kernel(
            ht_buffer: T.Tensor(state_shape, dtype=buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor(state_shape, dtype=res_dtype),
        ):
            with T.Kernel(
                T.ceildiv(DV, block_DV) * H * raw_batch_size, threads=128
            ) as (bbhv,):
                bbh, bv = (
                    bbhv // T.ceildiv(DV, block_DV),
                    bbhv % T.ceildiv(DV, block_DV),
                )
                bb, bh = bbh // H, bbh % H

                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")
                num_iters = T.alloc_var("int32")
                seq_start_idx = seq_map_r2c[bb]
                seq_end_idx = seq_map_r2c[bb + 1]
                num_iters = seq_end_idx - seq_start_idx

                h_fragment = T.alloc_fragment(
                    (block_DV, DK) if state_v_first else (DK, block_DV),
                    dtype=accum_dtype,
                )
                T.clear(h_fragment)

                kernel_body(
                    bb,
                    bh,
                    bv,
                    seq_start_idx,
                    seq_end_idx,
                    num_iters,
                    ht_buffer,
                    mt_buffer,
                    fallback_mask,
                    seq_map_r2c,
                    cp_h0,
                    h_fragment,
                )

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
        buffer_dtype=ht_buffer.dtype,
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
        buffer_dtype=dht_buffer.dtype,
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
