# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from typing import Optional

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_indices


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_kkt_solve(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)

    # Algorithm overview (32x32 unit lower-triangular inverse (I + L)^-1):
    # Instead of row-by-row forward substitution through shared memory (which
    # stalls on MIO queue and per-row barriers), we partition the 32 rows into
    # 4 horizontal 8x32 panels (one panel per warp, one column per lane):
    #   1. Each warp solves D_p * X_p = [-L_left | I | 0] for its 8x8 diagonal
    #      block D_p entirely in private FP32 registers (`column[8]`), with zero
    #      shared-memory writes or barriers during the 8-step elimination.
    #   2. Two hierarchical `merge_panels` passes (width 8 -> 16 -> 32) substitute
    #      upper panel solutions into lower panels, requiring only 5 barriers total.
    # Each warp owns all 32 columns of its row panel.
    threads = 128
    panel_rows = block_S // (threads // 32)

    @T.macro
    def merge_panels(panels, width):
        # Each diagonal block holds its inverse. Columns to its left hold
        # -inverse @ L_left. Substitute the upper block into the lower one.
        pairs = block_S // (2 * width)
        cols = block_S - width
        merged = T.alloc_fragment((pairs, width, cols), dtype=accum_dtype)
        for p, i, j in T.Parallel(pairs, width, cols):
            top = 2 * p * width
            merged[p, i, j] = T.if_then_else(j < top, panels[top + width + i, j], 0.0)
        for r in T.unroll(width):
            for p, i, j in T.Parallel(pairs, width, cols):
                top = 2 * p * width
                if j < top + width:
                    merged[p, i, j] += panels[top + width + i, top + r] * panels[top + r, j]
        # All reads of the coupling block must finish before overwriting it.
        T.sync_threads()
        for p, i, j in T.Parallel(pairs, width, cols):
            if j < (2 * p + 1) * width:
                panels[(2 * p + 1) * width + i, j] = merged[p, i, j]
        T.sync_threads()

    @T.macro
    def kernel_body_32(
        bb,
        bc,
        bh,
        bhg,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a32_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

        neg_gram = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
        a32_shared = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
        column = T.alloc_local((panel_rows,), dtype=accum_dtype)

        # Load K (async)
        if right <= seq_end_idx:
            T.async_copy(k[bb, left:right, bhg, 0:DK], k_shared)
        else:
            for j_s, j_k in T.Parallel(block_S, DK):
                if left + j_s < seq_end_idx:
                    k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                else:
                    k_shared[j_s, j_k] = 0
        # Load b (overlaps with K async copy)
        if right <= seq_end_idx:
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = b[bb, left + j_s, bh]
        else:
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = b[bb, left + j_s, bh]
                else:
                    b_shared[j_s] = 0
        # A = K @ K^T
        if right <= seq_end_idx:
            T.ptx_wait_group(0)
        T.gemm(k_shared, k_shared, a32_fragment, transpose_B=True, clear_accum=True)
        for i, j in T.Parallel(block_S, block_S):
            neg_gram[i, j] = -b_shared[i] * a32_fragment[i, j]
        T.sync_threads()

        # Solve D X = [-L_left | I | 0] for each diagonal panel D.
        # Each lane owns one RHS. The warp broadcasts each shared coefficient
        # while previous entries of the solution column stay in registers.
        tx = T.get_thread_binding()
        panel = (tx // block_S) * panel_rows
        j = tx % block_S
        for i in T.unroll(panel_rows):
            row = panel + i
            column[i] = T.if_then_else(
                j < panel, neg_gram[row, j], T.if_then_else(j == row, 1.0, 0.0)
            )
            for r in T.unroll(i):
                column[i] += neg_gram[row, panel + r] * column[r]
        for i in T.unroll(panel_rows):
            a32_shared[panel + i, j] = column[i]
        T.sync_threads()

        merge_panels(a32_shared, panel_rows)
        merge_panels(a32_shared, 2 * panel_rows)

        if right <= seq_end_idx:
            T.copy(a32_shared, a[bb, left:right, bh, 0:block_S])
        else:
            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a[bb, left + j_s, bh, j_t] = a32_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=threads) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]


                kernel_body_32(
                        bb,
                        bc,
                        bh,
                        bhg,
                        batch_idx,
                        chunk_idx,
                        seq_start_idx,
                        seq_end_idx,
                        k,
                        b,
                        a,
                    )

    else:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=threads) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body_32(
                        bb,
                        bc,
                        bh,
                        bhg,
                        batch_idx,
                        chunk_idx,
                        seq_start_idx,
                        seq_end_idx,
                        k,
                        b,
                        a,
                    )

    return tilelang_kkt_solve_kernel


def kkt_solve(
    k: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 32,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H = b.shape
    assert K == 128
    assert chunk_size == 32
    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    a = torch.empty(
        (batch_size, num_tokens, H, chunk_size), dtype=k.dtype, device=k.device
    )
    tilelang_kkt_solve_kernel = tilelang_kkt_solve(
        H,
        Hg,
        K,
        chunk_size,
        qkva_dtype=k.dtype,
        b_dtype=b.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
    )
    if is_varlen:
        tilelang_kkt_solve_kernel(k, b, cu_seqlens, chunk_indices, a)
    else:
        tilelang_kkt_solve_kernel(k, b, a, num_chunks)

    return a
