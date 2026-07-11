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

        # Diagonal 16x16 inversion workspaces (2 blocks)
        a16i_row = T.alloc_fragment((2, 16), dtype=accum_dtype)
        a16i_sum = T.alloc_fragment((2, 16), dtype=accum_dtype)

        # 2 diagonal 16x16 blocks (padded to 17 for alignment)
        a16i_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        # 1 sub-diagonal 16x16 coupling (blocks 0->1)
        a16o_shared = T.alloc_shared((1, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((1, 16, 16), dtype=accum_dtype)


        a32_shared = T.alloc_shared((block_S, block_S), dtype=accum_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

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
        T.gemm(k_shared,k_shared,a32_fragment,transpose_B=True,clear_accum=True)

        # A = b * A
        for j_s, j_t in T.Parallel(block_S, block_S):
            a32_fragment[j_s, j_t] *= b_shared[j_s]

            # A = I + StrictLower(A)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a32_fragment[j_s, j_t] = 0
            elif j_s == j_t:
                a32_fragment[j_s, j_t] = 1

        # Prepare inversion input: extract diagonal and sub-diagonal 16x16 blocks
        for j_s, j_t in T.Parallel(block_S, block_S):
            if (j_s // 16) == (j_t // 16) + 1:
                    # Sub-diagonal block: L10
                a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a32_fragment[
                        j_s, j_t
                    ]
            elif (j_s // 16) == (j_t // 16):
                    # Diagonal blocks: L00, L11
                a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a32_fragment[
                        j_s, j_t
                    ]

        # Invert diagonal 16x16 blocks (2 blocks in parallel)
        # Forward substitution for lower triangular L with unit diagonal
        for k_s in T.unroll(1, 16):
            for j_s, k_t in T.Parallel(2, 16):
                if k_t < k_s:
                    a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
            T.clear(a16i_sum)
            for k_r in T.unroll(k_s):
                for j_s, k_t in T.Parallel(2, 16):
                    a16i_sum[j_s, k_t] -= (
                            a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                        )
            for j_s, k_t in T.Parallel(2, 16):
                if k_t < k_s:
                    a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

            # Compute coupling between blocks 0 and 1 (single pair)
            # Step 1: temp = L11^{-1} @ (-L10)  [16x16]
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(1, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16i_shared[j_s * 2 + 1, k_s, k_r]
                        * a16o_shared[j_s, k_r, k_t]
                    )
            # Transpose for the second multiply
        for j_s, k_s, k_t in T.Parallel(1, 16, 16):
            a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
            # Step 2: result = temp @ L00^{-1}  [16x16]
        T.clear(a16o_fragment)
        for k_r in T.unroll(16):
            for j_s, k_s, k_t in T.Parallel(1, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16o_shared[j_s, k_r, k_s]
                        * a16i_shared[j_s * 2, k_r, k_t]
                    )
        T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

            # Assemble 32x32 result
            # Top-left: L00^{-1}
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[k_s, k_t] = a16i_shared[0, k_s, k_t]
            # Top-right: 0
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[k_s, 16 + k_t] = 0
            # Bottom-left: coupling (-L11^{-1} @ L10 @ L00^{-1})
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[16 + k_s, k_t] = a16o_fragment[0, k_s, k_t]
            # Bottom-right: L11^{-1}
        for k_s, k_t in T.Parallel(16, 16):
            a32_shared[16 + k_s, 16 + k_t] = a16i_shared[1, k_s, k_t]

            

        # Save A (unmasked)
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
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
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
            with T.Kernel(num_chunks * H, threads=128) as (bch,):
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