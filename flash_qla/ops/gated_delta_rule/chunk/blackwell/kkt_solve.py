# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

# Independent TileLang implementation of the FlashInfer blockwise inverse:
# flashinfer-ai/flashinfer@07cc0d60a9ba1aca87b7b803ab1810ca5e3738d6,
# gated_delta_net_chunked.py, _invert_diagonal_NxN / _blockwise_diagonal_*.
# FP32 Gram/row elimination; FP16 workspace and MMA inputs, FP32 accumulation.

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
    assert DK in (64, 128)
    assert chunk_size in (32, 64)
    assert accum_dtype == "float32"

    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def merge_inverse_blocks(a_shared, block_N):
        # [D0, 0; L10, D1]^-1 has X10 = -X11 @ L10 @ X00.
        # Diagonal blocks in a_shared already contain X00 and X11.
        thread_idx = T.get_thread_binding()
        lane_idx = thread_idx % 32
        warp_idx = thread_idx // 32
        block_idx = warp_idx if block_N < 32 else 0
        row_offset = 0 if block_N < 32 else warp_idx * 16

        a_local = T.alloc_local((block_N // 2,), dtype="float16")
        b_local = T.alloc_local((block_N * block_N // 32,), dtype="float16")
        c_local = T.alloc_local((block_N // 2,), dtype=accum_dtype)

        # One warp per 8/16 block pair; two warps split a 32x32 result's rows.
        if thread_idx < (block_S // (2 * block_N) if block_N < 32 else 2) * 32:
            # Load X11 in the MMA A-operand lane/register layout.
            # For block_N=8, duplicate rows to fill the m16n8k8 instruction.
            for j_v in T.unroll(block_N // 2):
                j_s = (lane_idx // 4 + 8 * (j_v % 4 // 2)) % min(block_N, 16)
                j_t = lane_idx % 4 * 2 + j_v % 2 + 8 * (j_v // 4 % 2) + 16 * (j_v // 8)
                a_local[j_v] = a_shared[
                    (2 * block_idx + 1) * block_N + row_offset + j_s,
                    (2 * block_idx + 1) * block_N + j_t,
                ]

            # Load L10 in the MMA B-operand lane/register layout.
            for j_n in T.unroll(block_N // 8):
                for k_r in T.unroll(max(block_N // 16, 1)):
                    for j_v in T.unroll(2 if block_N == 8 else 4):
                        j_s = lane_idx % 4 * 2 + j_v % 2 + 8 * (j_v // 2) + k_r * 16
                        j_t = lane_idx // 4 + j_n * 8
                        b_local[j_n * (block_N // 4) + k_r * 4 + j_v] = a_shared[
                            (2 * block_idx + 1) * block_N + j_s,
                            2 * block_idx * block_N + j_t,
                        ]

            # First product: X11 @ L10.
            T.clear(c_local)
            for j_n in T.unroll(block_N // 8):
                for k_r in T.unroll(max(block_N // 16, 1)):
                    T.ptx_mma(
                        "float32", "m16n8k8" if block_N == 8 else "m16n8k16",
                        "row", "col", "fp16", "fp16", "fp32",
                        a_local.data, k_r * 8,
                        b_local.data, j_n * (block_N // 4) + k_r * 4,
                        c_local.data, j_n * 4, T.bool(False),
                    )

            # The accumulator layout also matches the next A operand.
            # Negate/cast in registers, without a shared-memory intermediate.
            for j_v in T.unroll(block_N // 2):
                a_local[j_v] = -c_local[j_v]
            for j_n in T.unroll(block_N // 8):
                for k_r in T.unroll(max(block_N // 16, 1)):
                    for j_v in T.unroll(2 if block_N == 8 else 4):
                        j_s = lane_idx % 4 * 2 + j_v % 2 + 8 * (j_v // 2) + k_r * 16
                        j_t = lane_idx // 4 + j_n * 8
                        b_local[j_n * (block_N // 4) + k_r * 4 + j_v] = a_shared[
                            2 * block_idx * block_N + j_s,
                            2 * block_idx * block_N + j_t,
                        ]

            # Second product: -(X11 @ L10) @ X00.
            T.clear(c_local)
            for j_n in T.unroll(block_N // 8):
                for k_r in T.unroll(max(block_N // 16, 1)):
                    T.ptx_mma(
                        "float32", "m16n8k8" if block_N == 8 else "m16n8k16",
                        "row", "col", "fp16", "fp16", "fp32",
                        a_local.data, k_r * 8,
                        b_local.data, j_n * (block_N // 4) + k_r * 4,
                        c_local.data, j_n * 4, T.bool(False),
                    )

        # All warps join both barriers: finish old reads before overwriting L10,
        # then publish X10 to the next merge and the final cooperative store.
        T.sync_threads()
        if thread_idx < (block_S // (2 * block_N) if block_N < 32 else 2) * 32:
            for j_v in T.unroll(block_N // 2):
                j_s = lane_idx // 4 + 8 * (j_v % 4 // 2)
                j_t = lane_idx % 4 * 2 + j_v % 2 + 8 * (j_v // 4)
                if j_s < block_N:
                    a_shared[
                        (2 * block_idx + 1) * block_N + row_offset + j_s,
                        2 * block_idx * block_N + j_t,
                    ] = c_local[j_v]
        T.sync_threads()

    @T.macro
    def kernel_body(
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
        b_shared = T.alloc_shared((block_S,), dtype=accum_dtype)
        a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        a_output = T.alloc_fragment((block_S, block_S), dtype=qkva_dtype)
        a_shared = T.alloc_shared((block_S, block_S + 4), dtype=accum_dtype)
        a8i_row = T.alloc_local((8,), dtype=accum_dtype)
        a8i_scale = T.alloc_local((1,), dtype=accum_dtype)

        T.annotate_layout({
            a_shared: tilelang.layout.make_linear_layout(a_shared),
            a_output: T.Fragment(
                (block_S, block_S),
                forward_thread_fn=lambda r, c: (r * (block_S // 4) + c // 4) % 128,
                forward_index_fn=lambda r, c: r // (512 // block_S) * 4 + c % 4,
            ),
        })
        thread_idx = T.get_thread_binding()

        # Load K; overlap full-chunk async copies with beta loads.
        if right <= seq_end_idx:
            T.async_copy(k[bb, left:right, bhg, 0:DK], k_shared)
        else:
            for j_s, j_k in T.Parallel(block_S, DK):
                k_shared[j_s, j_k] = T.if_then_else(
                    left + j_s < seq_end_idx, k[bb, left + j_s, bhg, j_k], 0
                )
        for j_s in T.Parallel(block_S):
            b_shared[j_s] = T.if_then_else(
                left + j_s < seq_end_idx, b[bb, left + j_s, bh], 0
            )

        # Gram matrix: K @ K^T, with FP32 accumulation.
        if right <= seq_end_idx:
            T.ptx_wait_group(0)
        T.gemm(k_shared, k_shared, a_fragment, transpose_B=True, clear_accum=True)

        # A = I + StrictLower(diag(beta) @ K @ K^T).
        for j_s, j_t in T.Parallel(block_S, block_S):
            a_fragment[j_s, j_t] = T.if_then_else(
                j_s > j_t, a_fragment[j_s, j_t] * b_shared[j_s],
                T.if_then_else(j_s == j_t, 1, 0),
            )
        for j_s, j_t in T.Parallel(block_S, block_S):
            a_shared[j_s, j_t] = a_fragment[j_s, j_t]
        T.sync_threads()

        # Diagonal (block_S / 8) x 8x8 inverses, one row per thread.
        # Shuffle stays outside the row predicate so each participating warp
        # executes it with all lanes; only the row updates are predicated.
        if thread_idx < block_S:
            for j_t in T.unroll(8):
                a8i_row[j_t] = a_shared[thread_idx, thread_idx // 8 * 8 + j_t]
            for k_s in T.unroll(7):
                a8i_scale[0] = -a8i_row[k_s]
                for j_t in T.unroll(k_s):
                    pivot_value = T.shfl_sync(a8i_row[j_t], k_s, width=8)
                    if thread_idx % 8 > k_s:
                        a8i_row[j_t] += a8i_scale[0] * pivot_value
                if thread_idx % 8 > k_s:
                    a8i_row[k_s] = a8i_scale[0]
            for j_t in T.unroll(8):
                a_shared[thread_idx, thread_idx // 8 * 8 + j_t] = a8i_row[j_t]
        T.sync_threads()

        # Hierarchical inverse: 8 -> 16 -> 32 [-> 64].
        merge_inverse_blocks(a_shared, 8)
        merge_inverse_blocks(a_shared, 16)
        if block_S == 64:
            merge_inverse_blocks(a_shared, 32)

        # Save A in the original layout and input dtype; mask tail rows.
        for j_s, j_t in T.Parallel(block_S, block_S):
            a_output[j_s, j_t] = a_shared[j_s, j_t]
        for j_s, j_t in T.Parallel(block_S, block_S):
            if left + j_s < seq_end_idx:
                a[bb, left + j_s, bh, j_t] = a_output[j_s, j_t]

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
                T.annotate_min_blocks_per_sm(8)
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                # Keep token coordinates int32 even for int64 offsets, avoiding
                # TileLang 0.1.12's int64-slice cp.async lowering issue.
                bb = 0
                batch_idx = T.cast(chunk_indices[bc, 0], "int32")
                chunk_idx = T.cast(chunk_indices[bc, 1], "int32")
                seq_start_idx = T.cast(cu_seqlens[batch_idx], "int32")
                seq_end_idx = T.cast(cu_seqlens[batch_idx + 1], "int32")

                kernel_body(
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
                T.annotate_min_blocks_per_sm(8)
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                # Keep indices as expressions so bounds inference can remove
                # redundant predicates from the full-chunk async copies.
                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
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
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H = b.shape
    assert K in (64, 128)
    assert chunk_size in (32, 64)
    assert num_tokens <= 2**31 - 1
    assert H % Hg == 0
    assert k.dtype in (torch.float16, torch.bfloat16)
    assert b.dtype in (torch.float32, k.dtype)

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
