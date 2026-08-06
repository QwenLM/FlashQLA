# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_offsets


@tilelang.jit(
    # out_idx=[-5, -4, -3, -2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    },
)
def tilelang_fused_chunk_gdr_bwd(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h_dtype,
    o_dtype,
    seqlen_dtype,
    is_varlen,
    use_dht,
    state_v_first,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        o_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        h_shape = (
            (1, num_chunks, H, DV, DK)
            if state_v_first
            else (1, num_chunks, H, DK, DV)
        )
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        o_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        h_shape = (
            (batch_size, num_chunks, H, DV, DK)
            if state_v_first
            else (batch_size, num_chunks, H, DK, DV)
        )
    h0_shape = (
        (batch_size, H, DV, DK)
        if state_v_first
        else (batch_size, H, DK, DV)
    )
    ht_shape = (
        (batch_size, H, DV, DK)
        if state_v_first
        else (batch_size, H, DK, DV)
    )

    @T.prim_func
    def tilelang_fused_chunk_gdr_bwd_kernel(
        do: T.Tensor(o_shape, dtype=o_dtype),
        dht: T.Tensor(ht_shape, dtype=accum_dtype),
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        dq: T.Tensor(v_shape, dtype=qkva_dtype),
        dk: T.Tensor(v_shape, dtype=qkva_dtype),
        dv: T.Tensor(v_shape, dtype=qkva_dtype),
        dg: T.Tensor(g_shape, dtype=g_dtype),
        db: T.Tensor(b_shape, dtype=b_dtype),
        dh0: T.Tensor(h0_shape, dtype=accum_dtype),
    ):
        with T.Kernel(batch_size * H, threads=512) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")
            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

            # 2+2+2+2 + 1 + 4 = 13 units
            do_shared = T.alloc_shared((block_S, DV), dtype=o_dtype)
            # q -> tmp_shared_2_1
            # q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            # h_shared holds one DK-half of S0 at a time (W1-W4 windows)
            h_shared = T.alloc_shared(
                (DV, DK // 2) if state_v_first else (DK // 2, DV),
                dtype=h_dtype,
            )
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            # 2 units
            dqkv_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            dg_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            db_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            # 1+1 + 2+2+2 + 4 = 12 units
            tmp_shared_1_1 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_1_2 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_1_3 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_2_1 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_2 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_3 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            # tmp_shared_4_1 holds one DK-half of dSt; only the merged S+K consumer touches it
            tmp_shared_4_1 = T.alloc_shared(
                (DV, DK // 2) if state_v_first else (DK // 2, DV),
                dtype=qkva_dtype,
            )

            # CONSUMER_K
            # dk is split into two DK-halves: half-width N-split gemms on a single
            # buffer infer inconsistent extended layouts; two separate fragments
            # avoid the conflict (mirrors the dh_fragment_0/1 split).
            dk_fragment_0 = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            dk_fragment_1 = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_1 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_half_1 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_tmp_1_full = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_tmp_1_0 = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            dg_tmp_1_1 = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            dg_last_local_1 = T.alloc_fragment((1), dtype=accum_dtype)
            dg_half_last_1 = T.alloc_fragment((1), dtype=accum_dtype)

            # CONSUMER_A
            mask_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dp_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            da_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            hi_fragment = T.alloc_fragment((block_S, block_S), dtype="uint16")
            lo_fragment = T.alloc_fragment((block_S, block_S), dtype="uint16")
            uint32_fragment = T.alloc_fragment((block_S, block_S), dtype="uint32")
            u_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            # dq split into DK-halves for the same reason as dk (see above)
            dq_fragment_0 = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            dq_fragment_1 = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            db_fragment = T.alloc_fragment((block_S), dtype=accum_dtype)
            odot_fragment_2_0 = T.alloc_fragment(
                (block_S, DK // 2), dtype=qkva_dtype
            )
            odot_fragment_2_1 = T.alloc_fragment(
                (block_S, DK // 2), dtype=qkva_dtype
            )
            dg_fragment_2 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_half_2 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_a_2 = T.alloc_fragment((block_S), dtype=accum_dtype)

            # CONSUMER_S
            # dh is split into two DK-halves so that no row/column offset access is
            # ever needed on a single fragment (mma C layout spreads the 128-wide DK
            # dim across warps, which breaks offset slicing injectivity).
            dh_fragment_0 = T.alloc_fragment(
                (DV, DK // 2) if state_v_first else (DK // 2, DV),
                dtype=accum_dtype,
            )
            dh_fragment_1 = T.alloc_fragment(
                (DV, DK // 2) if state_v_first else (DK // 2, DV),
                dtype=accum_dtype,
            )
            # _odot_fragment_3 = T.alloc_fragment(
            #     (DV, DK) if state_v_first else (DK, DV),
            #     dtype=accum_dtype,
            # )
            reduce_fragment = T.alloc_fragment(
                (DV, DK // 2) if state_v_first else (DK // 2, DV),
                dtype=accum_dtype,
            )
            dg_last_local_3 = T.alloc_fragment((1), dtype=accum_dtype)
            dg_row_tmp = T.alloc_fragment((256,), dtype=accum_dtype)
            g_last_local_3 = T.alloc_local((1), dtype=accum_dtype)

            # 16 stages + 8 h-window stages
            # arrive counts assume: merged S+K consumer = 256 threads (tx<256),
            # A consumer = 128 threads, each producer warp = 32 threads
            bar_00 = T.alloc_barrier(arrive_count=448)
            bar_01 = T.alloc_barrier(arrive_count=384)
            bar_02 = T.alloc_barrier(arrive_count=416)
            bar_03 = T.alloc_barrier(arrive_count=384)
            bar_04 = T.alloc_barrier(arrive_count=416)
            bar_05 = T.alloc_barrier(arrive_count=416)
            bar_06 = T.alloc_barrier(arrive_count=384)
            bar_07 = T.alloc_barrier(arrive_count=384)
            bar_08 = T.alloc_barrier(arrive_count=384)
            bar_09 = T.alloc_barrier(arrive_count=384)
            bar_10 = T.alloc_barrier(arrive_count=416)
            bar_11 = T.alloc_barrier(arrive_count=384)
            bar_12 = T.alloc_barrier(arrive_count=256)
            bar_13 = T.alloc_barrier(arrive_count=384)
            bar_14 = T.alloc_barrier(arrive_count=384)
            bar_15 = T.alloc_barrier(arrive_count=384)
            # h windows: bar_h{0,2,4,6} = "S0 DK-half ready" (h-producer 32 + TMA tx),
            # bar_h{1,3,5,7} = "window consumed" (A + merged S+K)
            bar_h0 = T.alloc_barrier(arrive_count=32)
            bar_h1 = T.alloc_barrier(arrive_count=384)
            bar_h2 = T.alloc_barrier(arrive_count=32)
            bar_h3 = T.alloc_barrier(arrive_count=384)
            bar_h4 = T.alloc_barrier(arrive_count=32)
            bar_h5 = T.alloc_barrier(arrive_count=384)
            bar_h6 = T.alloc_barrier(arrive_count=32)
            bar_h7 = T.alloc_barrier(arrive_count=384)

            T.annotate_layout(
                {
                    do_shared: tilelang.layout.make_swizzled_layout(do_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    h_shared: tilelang.layout.make_swizzled_layout(h_shared),
                    dqkv_shared: tilelang.layout.make_swizzled_layout(dqkv_shared),
                    tmp_shared_1_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_1
                    ),
                    tmp_shared_1_2: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_2
                    ),
                    tmp_shared_1_3: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_3
                    ),
                    tmp_shared_2_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_1
                    ),
                    tmp_shared_2_2: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_2
                    ),
                    tmp_shared_2_3: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_3
                    ),
                    tmp_shared_4_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_4_1
                    ),
                }
            )

            # T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_SK_NREG = 224
            CONSUMER_A_NREG = 184

            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    # q_shared
                    tmp_shared_2_1[j_s, j_k] = q[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bhg,
                        j_k,
                    ]
                else:
                    tmp_shared_2_1[j_s, j_k] = 0
            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    k_shared[j_s, j_k] = k[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bhg,
                        j_k,
                    ]
                else:
                    k_shared[j_s, j_k] = 0
            for j_s, j_t in T.Parallel(block_S, block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    a_shared[j_s, j_t] = a[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_t,
                    ]
                else:
                    a_shared[j_s, j_t] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    do_shared[j_s, j_v] = do[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_v,
                    ]
                else:
                    do_shared[j_s, j_v] = 0
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    g_shared[j_s] = g[
                        batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh
                    ]
                else:
                    g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    b_shared[j_s] = b[
                        batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh
                    ]
                else:
                    b_shared[j_s] = 0

            # Merged S+K consumer (tx < 256): K-phase → S-phase interleaved.
            # dh_fragment stays register-resident across iterations; tmp_shared_4_1
            # (one DK-half of dSt) is staged from it right before each half-gemm.
            if tx < 256:
                T.set_max_nreg(CONSUMER_SK_NREG, 1)

                if use_dht:
                    # T.copy seeding (hopper style): offset-column T.Parallel
                    # loads poison the inferred fragment layout on
                    # tilelang>=0.1.13 and break downstream reduce/copy
                    # projectability checks.
                    if state_v_first:
                        T.copy(dht[bb, bh, 0:DV, 0 : DK // 2], dh_fragment_0)
                        T.copy(dht[bb, bh, 0:DV, DK // 2 : DK], dh_fragment_1)
                    else:
                        T.copy(dht[bb, bh, 0 : DK // 2, 0:DV], dh_fragment_0)
                        T.copy(dht[bb, bh, DK // 2 : DK, 0:DV], dh_fragment_1)
                else:
                    T.clear(dh_fragment_0)
                    T.clear(dh_fragment_1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00 — g_exp / g_rev_exp; S2[S] dK
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                        )
                    if i_s > 0:
                        for j_s, j_k in T.Parallel(block_S, DK // 2):
                            dqkv_shared[j_s, j_k] = T.Cast(
                                qkva_dtype, dk_fragment_0[j_s, j_k]
                            )
                        for j_s, j_k in T.Parallel(block_S, DK // 2):
                            dqkv_shared[j_s, DK // 2 + j_k] = T.Cast(
                                qkva_dtype, dk_fragment_1[j_s, j_k]
                            )
                    T.barrier_arrive(bar_01)

                    # 01 — dV' = K @ dSt  (two half gemms; dSt staged from dh_fragment)
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            tmp_shared_4_1[j_v, j_k] = T.Cast(
                                qkva_dtype, dh_fragment_0[j_v, j_k]
                            )
                        T.gemm(
                            k_shared[:, : DK // 2],
                            tmp_shared_4_1,
                            dv_fragment,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            tmp_shared_4_1[j_v, j_k] = T.Cast(
                                qkva_dtype, dh_fragment_1[j_v, j_k]
                            )
                        T.gemm(
                            k_shared[:, DK // 2 :],
                            tmp_shared_4_1,
                            dv_fragment,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            tmp_shared_4_1[j_k, j_v] = T.Cast(
                                qkva_dtype, dh_fragment_0[j_k, j_v]
                            )
                        T.gemm(
                            k_shared[:, : DK // 2],
                            tmp_shared_4_1,
                            dv_fragment,
                            clear_accum=True,
                        )
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            tmp_shared_4_1[j_k, j_v] = T.Cast(
                                qkva_dtype, dh_fragment_1[j_k, j_v]
                            )
                        T.gemm(
                            k_shared[:, DK // 2 :],
                            tmp_shared_4_1,
                            dv_fragment,
                            clear_accum=False,
                        )
                    # dV' = g_last/g * dV'
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.barrier_arrive(bar_02)

                    # 02 — dV' += Pg^T @ dO
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    T.gemm(
                        tmp_shared_1_1,
                        do_shared,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_03)

                    # 03 — S2[1] dV'
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        tmp_shared_2_1[j_s, j_v] = T.Cast(
                            qkva_dtype, dv_fragment[j_s, j_v]
                        )
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        tmp_shared_2_1[j_s, DV // 2 + j_v] = T.Cast(
                            qkva_dtype, dv_fragment[j_s, DV // 2 + j_v]
                        )
                    T.barrier_arrive(bar_04)

                    # 04 — dV = Ag^T @ dV'; S2[S] dV
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    T.gemm(
                        tmp_shared_1_2,
                        tmp_shared_2_1,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        dqkv_shared[j_s, j_v] = T.Cast(
                            qkva_dtype, dv_fragment[j_s, j_v]
                        )
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        dqkv_shared[j_s, DV // 2 + j_v] = T.Cast(
                            qkva_dtype, dv_fragment[j_s, DV // 2 + j_v]
                        )
                    T.barrier_arrive(bar_05)

                    # 05 — dVg = -g * dV; dg += sum(dVg * U); S2[3] dVg
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # dVg = -g * dV
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] = (
                            -dv_fragment[j_s, j_v] * g_exp_shared[j_s]
                        )
                    # !!! do not use T.Parallel(block_S,DV), dg+= sum(dVg * U) 
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        dg_tmp_1_full[j_s, j_v] = (
                            tmp_shared_2_3[j_s, j_v]
                            * dv_fragment[j_s, j_v]
                        )
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        dg_tmp_1_full[j_s, DV // 2 + j_v] = (
                            tmp_shared_2_3[j_s, DV // 2 + j_v]
                            * dv_fragment[j_s, DV // 2 + j_v]
                        )
                    T.reduce_sum(dg_tmp_1_full, dg_fragment_1, dim=1, clear=True)
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] = dg_fragment_1[j_s]
                    # T.copy(dg_fragment_1,dg_shared),  Store dV
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        tmp_shared_2_3[j_s, j_v] = dv_fragment[j_s, j_v]
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        tmp_shared_2_3[j_s, DV // 2 + j_v] = dv_fragment[j_s, DV // 2 + j_v]
                    T.barrier_arrive(bar_06)

                    # 06 — S2[2] K
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    T.copy(k_shared, tmp_shared_2_2)
                    T.barrier_arrive(bar_07)

                    # 07 — dK = V' @ dSt^T  (two half gemms, split dk fragments)
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    if state_v_first:
                        T.copy(dh_fragment_0, tmp_shared_4_1)
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_fragment_0,
                            clear_accum=True,
                        )
                        T.copy(dh_fragment_1, tmp_shared_4_1)
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_fragment_1,
                            clear_accum=True,
                        )
                    else:
                        T.copy(dh_fragment_0, tmp_shared_4_1)
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_fragment_0,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.copy(dh_fragment_1, tmp_shared_4_1)
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_fragment_1,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    T.barrier_arrive(bar_08)

                    # 08 — dS0 = g_last * dS0; dg_last += sum(dS0 * S0)  [W1/W2]
                    # original Consumer A bar 1 2 3
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    g_last_local_3[0] = g_exp_shared[block_S - 1]
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            dh_fragment_0[j_v, j_k] *= g_last_local_3[0]
                            dh_fragment_1[j_v, j_k] *= g_last_local_3[0]
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            dh_fragment_0[j_k, j_v] *= g_last_local_3[0]
                            dh_fragment_1[j_k, j_v] *= g_last_local_3[0]
                    T.clear(reduce_fragment)
                    # W1: S0[:, :DK/2]
                    T.barrier_wait(bar_h0, (i_s + 0) % 2)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            reduce_fragment[j_v, j_k] += (
                                dh_fragment_0[j_v, j_k] * h_shared[j_v, j_k]
                            )
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            reduce_fragment[j_k, j_v] += (
                                dh_fragment_0[j_k, j_v] * h_shared[j_k, j_v]
                            )
                    # Arrive bar_h1 BEFORE waiting for W2: the W2 load is only
                    # issued after both consumers signal W1 consumption, and
                    # this group's W2 read happens later (after bar_h2).
                    T.barrier_arrive(bar_h1)
                    # W2: S0[:, DK/2:]
                    T.barrier_wait(bar_h2, (i_s + 0) % 2)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            reduce_fragment[j_v, j_k] += (
                                dh_fragment_1[j_v, j_k] * h_shared[j_v, j_k]
                            )
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            reduce_fragment[j_k, j_v] += (
                                dh_fragment_1[j_k, j_v] * h_shared[j_k, j_v]
                            )
                    T.barrier_arrive(bar_h3)
                    # dg_last += sum(dh * S0): stage the accumulator through
                    # shared memory and reduce manually. T.reduce_sum on the
                    # fragment (or a T.reshape of it) trips tilelang>=0.1.13's
                    # thread-owned-segment projectability check for the layouts
                    # inferred here. tmp_shared_4_1 is dead at this point (its
                    # last use was the dSt staging in step 07). Element-wise
                    # stores are used instead of T.copy because the fragment->
                    # shared copy lowering also hits the same ReduceOp check.
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            tmp_shared_4_1[j_v, j_k] = reduce_fragment[j_v, j_k]
                        for j in T.Parallel(256):
                            dg_row_tmp[j] = 0
                            for jm in T.serial(32):
                                dg_row_tmp[j] += tmp_shared_4_1[
                                    (j * 32 + jm) // (DK // 2),
                                    (j * 32 + jm) % (DK // 2),
                                ]
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            tmp_shared_4_1[j_k, j_v] = reduce_fragment[j_k, j_v]
                        for j in T.Parallel(256):
                            dg_row_tmp[j] = 0
                            for jm in T.serial(32):
                                dg_row_tmp[j] += tmp_shared_4_1[
                                    (j * 32 + jm) // DV,
                                    (j * 32 + jm) % DV,
                                ]
                    T.reduce_sum(dg_row_tmp, dg_last_local_3, dim=0, clear=True)
                    dg_shared[block_S - 1] += dg_last_local_3[0]
                    T.barrier_arrive(bar_11)

                    # 09 — dK = g_last/g * dK; dg -= sum(K * dK); dg_last
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dk_fragment_0[j_s, j_k] *= g_rev_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dk_fragment_1[j_s, j_k] *= g_rev_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dg_tmp_1_0[j_s, j_k] = (
                            -T.Cast(accum_dtype, k_shared[j_s, j_k])
                            * dk_fragment_0[j_s, j_k]
                        )
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dg_tmp_1_1[j_s, j_k] = (
                            -T.Cast(accum_dtype, k_shared[j_s, DK // 2 + j_k])
                            * dk_fragment_1[j_s, j_k]
                        )
                    T.reduce_sum(dg_tmp_1_0, dg_fragment_1, dim=1, clear=True)
                    T.reduce_sum(dg_tmp_1_1, dg_half_1, dim=1, clear=True)
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_1[j_s] + dg_half_1[j_s]
                    T.reduce_sum(dg_fragment_1, dg_last_local_1, dim=0, clear=True)
                    T.reduce_sum(dg_half_1, dg_half_last_1, dim=0, clear=True)
                    dg_shared[block_S - 1] -= (
                        dg_last_local_1[0] + dg_half_last_1[0]
                    )
                    T.barrier_arrive(bar_09)
                    T.barrier_wait(bar_09, (i_s + 0) % 2)

                    # 10 — dK += dVg @ S0^T  [W3/W4]
                    # W3: S0[:, :DK/2]
                    T.barrier_wait(bar_h4, (i_s + 0) % 2)
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_fragment_0,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_fragment_0,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_h5)
                    # W4: S0[:, DK/2:]
                    T.barrier_wait(bar_h6, (i_s + 0) % 2)
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_fragment_1,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_fragment_1,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_h7)
                    T.barrier_arrive(bar_10)
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    T.barrier_wait(bar_11, (i_s + 0) % 2)

                    # 11 — dS0 += K^T @ dVg (two half gemms, DK-split output)
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_2[:, : DK // 2],
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_2[:, DK // 2 :],
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_2[:, : DK // 2],
                            tmp_shared_2_3,
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_2[:, DK // 2 :],
                            tmp_shared_2_3,
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_12)
                    T.barrier_wait(bar_12, (i_s + 0) % 2)

                    # 12 — dK += dP^T @ Q (two half gemms, N-split output)
                    T.gemm(
                        tmp_shared_1_1,
                        tmp_shared_2_1[:, : DK // 2],
                        dk_fragment_0,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.gemm(
                        tmp_shared_1_1,
                        tmp_shared_2_1[:, DK // 2 :],
                        dk_fragment_1,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_13)
                    T.barrier_wait(bar_13, (i_s + 0) % 2)

                    # 13 — dOg = s * g * dO → S2[3]
                    for j_s, j_v in T.Parallel(block_S, DV):
                        tmp_shared_2_3[j_s, j_v] = (
                            scale * do_shared[j_s, j_v] * g_exp_shared[j_s]
                        )
                    T.barrier_arrive(bar_14)

                    # 14 — dS0 += Q^T @ dOg (two half gemms, DK-split output)
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_1[:, : DK // 2],
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_1[:, DK // 2 :],
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_1[:, : DK // 2],
                            tmp_shared_2_3,
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_1[:, DK // 2 :],
                            tmp_shared_2_3,
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_15)

                    # 15 — dK += dAs @ K (two half gemms, N-split output)
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    T.gemm(
                        tmp_shared_1_2,
                        tmp_shared_2_2[:, : DK // 2],
                        dk_fragment_0,
                        clear_accum=False,
                    )
                    T.gemm(
                        tmp_shared_1_2,
                        tmp_shared_2_2[:, DK // 2 :],
                        dk_fragment_1,
                        clear_accum=False,
                    )

                for j_s, j_k in T.Parallel(block_S, DK // 2):
                    if seq_start_idx + j_s < seq_end_idx:
                        dk[batch_idx, seq_start_idx + j_s, bh, j_k] = (
                            dk_fragment_0[j_s, j_k]
                        )
                for j_s, j_k in T.Parallel(block_S, DK // 2):
                    if seq_start_idx + j_s < seq_end_idx:
                        dk[
                            batch_idx, seq_start_idx + j_s, bh, DK // 2 + j_k
                        ] = dk_fragment_1[j_s, j_k]
                if use_dht:
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            dh0[bb, bh, j_v, j_k] = dh_fragment_0[j_v, j_k]
                            dh0[bb, bh, j_v, DK // 2 + j_k] = dh_fragment_1[
                                j_v, j_k
                            ]
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            dh0[bb, bh, j_k, j_v] = dh_fragment_0[j_k, j_v]
                            dh0[bb, bh, DK // 2 + j_k, j_v] = dh_fragment_1[
                                j_k, j_v
                            ]

            elif tx < 384:
                T.set_max_nreg(CONSUMER_A_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # P = Q @ K^T
                    # q_shared -> tmp_shared_2_1
                    T.gemm(
                        tmp_shared_2_1,
                        k_shared,
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_01)

                    # 01
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        mask_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            mask_fragment[j_s, j_t] = T.exp2(
                                mask_fragment[j_s, j_t] * 1.442695
                            )
                        else:
                            mask_fragment[j_s, j_t] = 0
                    # Pg = s * P * G
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale
                    # S1[1] Pg
                    T.copy(p_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    # Ab = Ar * b
                    T.copy(a_shared, a_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    # Ag = G * Ab
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    # S1[2] Ag
                    T.copy(a_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_03)
                    # issue 30 opt1: Q is reloaded from global in step 09 instead
                    # of being pre-staged here (avoids fragment layout seeding)
                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # U = K @ S0  (W1: S0[:, :DK/2])
                    T.barrier_wait(bar_h0, (i_s + 0) % 2)
                    if state_v_first:
                        T.gemm(
                            k_shared[:, : DK // 2],
                            h_shared,
                            u_fragment,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            k_shared[:, : DK // 2],
                            h_shared,
                            u_fragment,
                            clear_accum=True,
                        )
                    T.barrier_arrive(bar_h1)
                    # Ag was staged to S1[2] in step 02; releasing bar_04 here
                    # (instead of after the W2 gemm, which is deferred) keeps
                    # the merged S+K consumer and the kv-store warp moving.
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # W2: S0[:, DK/2:] — U must be complete before it is
                    # staged to S2[3] and used for W below; deferring this
                    # gemm past the V-reload left S2[3] (=U) and W holding
                    # only the W1 partial sum. Doing it here also consumes
                    # h_shared (W2) before bar_h3 releases the h-producer to
                    # overwrite it with W3.
                    T.barrier_wait(bar_h2, (i_s + 0) % 2)
                    if state_v_first:
                        T.gemm(
                            k_shared[:, DK // 2 :],
                            h_shared,
                            u_fragment,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            k_shared[:, DK // 2 :],
                            h_shared,
                            u_fragment,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_h3)
                    # S2[3] U
                    T.copy(u_fragment, tmp_shared_2_3)
                    # W = V - g * U (V reloaded from global; v_shared dropped to
                    # fit the 99KB smem budget)
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        u_fragment[j_s, DV // 2 + j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        if (
                            seq_start_idx
                            + (num_iters - 1 - i_s) * block_S
                            + j_s
                            < seq_end_idx
                        ):
                            u_fragment[j_s, j_v] += T.Cast(
                                accum_dtype,
                                v[
                                    batch_idx,
                                    seq_start_idx
                                    + (num_iters - 1 - i_s) * block_S
                                    + j_s,
                                    bh,
                                    j_v,
                                ],
                            )
                    for j_s, j_v in T.Parallel(block_S, DV // 2):
                        if (
                            seq_start_idx
                            + (num_iters - 1 - i_s) * block_S
                            + j_s
                            < seq_end_idx
                        ):
                            u_fragment[j_s, DV // 2 + j_v] += T.Cast(
                                accum_dtype,
                                v[
                                    batch_idx,
                                    seq_start_idx
                                    + (num_iters - 1 - i_s) * block_S
                                    + j_s,
                                    bh,
                                    DV // 2 + j_v,
                                ],
                            )
                    # S2[2] W
                    T.copy(u_fragment, tmp_shared_2_2)
                    T.barrier_arrive(bar_05)
                    
                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # dAg = dV' @ W^T
                    T.gemm(
                        tmp_shared_2_1,
                        tmp_shared_2_2,
                        da_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    # V' = Ag @ W
                    T.gemm(
                        tmp_shared_1_2, tmp_shared_2_2, u_fragment, clear_accum=True
                    )
                    # S2[1] V'
                    T.copy(u_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    # dPg = dO @ V'^T
                    T.gemm(
                        do_shared,
                        tmp_shared_2_1,
                        dp_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # dAb = G * dAg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    # dg += sum((dPg * P) - (dPg * P)^T)
                    T.copy(tmp_shared_1_1, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= dp_fragment[j_s, j_t]
                    # T.copy(p_fragment, tmp_shared_1_1)
                    # for j_s, j_t in T.Parallel(block_S, block_S):
                    #     p_fragment[j_s, j_t] -= tmp_shared_1_1[j_t, j_s]
                    # T.reduce_sum(p_fragment, dg_fragment_2, dim=1, clear=True)
                    # dP = s * G * dPg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= scale
                    # S1[1] dP
                    T.copy(dp_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_08)

                    # 08
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    # dQ = dO @ S0^T  (W3: S0[:, :DK/2])
                    T.barrier_wait(bar_h4, (i_s + 0) % 2)
                    if state_v_first:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_fragment_0,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_fragment_0,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    T.barrier_arrive(bar_h5)
                    # W4: S0[:, DK/2:]
                    T.barrier_wait(bar_h6, (i_s + 0) % 2)
                    if state_v_first:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_fragment_1,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_fragment_1,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    T.barrier_arrive(bar_h7)
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # dQ = s * g * dQ
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dq_fragment_0[j_s, j_k] *= g_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dq_fragment_1[j_s, j_k] *= g_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dq_fragment_0[j_s, j_k] *= scale
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dq_fragment_1[j_s, j_k] *= scale
                    # S2[1] Q
                    # dg += sum(Q * dQ); reload Q from global (tmp_shared_2_1 was
                    # overwritten), stage it back to S2[1] and fuse the multiply
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        if (
                            seq_start_idx
                            + (num_iters - 1 - i_s) * block_S
                            + j_s
                            < seq_end_idx
                        ):
                            tmp_shared_2_1[j_s, j_k] = q[
                                batch_idx,
                                seq_start_idx
                                + (num_iters - 1 - i_s) * block_S
                                + j_s,
                                bhg,
                                j_k,
                            ]
                            odot_fragment_2_0[j_s, j_k] = T.Cast(
                                qkva_dtype,
                                T.Cast(
                                    accum_dtype,
                                    q[
                                        batch_idx,
                                        seq_start_idx
                                        + (num_iters - 1 - i_s) * block_S
                                        + j_s,
                                        bhg,
                                        j_k,
                                    ],
                                )
                                * dq_fragment_0[j_s, j_k],
                            )
                        else:
                            tmp_shared_2_1[j_s, j_k] = 0
                            odot_fragment_2_0[j_s, j_k] = 0
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        if (
                            seq_start_idx
                            + (num_iters - 1 - i_s) * block_S
                            + j_s
                            < seq_end_idx
                        ):
                            tmp_shared_2_1[j_s, DK // 2 + j_k] = q[
                                batch_idx,
                                seq_start_idx
                                + (num_iters - 1 - i_s) * block_S
                                + j_s,
                                bhg,
                                DK // 2 + j_k,
                            ]
                            odot_fragment_2_1[j_s, j_k] = T.Cast(
                                qkva_dtype,
                                T.Cast(
                                    accum_dtype,
                                    q[
                                        batch_idx,
                                        seq_start_idx
                                        + (num_iters - 1 - i_s) * block_S
                                        + j_s,
                                        bhg,
                                        DK // 2 + j_k,
                                    ],
                                )
                                * dq_fragment_1[j_s, j_k],
                            )
                        else:
                            tmp_shared_2_1[j_s, DK // 2 + j_k] = 0
                            odot_fragment_2_1[j_s, j_k] = 0
                    T.reduce_sum(odot_fragment_2_0, dg_fragment_2, dim=1, clear=True)
                    T.reduce_sum(odot_fragment_2_1, dg_half_2, dim=1, clear=True)
                    T.barrier_arrive(bar_10)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    # dQ += dP @ K (two half gemms, N-split output)
                    T.gemm(
                        tmp_shared_1_1,
                        tmp_shared_2_2[:, : DK // 2],
                        dq_fragment_0,
                        clear_accum=False,
                    )
                    T.gemm(
                        tmp_shared_1_1,
                        tmp_shared_2_2[:, DK // 2 :],
                        dq_fragment_1,
                        clear_accum=False,
                    )
                    # S2[S] dQ
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dqkv_shared[j_s, j_k] = T.Cast(
                            qkva_dtype, dq_fragment_0[j_s, j_k]
                        )
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        dqkv_shared[j_s, DK // 2 + j_k] = T.Cast(
                            qkva_dtype, dq_fragment_1[j_s, j_k]
                        )
                    T.barrier_arrive(bar_11)

                    # 11, 12
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    # dAb * Ar
                    T.copy(a_shared, a_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= da_fragment[j_s, j_t]
                    T.copy(a_fragment, tmp_shared_1_3)
                    # dAb * Ab [ = G * dAg * Ab ]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    # dg += sum((dAb * Ab) - (dAb * Ab)^T)
                    # T.copy(a_fragment, tmp_shared_1_2)
                    # for j_s, j_t in T.Parallel(block_S, block_S):
                    #     a_fragment[j_s, j_t] -= tmp_shared_1_2[j_t, j_s]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] += p_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        x = T.reinterpret(a_fragment[j_s, j_t], dtype="uint32")
                        lo_fragment[j_s, j_t] = x & 0xffff
                        hi_fragment[j_s, j_t] = x >> 16
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            tmp_shared_1_2[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                hi_fragment[j_s, j_t * 2 + j_t_vec],
                                dtype=qkva_dtype,
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            hi_fragment[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                tmp_shared_1_2[j_t * 2 + j_t_vec, j_s],
                                dtype="uint16",
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            tmp_shared_1_2[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                lo_fragment[j_s, j_t * 2 + j_t_vec],
                                dtype=qkva_dtype,
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            lo_fragment[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                tmp_shared_1_2[j_t * 2 + j_t_vec, j_s],
                                dtype="uint16",
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        uint32_fragment[j_s, j_t] = (hi_fragment[j_s, j_t] << 16) + \
                            lo_fragment[j_s, j_t]
                        p_fragment[j_s, j_t] = T.reinterpret(
                            uint32_fragment[j_s, j_t],
                            dtype=accum_dtype,
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] -= p_fragment[j_s, j_t]
                    T.reduce_sum(a_fragment, dg_a_2, dim=1, clear=True)
                    # Sg[S] dg
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += (
                            dg_fragment_2[j_s] + dg_half_2[j_s] + dg_a_2[j_s]
                        )
                    # db = sum((dAb * Ar)^T)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = tmp_shared_1_3[j_t, j_s]
                    T.reduce_sum(a_fragment, db_fragment, dim=1, clear=True)
                    # dAr = dAb * b
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= b_shared[j_t]
                    # S1[2] dAr
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_13)

                    # 13
                    T.barrier_wait(bar_13, (i_s + 0) % 2)
                    # dA = -Ar^T @ dAr @ Ar^T
                    T.gemm(
                        a_shared,
                        tmp_shared_1_2,
                        da_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.gemm(
                        tmp_shared_1_2,
                        a_shared,
                        da_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    # At = K @ K^T
                    T.gemm(
                        tmp_shared_2_2,
                        tmp_shared_2_2,
                        a_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_14)

                    # 14
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s <= j_t:
                            da_fragment[j_s, j_t] = 0
                        else:
                            da_fragment[j_s, j_t] = -da_fragment[j_s, j_t]
                    # db += sum(dA * At)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= da_fragment[j_s, j_t]
                    T.reduce_sum(a_fragment, db_fragment, dim=1, clear=False)
                    T.copy(db_fragment, db_shared)
                    # dAt = b * dA
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= b_shared[j_s]
                    # dAs = dAt + dAt^T
                    T.copy(da_fragment, tmp_shared_1_2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] += tmp_shared_1_2[j_t, j_s]
                    # S1[1] dAs
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_15)
                    T.barrier_wait(bar_15, (i_s + 0) % 2)

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_wait(bar_03, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]

                        T.barrier_wait(bar_07, (i_s + 0) % 2)
                        T.tma_copy(
                            k[batch_idx, left:right, bhg, 0:DK],
                            k_shared,
                            barrier=bar_00,
                        )
                        # bar_10 -> bar_15,q_shared
                        T.barrier_wait(bar_15, (i_s + 0) % 2)
                        T.tma_copy(
                            q[batch_idx, left:right, bhg, 0:DK],
                            tmp_shared_2_1,
                            barrier=bar_00,
                        )

                    if num_iters > 0:
                        T.barrier_arrive(bar_00)

                elif tx < 384 + 64:  # TODO: set padding to 0
                    if bb == batch_size - 1:
                        for j_s, j_v in T.Parallel(block_S, DV):
                            if seq_end_idx + j_s < num_tokens:
                                dv[batch_idx, seq_end_idx + j_s, bh, j_v] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dq[batch_idx, seq_end_idx + j_s, bh, j_k] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dk[batch_idx, seq_end_idx + j_s, bh, j_k] = 0

                    for i_s in T.serial(num_iters):
                        left = seq_start_idx + (num_iters - i_s - 1) * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_wait(bar_01, (i_s + 0) % 2)
                        if i_s == 1:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + block_S + j_s < seq_end_idx:
                                    dk[batch_idx, left + block_S + j_s, bh, j_k] = (
                                        dqkv_shared[j_s, j_k]
                                    )
                        elif i_s > 1:
                            T.copy(
                                dqkv_shared,
                                dk[
                                    batch_idx,
                                    left + block_S : right + block_S,
                                    bh,
                                    0:DK,
                                ],
                            )
                        T.barrier_arrive(bar_04)
                        T.barrier_wait(bar_04, (i_s + 0) % 2)

                        T.barrier_wait(bar_05, (i_s + 0) % 2)
                        if i_s == 0:
                            for j_s, j_v in T.Parallel(block_S, DV):
                                if left + j_s < seq_end_idx:
                                    dv[batch_idx, left + j_s, bh, j_v] = dqkv_shared[
                                        j_s, j_v
                                    ]
                        else:
                            T.copy(dqkv_shared, dv[batch_idx, left:right, bh, 0:DV])
                        T.barrier_arrive(bar_10)
                        T.barrier_wait(bar_10, (i_s + 0) % 2)

                        T.barrier_wait(bar_11, (i_s + 0) % 2)
                        if i_s == 0:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    dq[batch_idx, left + j_s, bh, j_k] = dqkv_shared[
                                        j_s, j_k
                                    ]
                        else:
                            T.copy(dqkv_shared, dq[batch_idx, left:right, bh, 0:DK])

                elif tx < 384 + 96:  # TODO: set padding to 0
                    # h windows serve the CURRENT consumer iteration; a/do/b serve the next.
                    for i_s in T.serial(num_iters):
                        chunk_idx = num_iters - i_s - 1
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_02)
                        T.barrier_wait(bar_02, (i_s + 0) % 2)

                        # W1: S0[:, :DK/2]
                        if state_v_first:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DV,
                                    0:DK // 2,
                                ],
                                h_shared,
                                barrier=bar_h0,
                            )
                        else:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DK // 2,
                                    0:DV,
                                ],
                                h_shared,
                                barrier=bar_h0,
                            )
                        # arrive AFTER tma_copy: expect_tx must be registered
                        # before the arrives can complete the phase.
                        T.barrier_arrive(bar_h0)
                        T.barrier_wait(bar_h1, (i_s + 0) % 2)

                        # W2: S0[:, DK/2:]
                        if state_v_first:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DV,
                                    DK // 2 : DK,
                                ],
                                h_shared,
                                barrier=bar_h2,
                            )
                        else:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    DK // 2 : DK,
                                    0:DV,
                                ],
                                h_shared,
                                barrier=bar_h2,
                            )
                        T.barrier_arrive(bar_h2)
                        T.barrier_wait(bar_h3, (i_s + 0) % 2)

                        # W3: S0[:, :DK/2]
                        if state_v_first:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DV,
                                    0:DK // 2,
                                ],
                                h_shared,
                                barrier=bar_h4,
                            )
                        else:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DK // 2,
                                    0:DV,
                                ],
                                h_shared,
                                barrier=bar_h4,
                            )
                        T.barrier_arrive(bar_h4)
                        T.barrier_wait(bar_h5, (i_s + 0) % 2)

                        # W4: S0[:, DK/2:]
                        if state_v_first:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DV,
                                    DK // 2 : DK,
                                ],
                                h_shared,
                                barrier=bar_h6,
                            )
                        else:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    DK // 2 : DK,
                                    0:DV,
                                ],
                                h_shared,
                                barrier=bar_h6,
                            )
                        T.barrier_arrive(bar_h6)
                        # No bar_h7 wait here: W4 is consumed by both consumers
                        # during this iteration (steps 08/10), and h_shared is
                        # only overwritten again after bar_02 of the NEXT
                        # iteration completes, which happens strictly after both
                        # consumers finished their step-02 work.

                        if i_s < num_iters - 1:
                            chunk_idx_next = num_iters - i_s - 2
                            left_next = seq_start_idx + chunk_idx_next * block_S
                            right_next = left_next + block_S

                            T.barrier_wait(bar_14, (i_s + 0) % 2)
                            T.tma_copy(
                                a[batch_idx, left_next:right_next, bh, 0:block_S],
                                a_shared,
                                barrier=bar_02,
                            )

                            T.tma_copy(
                                do[batch_idx, left_next:right_next, bh, 0:DV],
                                do_shared,
                                barrier=bar_02,
                            )

                            T.barrier_wait(bar_15, (i_s + 0) % 2)
                            for j_s in T.Parallel(block_S):
                                b_shared[j_s] = b[batch_idx, left_next + j_s, bh]

                    if num_iters > 0:
                        T.barrier_wait(bar_00, (num_iters - 1) % 2)

                else:
                    if bb == batch_size - 1:
                        for j_s, j_v in T.Parallel(block_S, DV):
                            if seq_end_idx + j_s < num_tokens:
                                dv[batch_idx, seq_end_idx + j_s, bh, j_v] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dq[batch_idx, seq_end_idx + j_s, bh, j_k] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dk[batch_idx, seq_end_idx + j_s, bh, j_k] = 0

                    for i_s in T.serial(num_iters):
                        left = seq_start_idx + (num_iters - i_s - 1) * block_S

                        T.barrier_arrive(bar_05)
                        T.barrier_wait(bar_05, (i_s + 0) % 2)

                        T.barrier_wait(bar_15, (i_s + 0) % 2)

                        if i_s == 0:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    dg[batch_idx, left + j_s, bh] = dg_shared[j_s]
                            if (seq_end_idx - seq_start_idx) % block_S > 0:
                                dg[batch_idx, seq_end_idx - 1, bh] += dg_shared[
                                    block_S - 1
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                dg[batch_idx, left + j_s, bh] = dg_shared[j_s]

                        if i_s == 0:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    db[batch_idx, left + j_s, bh] = db_shared[j_s]
                        else:
                            for j_s in T.Parallel(block_S):
                                db[batch_idx, left + j_s, bh] = db_shared[j_s]

    return tilelang_fused_chunk_gdr_bwd_kernel


def fused_gdr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    h: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 32,
    state_v_first: bool = False,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 32

    if cu_seqlens is None:
        real_batch_size = batch_size
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        chunk_offsets = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        is_varlen = True

    use_dht = dht is not None
    if dht is None:
        dht = torch.empty(
            (real_batch_size, H, V, K)
            if state_v_first
            else (real_batch_size, H, K, V),
            dtype=torch.float32,
            device=k.device,
        )
    dq = torch.empty_like(v)
    dk = torch.empty_like(v)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    db = torch.empty_like(b)
    dh0 = torch.empty_like(dht)

    tilelang_fused_chunk_gdr_bwd_kernel = tilelang_fused_chunk_gdr_bwd(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h_dtype=h.dtype,
        o_dtype=do.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
        use_dht=use_dht,
        state_v_first=state_v_first,
    )
    tilelang_fused_chunk_gdr_bwd_kernel(
        do,
        dht,
        q,
        k,
        v,
        a,
        g,
        b,
        h,
        cu_seqlens,
        chunk_offsets,
        dq,
        dk,
        dv,
        dg,
        db,
        dh0,
    )

    return dq, dk, dv, dg, db, dh0
