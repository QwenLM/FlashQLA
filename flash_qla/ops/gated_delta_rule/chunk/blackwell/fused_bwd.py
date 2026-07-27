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
        # tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
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
    # TCGEN05 Layout-E maps relative Consumer-A threads; do not add the warp
    # group's absolute +256 thread offset.
    mask_tmem_layout = tilelang.layout.Layout(
        [block_S, block_S],
        lambda i, j: [i + (j // 32) * 64, j % 32],
    )

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
            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared(
                (DV, DK) if state_v_first else (DK, DV),
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
            tmp_shared_2_1 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_2 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_3 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_4_1 = T.alloc_shared(
                (DV, DK) if state_v_first else (DK, DV),
                dtype=qkva_dtype,
            )

            # CONSUMER_K
            dk_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_1 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_last_local_1 = T.alloc_fragment((1), dtype=accum_dtype)

            # CONSUMER_A
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dp_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            da_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            hi_fragment = T.alloc_fragment((block_S, block_S), dtype="uint16")
            lo_fragment = T.alloc_fragment((block_S, block_S), dtype="uint16")
            uint32_fragment = T.alloc_fragment((block_S, block_S), dtype="uint32")
            u_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dq_fragment = u_fragment
            db_fragment = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_fragment_2 = T.alloc_fragment((block_S), dtype=accum_dtype)

            # CONSUMER_S
            dh_fragment_L = T.alloc_fragment(
                (DV, DK // 2) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )
            dh_fragment_R = T.alloc_fragment(
                (DV, DK // 2) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )
            _odot_fragment_3 = T.alloc_fragment(
                (DV, DK) if state_v_first else (DK, DV),
                dtype=accum_dtype,
            )
            reduce_fragment = T.alloc_fragment((128, 2), dtype=accum_dtype)
            dg_last_local_3 = T.alloc_fragment((1), dtype=accum_dtype)

            a_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            da_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            dp_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            mask_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            u_tmem = T.alloc_tmem((block_S, DK), dtype=accum_dtype)
            dq_tmem = u_tmem
            dk_tmem = T.alloc_tmem((block_S, DK), dtype=accum_dtype)
            dv_tmem = T.alloc_tmem((block_S, DK), dtype=accum_dtype)
            dh_tmem_L = T.alloc_tmem(
                (DV, DK // 2) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )
            dh_tmem_R = T.alloc_tmem(
                (DV, DK // 2) if state_v_first else (DK, DV // 2),
                dtype=accum_dtype,
            )

            tcbar_00 = T.alloc_barrier(arrive_count=1)
            tcbar_01 = T.alloc_barrier(arrive_count=1)
            tcbar_02 = T.alloc_barrier(arrive_count=1)
            tcbar_03 = T.alloc_barrier(arrive_count=1)
            tcbar_04 = T.alloc_barrier(arrive_count=1)
            tcbar_05a = T.alloc_barrier(arrive_count=1)
            tcbar_05b = T.alloc_barrier(arrive_count=1)
            tcbar_06 = T.alloc_barrier(arrive_count=1)
            tcbar_07 = T.alloc_barrier(arrive_count=1)
            tcbar_08 = T.alloc_barrier(arrive_count=1)
            tcbar_09 = T.alloc_barrier(arrive_count=1)
            tcbar_10 = T.alloc_barrier(arrive_count=1)
            tcbar_11a = T.alloc_barrier(arrive_count=1)
            tcbar_11b = T.alloc_barrier(arrive_count=1)
            tcbar_12 = T.alloc_barrier(arrive_count=1)
            tcbar_13a = T.alloc_barrier(arrive_count=1)
            tcbar_13b = T.alloc_barrier(arrive_count=1)
            tcbar_13c = T.alloc_barrier(arrive_count=1)
            tcbar_14a = T.alloc_barrier(arrive_count=1)
            tcbar_14b = T.alloc_barrier(arrive_count=1)
            tcbar_15 = T.alloc_barrier(arrive_count=1)

            # 16 stages
            bar_00 = T.alloc_barrier(arrive_count=480)
            bar_01 = T.alloc_barrier(arrive_count=288)
            bar_02 = T.alloc_barrier(arrive_count=288)
            bar_03 = T.alloc_barrier(arrive_count=256)
            bar_04 = T.alloc_barrier(arrive_count=384)
            bar_05 = T.alloc_barrier(arrive_count=288)
            bar_06 = T.alloc_barrier(arrive_count=256)
            bar_07 = T.alloc_barrier(arrive_count=256)
            bar_08 = T.alloc_barrier(arrive_count=384)
            bar_09 = T.alloc_barrier(arrive_count=256)
            bar_10 = T.alloc_barrier(arrive_count=288)
            bar_11 = T.alloc_barrier(arrive_count=256)
            bar_12 = T.alloc_barrier(arrive_count=128)
            bar_13 = T.alloc_barrier(arrive_count=256)
            bar_13a = T.alloc_barrier(arrive_count=128)
            bar_14 = T.alloc_barrier(arrive_count=256)
            bar_15 = T.alloc_barrier(arrive_count=256)

            T.annotate_layout(
                {
                    do_shared: tilelang.layout.make_swizzled_layout(do_shared),
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    h_shared: tilelang.layout.make_swizzled_layout(h_shared),
                    mask_tmem: mask_tmem_layout,
                    dqkv_shared: tilelang.layout.make_swizzled_layout(dqkv_shared),
                    tmp_shared_1_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_1
                    ),
                    tmp_shared_1_2: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_2
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

            PRODUCER_NREG = 64
            CONSUMER_K_NREG = 136
            CONSUMER_A_NREG = 136
            CONSUMER_S_NREG = 176

            # Prefetch the last chunk of data
            if state_v_first:
                T.copy(
                    h[batch_idx, chunk_start_idx + num_iters - 1, bh, 0:DV, 0:DK],
                    h_shared,
                )
            else:
                T.copy(
                    h[batch_idx, chunk_start_idx + num_iters - 1, bh, 0:DK, 0:DV],
                    h_shared,
                )
            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    q_shared[j_s, j_k] = q[batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bhg, j_k]
                else:
                    q_shared[j_s, j_k] = 0
            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    k_shared[j_s, j_k] = k[batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bhg, j_k]
                else:
                    k_shared[j_s, j_k] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    v_shared[j_s, j_v] = v[batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh, j_v]
                else:
                    v_shared[j_s, j_v] = 0
            for j_s, j_t in T.Parallel(block_S, block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    a_shared[j_s, j_t] = a[batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh, j_t]
                else:
                    a_shared[j_s, j_t] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    do_shared[j_s, j_v] = do[batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh, j_v]
                else:
                    do_shared[j_s, j_v] = 0
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    g_shared[j_s] = g[batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh]
                else:
                    g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    b_shared[j_s] = b[batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh]
                else:
                    b_shared[j_s] = 0

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                if use_dht:
                    if state_v_first:
                        T.copy(dht[bb, bh, 0:DV, :DK // 2], dh_fragment_L)
                        T.copy(dht[bb, bh, 0:DV, DK // 2:], dh_fragment_R)
                    else:
                        T.copy(dht[bb, bh, 0:DK, :DV // 2], dh_fragment_L)
                        T.copy(dht[bb, bh, 0:DK, DV // 2:], dh_fragment_R)
                else:
                    T.clear(dh_fragment_L)
                    T.clear(dh_fragment_R)
                if state_v_first:
                    for j_v, j_k in T.Parallel(DV, DK // 2):
                        tmp_shared_4_1[j_v, j_k] = dh_fragment_L[j_v, j_k]
                    for j_v, j_k in T.Parallel(DV, DK // 2):
                        tmp_shared_4_1[j_v, j_k + DK // 2] = dh_fragment_R[j_v, j_k]
                else:
                    for j_k, j_v in T.Parallel(DK, DV // 2):
                        tmp_shared_4_1[j_k, j_v] = dh_fragment_L[j_k, j_v]
                    for j_k, j_v in T.Parallel(DK, DV // 2):
                        tmp_shared_4_1[j_k, j_v + DV // 2] = dh_fragment_R[j_k, j_v]

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                        )
                    T.barrier_arrive(bar_01)

                    # 01, 02, 03
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    # dS0 = g_last * dSt. Referencing the stable shared scalar directly
                    # enables TileLang packed fmul2 lowering for contiguous fragment lanes.
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            dh_fragment_L[j_v, j_k] *= g_exp_shared[block_S - 1]
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            dh_fragment_R[j_v, j_k] *= g_exp_shared[block_S - 1]
                    else:
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            dh_fragment_L[j_k, j_v] *= g_exp_shared[block_S - 1]
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            dh_fragment_R[j_k, j_v] *= g_exp_shared[block_S - 1]
                    T.copy(dh_fragment_L, dh_tmem_L)
                    T.copy(dh_fragment_R, dh_tmem_R)
                    T.barrier_arrive(bar_04)

                    # 04, 05, 06, 07
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # dg_last += sum(dS0 * S0)
                    if state_v_first:
                        T.clear(reduce_fragment)
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            reduce_fragment[
                                j_v, j_k % 2,
                            ] += dh_fragment_L[j_v, j_k] * h_shared[j_v, j_k]
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            reduce_fragment[
                                j_v, j_k % 2,
                            ] += dh_fragment_R[j_v, j_k] * h_shared[j_v, j_k + DK // 2]
                    else:
                        T.clear(reduce_fragment)
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            reduce_fragment[
                                j_k, j_v % 2,
                            ] += dh_fragment_L[j_k, j_v] * h_shared[j_k, j_v]
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            reduce_fragment[
                                j_k, j_v % 2,
                            ] += dh_fragment_R[j_k, j_v] * h_shared[j_k, j_v + DV // 2]
                    T.barrier_arrive(bar_08)
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    T.barrier_wait(bar_09, (i_s + 0) % 2)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    T.reduce_sum(
                        T.reshape(reduce_fragment, (128 * 2,)),
                        dg_last_local_3,
                        dim=0,
                        clear=True,
                    )
                    dg_shared[block_S - 1] += dg_last_local_3[0]
                    T.barrier_arrive(bar_11)

                    # 11
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_11a, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_11b, (i_s + 0) % 2)
                    T.barrier_arrive(bar_12)
                    T.barrier_wait(bar_12, (i_s + 0) % 2)

                    # 13
                    T.barrier_wait(bar_13, (i_s + 0) % 2)
                    # dOg = s * g * dO
                    for j_s, j_v in T.Parallel(block_S, DV):
                        tmp_shared_2_3[j_s, j_v] = (
                            scale * do_shared[j_s, j_v] * g_exp_shared[j_s]
                        )
                    T.barrier_arrive(bar_14)

                    # 14
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_14a, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_14b, (i_s + 0) % 2)
                    T.barrier_arrive(bar_15)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    # S4[1] = dS0
                    T.copy(dh_tmem_L, dh_fragment_L)
                    T.copy(dh_tmem_R, dh_fragment_R)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            tmp_shared_4_1[j_v, j_k] = dh_fragment_L[j_v, j_k]
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            tmp_shared_4_1[j_v, j_k + DK // 2] = dh_fragment_R[j_v, j_k]
                    else:
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            tmp_shared_4_1[j_k, j_v] = dh_fragment_L[j_k, j_v]
                        for j_k, j_v in T.Parallel(DK, DV // 2):
                            tmp_shared_4_1[j_k, j_v + DV // 2] = dh_fragment_R[j_k, j_v]

                if use_dht:
                    if state_v_first:
                        T.copy(dh_fragment_L, dh0[bb, bh, 0:DV, :DK // 2])
                        T.copy(dh_fragment_R, dh0[bb, bh, 0:DV, DK // 2:])
                    else:
                        T.copy(dh_fragment_L, dh0[bb, bh, 0:DK, :DV // 2])
                        T.copy(dh_fragment_R, dh0[bb, bh, 0:DK, DV // 2:])

            elif tx < 256:
                T.set_max_nreg(CONSUMER_K_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 16 == 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # S2[S] dK
                    if i_s > 0:
                        T.copy(dk_fragment, dqkv_shared)
                    T.barrier_arrive(bar_01)

                    # 01
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_01, (i_s + 0) % 2)
                    # dV' = g_last/g * dV'
                    T.copy(dv_tmem, dv_fragment)
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.copy(dv_fragment, dv_tmem)
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_02, (i_s + 0) % 2)
                    T.copy(dv_tmem, dv_fragment)
                    T.barrier_arrive(bar_03)

                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # S2[1] dV'
                    T.copy(dv_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    T.copy(u_tmem, dk_fragment)
                    T.barrier_wait(tcbar_04, (i_s + 0) % 2)
                    T.barrier_arrive(bar_05)

                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # S2[S] dV
                    T.copy(dv_tmem, dv_fragment)
                    T.copy(dv_fragment, dqkv_shared)
                    # dVg = -g * dV
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    # dg += sum(dVg * U)
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dk_fragment[j_s, j_v] *= dv_fragment[j_s, j_v]
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    T.copy(dv_fragment, u_tmem)
                    T.reduce_sum(dk_fragment, dg_fragment_1, dim=1, clear=True)
                    T.copy(dg_fragment_1, dg_shared)
                    # S2[2] K
                    T.copy(k_shared, dv_fragment)
                    T.copy(dv_fragment, tmp_shared_2_2)
                    T.copy(dv_fragment, dv_tmem)
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # S2[3] dVg
                    T.copy(u_tmem, dv_fragment)
                    T.copy(dv_fragment, tmp_shared_2_3)
                    T.barrier_wait(tcbar_07, (i_s + 0) % 2)
                    T.barrier_arrive(bar_08)

                    # 08
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    T.copy(dv_tmem, dv_fragment)
                    # dK = g_last/g * dK
                    T.copy(dk_tmem, dk_fragment)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dk_fragment[j_s, j_k] *= g_rev_exp_shared[j_s]
                    T.copy(dk_fragment, dk_tmem)
                    # dg -= sum(K * dK)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dv_fragment[j_s, j_k] *= -dk_fragment[j_s, j_k]
                    T.reduce_sum(dv_fragment, dg_fragment_1, dim=1, clear=True)
                    # dg_last += sum(K * dK)
                    T.reduce_sum(dg_fragment_1, dg_last_local_1, dim=0, clear=True)
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # Sg[S] dg
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_1[j_s]
                    dg_shared[block_S - 1] -= dg_last_local_1[0]
                    T.barrier_wait(tcbar_09, (i_s + 0) % 2)
                    T.barrier_arrive(bar_10)
                    T.barrier_wait(bar_10, (i_s + 0) % 2)

                    # 12
                    T.barrier_wait(bar_12, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_12, (i_s + 0) % 2)
                    T.barrier_arrive(bar_13)
                    T.barrier_wait(bar_13, (i_s + 0) % 2)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_15, (i_s + 0) % 2)
                    T.copy(dk_tmem, dk_fragment)

                for j_s, j_k in T.Parallel(block_S, DK):
                    if seq_start_idx + j_s < seq_end_idx:
                        dk[batch_idx, seq_start_idx + j_s, bh, j_k] = dk_fragment[
                            j_s, j_k
                        ]

            elif tx < 384:
                T.set_max_nreg(CONSUMER_A_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00, 01
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            a_fragment[j_s, j_t] = T.exp2(
                                a_fragment[j_s, j_t] * 1.442695
                            )
                        else:
                            a_fragment[j_s, j_t] = 0
                    T.copy(a_fragment, mask_tmem)
                    # Pg = s * P * G
                    T.barrier_wait(tcbar_00, (i_s + 0) % 2)
                    T.copy(p_tmem, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            p_fragment[j_s, j_t * 2 + j_t_vec] *= a_fragment[
                                j_s, j_t * 2 + j_t_vec
                            ]
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            p_fragment[j_s, j_t * 2 + j_t_vec] *= scale
                    # S1[1] Pg
                    T.copy(p_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    # Ab = Ar * b
                    T.copy(a_shared, a_fragment)
                    T.copy(mask_tmem, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    # Ag = G * Ab
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            a_fragment[j_s, j_t * 2 + j_t_vec] *= p_fragment[
                                j_s, j_t * 2 + j_t_vec
                            ]
                    T.barrier_arrive(bar_03)

                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # S1[2] Ag
                    T.copy(a_fragment, tmp_shared_1_2)
                    T.barrier_wait(tcbar_03, (i_s + 0) % 2)
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # S2[3] U
                    T.copy(u_tmem, u_fragment)
                    # W = V - g * U
                    for j_s, j_v in T.Parallel(block_S, DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, DV):
                        u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    # S2[2] W
                    T.copy(u_fragment, tmp_shared_2_2)
                    T.barrier_arrive(bar_05)

                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_05a, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_05b, (i_s + 0) % 2)
                    # S2[1] V'
                    T.copy(u_tmem, u_fragment)
                    T.copy(u_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    # dAb = G * dAg
                    T.copy(a_tmem, da_fragment)
                    T.copy(mask_tmem, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            da_fragment[j_s, j_t * 2 + j_t_vec] *= p_fragment[
                                j_s, j_t * 2 + j_t_vec
                            ]
                    T.copy(da_fragment, da_tmem)
                    T.barrier_wait(tcbar_06, (i_s + 0) % 2)
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # dP = G * dPg
                    T.copy(dp_tmem, dp_fragment)
                    T.copy(mask_tmem, a_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            dp_fragment[j_s, j_t * 2 + j_t_vec] *= a_fragment[
                                j_s, j_t * 2 + j_t_vec
                            ]
                    # dg += sum((dPg * P) - (dPg * P)^T)
                    T.copy(p_tmem, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            p_fragment[j_s, j_t * 2 + j_t_vec] *= (
                                dp_fragment[j_s, j_t * 2 + j_t_vec] * scale
                            )
                    T.copy(p_fragment, mask_tmem)
                    # dPg = s * dPg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= scale
                    # S1[1] dP
                    T.copy(dp_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_08)

                    # 08
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    # S2[1] Q
                    T.copy(q_shared[:, : DK // 2], a_fragment)
                    # Snapshot Q before the producer overwrites q_shared after bar_09.
                    T.copy(a_fragment, tmp_shared_2_1[:, : DK // 2])
                    T.copy(q_shared[:, DK // 2 :], p_fragment)
                    T.copy(p_fragment, tmp_shared_2_1[:, DK // 2 :])
                    T.barrier_wait(tcbar_08, (i_s + 0) % 2)
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # dQ = s * g * dQ. Use the whole supported TMEM layout,
                    # then reuse the dead full-width U fragment for arithmetic.
                    T.copy(dq_tmem, dq_fragment)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dq_fragment[j_s, j_k] *= g_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK // 2):
                        for j_k_vec in T.vectorized(2):
                            dq_fragment[j_s, j_k * 2 + j_k_vec] *= scale
                    T.copy(dq_fragment, dq_tmem)
                    # dg += sum(Q * dQ)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dq_fragment[j_s, j_k] *= tmp_shared_2_1[j_s, j_k]
                    T.reduce_sum(dq_fragment, dg_fragment_2, dim=1, clear=True)
                    T.barrier_arrive(bar_10)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_10, (i_s + 0) % 2)
                    # S2[S] dQ
                    T.copy(dq_tmem, dq_fragment)
                    T.copy(dq_fragment, dqkv_shared)
                    T.barrier_arrive(bar_11)

                    # 11, 12
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    # dAb * Ar
                    T.copy(a_shared, a_fragment)
                    T.copy(da_tmem, da_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            a_fragment[j_s, j_t * 2 + j_t_vec] *= da_fragment[
                                j_s, j_t * 2 + j_t_vec
                            ]
                    T.copy(a_fragment, a_tmem)
                    # dAb * Ab [ = G * dAg * Ab ]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    T.copy(mask_tmem, p_fragment)
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
                    T.reduce_sum(a_fragment, dg_fragment_2, dim=1, clear=False)
                    # Sg[S] dg
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_2[j_s]
                    # db = sum((dAb * Ar)^T)
                    T.copy(a_tmem, a_fragment)
                    T.copy(a_fragment, tmp_shared_1_2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = tmp_shared_1_2[j_t, j_s]
                    T.reduce_sum(a_fragment, db_fragment, dim=1, clear=True)
                    # dAr = dAb * b
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= b_shared[j_t]
                    # S1[2] dAr
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_13)

                    # 13
                    T.barrier_wait(bar_13, (i_s + 0) % 2)
                    T.barrier_wait(tcbar_13a, (i_s + 0) % 2)
                    T.copy(da_tmem, da_fragment)
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_13a)
                    T.barrier_wait(tcbar_13b, (i_s + 0) % 2)
                    T.copy(da_tmem, da_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s <= j_t:
                            da_fragment[j_s, j_t] = 0
                        else:
                            da_fragment[j_s, j_t] = -da_fragment[j_s, j_t]
                    T.barrier_wait(tcbar_13c, (i_s + 0) % 2)
                    T.barrier_arrive(bar_14)

                    # 14
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    # db += sum(dA * At)
                    T.copy(a_tmem, a_fragment)
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

                if tx < 384 + 32:

                    T.set_max_nreg(PRODUCER_NREG, 0)
                    for i_s in T.serial(num_iters):
                        T.barrier_arrive(bar_00)

                        T.barrier_wait(bar_00, (i_s + 0) % 2)
                        # P = Q @ K^T
                        T.tcgen05_gemm(
                            q_shared,
                            k_shared,
                            p_tmem,
                            transpose_B=True,
                            clear_accum=True,
                            mbar=tcbar_00,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_01, (i_s + 0) % 2)
                        # dV' = K @ dSt
                        T.tcgen05_gemm(
                            k_shared,
                            tmp_shared_4_1,
                            dv_tmem,
                            transpose_B=state_v_first,
                            clear_accum=True,
                            mbar=tcbar_01,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_02, (i_s + 0) % 2)
                        # dV' += Pg^T @ dO
                        T.tcgen05_gemm(
                            tmp_shared_1_1,
                            do_shared,
                            dv_tmem,
                            transpose_A=True,
                            clear_accum=False,
                            mbar=tcbar_02,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_03, (i_s + 0) % 2)
                        # U = K @ S0
                        T.tcgen05_gemm(
                            k_shared,
                            h_shared,
                            u_tmem,
                            transpose_B=state_v_first,
                            clear_accum=True,
                            mbar=tcbar_03,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_04, (i_s + 0) % 2)
                        # dV = Ag^T @ dV'
                        T.tcgen05_gemm(
                            tmp_shared_1_2,
                            tmp_shared_2_1,
                            dv_tmem,
                            transpose_A=True,
                            clear_accum=True,
                            mbar=tcbar_04,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_05, (i_s + 0) % 2)
                        # V' = Ag @ W
                        T.tcgen05_gemm(
                            tmp_shared_1_2,
                            tmp_shared_2_2,
                            u_tmem,
                            clear_accum=True,
                            mbar=tcbar_05a,
                            use_2cta=False,
                        )
                        # dAg = dV' @ W^T
                        T.tcgen05_gemm(
                            tmp_shared_2_1,
                            tmp_shared_2_2,
                            a_tmem,
                            transpose_B=True,
                            clear_accum=True,
                            mbar=tcbar_05b,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_06, (i_s + 0) % 2)
                        # dPg = dO @ V'^T
                        T.tcgen05_gemm(
                            do_shared,
                            tmp_shared_2_1,
                            dp_tmem,
                            transpose_B=True,
                            clear_accum=True,
                            mbar=tcbar_06,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_07, (i_s + 0) % 2)
                        # dK = V' @ dSt^T
                        T.tcgen05_gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_tmem,
                            transpose_B=not state_v_first,
                            clear_accum=True,
                            mbar=tcbar_07,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_08, (i_s + 0) % 2)
                        # dQ = dO @ S0^T
                        T.tcgen05_gemm(
                            do_shared,
                            h_shared,
                            dq_tmem,
                            transpose_B=not state_v_first,
                            clear_accum=True,
                            mbar=tcbar_08,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_09, (i_s + 0) % 2)
                        # dK += dVg @ S0^T
                        T.tcgen05_gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_tmem,
                            transpose_B=not state_v_first,
                            clear_accum=False,
                            mbar=tcbar_09,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_10, (i_s + 0) % 2)
                        # dQ += dP @ K
                        T.tcgen05_gemm(
                            tmp_shared_1_1,
                            tmp_shared_2_2,
                            dq_tmem,
                            clear_accum=False,
                            mbar=tcbar_10,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_11, (i_s + 0) % 2)
                        # dS0 += K^T @ dVg
                        if state_v_first:
                            T.tcgen05_gemm(
                                tmp_shared_2_3,
                                tmp_shared_2_2[:, :DK // 2],
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_11a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                tmp_shared_2_3,
                                tmp_shared_2_2[:, DK // 2:],
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_11b,
                                use_2cta=False,
                            )
                        else:
                            T.tcgen05_gemm(
                                tmp_shared_2_2,
                                tmp_shared_2_3[:, :DV // 2],
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_11a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                tmp_shared_2_2,
                                tmp_shared_2_3[:, DV // 2:],
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_11b,
                                use_2cta=False,
                            )

                        T.barrier_wait(bar_12, (i_s + 0) % 2)
                        # dK += dP^T @ Q
                        T.tcgen05_gemm(
                            tmp_shared_1_1,
                            tmp_shared_2_1,
                            dk_tmem,
                            transpose_A=True,
                            clear_accum=False,
                            mbar=tcbar_12,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_13, (i_s + 0) % 2)
                        # dA = -Ar^T @ dAr @ Ar^T
                        T.tcgen05_gemm(
                            a_shared,
                            tmp_shared_1_2,
                            da_tmem,
                            transpose_A=True,
                            clear_accum=True,
                            mbar=tcbar_13a,
                            use_2cta=False,
                        )
                        # At = K @ K^T
                        T.tcgen05_gemm(
                            tmp_shared_2_2,
                            tmp_shared_2_2,
                            a_tmem,
                            transpose_B=True,
                            clear_accum=True,
                            mbar=tcbar_13c,
                            use_2cta=False,
                        )
                        T.barrier_wait(bar_13a, (i_s + 0) % 2)
                        T.tcgen05_gemm(
                            tmp_shared_1_2,
                            a_shared,
                            da_tmem,
                            transpose_B=True,
                            clear_accum=True,
                            mbar=tcbar_13b,
                            use_2cta=False,
                        )

                        T.barrier_wait(bar_14, (i_s + 0) % 2)
                        # dS0 += Q^T @ dOg
                        if state_v_first:
                            T.tcgen05_gemm(
                                tmp_shared_2_3,
                                tmp_shared_2_1[:, :DK // 2],
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_14a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                tmp_shared_2_3,
                                tmp_shared_2_1[:, DK // 2:],
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_14b,
                                use_2cta=False,
                            )
                        else:
                            T.tcgen05_gemm(
                                tmp_shared_2_1,
                                tmp_shared_2_3[:, :DV // 2],
                                dh_tmem_L,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_14a,
                                use_2cta=False,
                            )
                            T.tcgen05_gemm(
                                tmp_shared_2_1,
                                tmp_shared_2_3[:, DV // 2:],
                                dh_tmem_R,
                                transpose_A=True,
                                clear_accum=False,
                                mbar=tcbar_14b,
                                use_2cta=False,
                            )

                        T.barrier_wait(bar_15, (i_s + 0) % 2)
                        # dK += dAs @ K
                        T.tcgen05_gemm(
                            tmp_shared_1_2,
                            tmp_shared_2_2,
                            dk_tmem,
                            clear_accum=False,
                            mbar=tcbar_15,
                            use_2cta=False,
                        )

                elif tx < 384 + 64:

                    T.set_max_nreg(PRODUCER_NREG, 0)
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_wait(bar_03, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]

                        T.barrier_wait(bar_05, (i_s + 0) % 2)
                        T.tma_copy(
                            v[batch_idx, left:right, bh, 0:DV],
                            v_shared,
                            barrier=bar_00,
                        )

                        T.barrier_wait(bar_07, (i_s + 0) % 2)
                        T.tma_copy(
                            k[batch_idx, left:right, bhg, 0:DK],
                            k_shared,
                            barrier=bar_00,
                        )

                        T.barrier_wait(bar_09, (i_s + 0) % 2)
                        T.tma_copy(
                            q[batch_idx, left:right, bhg, 0:DK],
                            q_shared,
                            barrier=bar_00,
                        )

                    if num_iters > 0:
                        T.barrier_arrive(bar_00)

                elif tx < 384 + 96:

                    T.set_max_nreg(PRODUCER_NREG, 0)
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_02)
                        T.barrier_wait(bar_02, (i_s + 0) % 2)

                        T.barrier_wait(bar_10, (i_s + 0) % 2)
                        if state_v_first:
                            T.tma_copy(
                                h[batch_idx, chunk_start_idx + chunk_idx, bh, 0:DV, 0:DK],
                                h_shared,
                                barrier=bar_02,
                            )
                        else:
                            T.tma_copy(
                                h[batch_idx, chunk_start_idx + chunk_idx, bh, 0:DK, 0:DV],
                                h_shared,
                                barrier=bar_02,
                            )

                        T.barrier_wait(bar_14, (i_s + 0) % 2)
                        T.tma_copy(
                            a[batch_idx, left:right, bh, 0:block_S],
                            a_shared,
                            barrier=bar_02,
                        )

                        T.tma_copy(
                            do[batch_idx, left:right, bh, 0:DV],
                            do_shared,
                            barrier=bar_02,
                        )

                        T.barrier_wait(bar_15, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            b_shared[j_s] = b[batch_idx, left + j_s, bh]

                    if num_iters > 0:
                        T.barrier_wait(bar_00, (num_iters - 1) % 2)
                        T.barrier_arrive(bar_02)

                else:

                    T.set_max_nreg(PRODUCER_NREG, 0)

                    if num_iters > 0:
                        left = seq_start_idx + (num_iters - 1) * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, 0)

                        T.barrier_arrive(bar_01)
                        T.barrier_wait(bar_01, 0)
                        T.barrier_arrive(bar_05)
                        T.barrier_wait(bar_05, 0)

                        T.barrier_wait(bar_06, 0)
                        for j_s, j_v in T.Parallel(block_S, DV):
                            if left + j_s < seq_end_idx:
                                dv[batch_idx, left + j_s, bh, j_v] = dqkv_shared[j_s, j_v]
                        T.barrier_arrive(bar_10)
                        T.barrier_wait(bar_10, 0)

                        T.barrier_wait(bar_11, 0)
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if left + j_s < seq_end_idx:
                                dq[batch_idx, left + j_s, bh, j_k] = dqkv_shared[j_s, j_k]

                        T.barrier_wait(bar_15, 0)
                        for j_s in T.Parallel(block_S):
                            if left + j_s < seq_end_idx:
                                dg[batch_idx, left + j_s, bh] = dg_shared[j_s]
                        if (seq_end_idx - seq_start_idx) % block_S > 0:
                            dg[batch_idx, seq_end_idx - 1, bh] += dg_shared[block_S - 1]

                        for j_s in T.Parallel(block_S):
                            if left + j_s < seq_end_idx:
                                db[batch_idx, left + j_s, bh] = db_shared[j_s]

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
                            for j_s in T.Parallel(block_S):
                                if seq_end_idx + j_s < num_tokens:
                                    dg[batch_idx, seq_end_idx + j_s, bh] = 0
                            for j_s in T.Parallel(block_S):
                                if seq_end_idx + j_s < num_tokens:
                                    db[batch_idx, seq_end_idx + j_s, bh] = 0

                    for i_s in T.serial(1, num_iters):  # TODO: check acc error w/ TMA
                        left = seq_start_idx + (num_iters - i_s - 1) * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_arrive(bar_01)
                        T.barrier_wait(bar_01, (i_s + 0) % 2)
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if left + block_S + j_s < seq_end_idx:
                                dk[batch_idx, left + block_S + j_s, bh, j_k] = dqkv_shared[j_s, j_k]
                        T.barrier_arrive(bar_05)
                        T.barrier_wait(bar_05, (i_s + 0) % 2)

                        T.barrier_wait(bar_06, (i_s + 0) % 2)
                        for j_s, j_v in T.Parallel(block_S, DV):
                            dv[batch_idx, left + j_s, bh, j_v] = dqkv_shared[j_s, j_v]
                        T.barrier_arrive(bar_10)
                        T.barrier_wait(bar_10, (i_s + 0) % 2)

                        T.barrier_wait(bar_11, (i_s + 0) % 2)
                        for j_s, j_k in T.Parallel(block_S, DK):
                            dq[batch_idx, left + j_s, bh, j_k] = dqkv_shared[j_s, j_k]

                        T.barrier_wait(bar_15, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            dg[batch_idx, left + j_s, bh] = dg_shared[j_s]

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
    chunk_size: int = 64,
    state_v_first: bool = False,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64

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
