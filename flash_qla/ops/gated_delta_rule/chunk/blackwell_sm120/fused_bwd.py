# Copyright (c) 2026 The Qwen team, Alibaba Group.
# dbg-bump-1
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_offsets


@tilelang.jit
def tilelang_zero_bwd_padding(
    H,
    DK,
    DV,
    qkva_dtype,
    g_dtype,
    b_dtype,
    seqlen_dtype,
):
    num_tokens = T.dynamic("num_tokens")
    real_batch_size = T.dynamic("real_batch_size")

    @T.prim_func
    def tilelang_zero_bwd_padding_kernel(
        cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
        dq: T.Tensor([1, num_tokens, H, DK], dtype=qkva_dtype),
        dk: T.Tensor([1, num_tokens, H, DK], dtype=qkva_dtype),
        dv: T.Tensor([1, num_tokens, H, DV], dtype=qkva_dtype),
        dg: T.Tensor([1, num_tokens, H], dtype=g_dtype),
        db: T.Tensor([1, num_tokens, H], dtype=b_dtype),
    ):
        # One CTA per value head keeps the no-padding path cheap: it performs
        # only the dynamic end lookup and exits.  Padding is uncommon and is
        # generally short, so serializing tokens while vectorizing the 128-D
        # row avoids zeroing every valid output before the main kernel.
        with T.Kernel(H, threads=128) as (bh,):
            seq_end_idx = T.alloc_var("int32")
            seq_end_idx = cu_seqlens[real_batch_size]
            for i in T.serial(num_tokens - seq_end_idx):
                for j in T.Parallel(DK):
                    dq[0, seq_end_idx + i, bh, j] = 0
                    dk[0, seq_end_idx + i, bh, j] = 0
                for j in T.Parallel(DV):
                    dv[0, seq_end_idx + i, bh, j] = 0
                if T.get_thread_binding() == 0:
                    dg[0, seq_end_idx + i, bh] = 0
                    db[0, seq_end_idx + i, bh] = 0

    return tilelang_zero_bwd_padding_kernel


@tilelang.jit(
    # out_idx=[-5, -4, -3, -2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
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
            # v_shared was removed: A reads v from global memory, saving 8192 bytes of shared memory.
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            # half the dk dim
            h_shared = T.alloc_shared(
                (DV, DK//2) if state_v_first else (DK//2, DV),
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
            # Half-sized 4_1: holds either the left or right DK half of dS0.
            tmp_shared_4_1 = T.alloc_shared(
                (DV, DK//2) if state_v_first else (DK//2, DV),
                dtype=qkva_dtype,
            )

            # CONSUMER_K
            # Split dK along DK to match the half-DK tmp_shared_4_1 and h_shared.
            dk_frag_l = T.alloc_fragment((block_S, DK//2), dtype=accum_dtype)
            dk_frag_r = T.alloc_fragment((block_S, DK//2), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            odot_fragment_1 = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_1 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_last_local_1 = T.alloc_fragment((1), dtype=accum_dtype)
            dh_left_fragment = T.alloc_fragment(
                (DV, DK // 2) if state_v_first else (DK // 2, DV),
                dtype=qkva_dtype,
            )

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
            # Split dQ along DK to match the resident right/left half of h_shared.
            dq_frag_r = T.alloc_fragment((block_S, DK//2), dtype=accum_dtype)
            dq_frag_l = T.alloc_fragment((block_S, DK//2), dtype=accum_dtype)
            db_fragment = T.alloc_fragment((block_S), dtype=accum_dtype)
            odot_fragment_2 = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_2 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_fragment_2b = T.alloc_fragment((block_S), dtype=accum_dtype)


            # CONSUMER_S
            # Physically split dh into two half-DK fragments to avoid offset views.
            # Their MMA C-layout projection cannot be normalized on SM120.
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
            # todo make reduce_fragment(128,2)
            reduce_fragment = T.alloc_fragment((128,2),dtype=accum_dtype)
            dg_last_local_3 = T.alloc_fragment((1), dtype=accum_dtype)
            g_last_local_3 = T.alloc_local((1), dtype=accum_dtype)

            # 16 stages
            bar_00 = T.alloc_barrier(arrive_count=448)
            bar_01 = T.alloc_barrier(arrive_count=384)
            bar_02 = T.alloc_barrier(arrive_count=288)
            bar_03 = T.alloc_barrier(arrive_count=256)
            bar_04 = T.alloc_barrier(arrive_count=416)
            bar_05 = T.alloc_barrier(arrive_count=288)
            bar_06 = T.alloc_barrier(arrive_count=256)
            bar_07 = T.alloc_barrier(arrive_count=256)
            # when barrier 8_1 arrivee,all consumer complete the left part
            bar_08_1 = T.alloc_barrier(arrive_count=256)
            bar_08_2 = T.alloc_barrier(arrive_count=384) #all right windows has been cosnumed
            bar_08_3 = T.alloc_barrier(arrive_count=128)
            # Synchronize K's self-consumed half-DK transfers through tmp_shared_4_1 (dS0).
            # bar_k_right_ready = T.alloc_barrier(arrive_count=128)  # K-01 loads the right half before computing the right half of dV'.
            bar_k_left_ready = T.alloc_barrier(arrive_count=128)   # K-07 loads the left half before producing the left half of dK.
            # K-07 write-after-read synchronization for dk_frag -> dqkv_shared in the dg dot product.
            # bar_s4_merge = T.alloc_barrier(arrive_count=128)
            # S publishes the left half of dSt through tmp_shared_4_1; K waits,
            # retains it in registers, and then allows S to overwrite the same
            # tile with the right half.
            bar_dhtmp_ready = T.alloc_barrier(arrive_count=128)
            # After K retains the published left half in registers, S reuses
            # the same shared tile to publish the right half.  This replaces
            # the remaining intra-CTA HBM handoff.
            bar_dh_right_ready = T.alloc_barrier(arrive_count=128)
            # A-10 write-after-read synchronization for the dg dot product staged through dqkv_shared.
            bar_s4_dot_a = T.alloc_barrier(arrive_count=128)
            # dg_shared is updated through scatter stores whose producer
            # threads differ from the following read-modify-write threads.
            # Warpgroup-wide mbarriers keep those phases ordered; a compiler
            # inserted storage sync can be hoisted out of the divergent
            # scatter predicate and does not close the race on SM120.
            bar_dg_k_init = T.alloc_barrier(arrive_count=128)
            bar_dg_k_rows = T.alloc_barrier(arrive_count=128)
            bar_dg_a_first = T.alloc_barrier(arrive_count=128)
            # 

            # add bar 09 128, for consumer S
            bar_09 = T.alloc_barrier(arrive_count=256)
            bar_10 = T.alloc_barrier(arrive_count=288)
            bar_11 = T.alloc_barrier(arrive_count=256)
            bar_12 = T.alloc_barrier(arrive_count=128)
            bar_13 = T.alloc_barrier(arrive_count=256)
            bar_14 = T.alloc_barrier(arrive_count=256)
            bar_15 = T.alloc_barrier(arrive_count=256)
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
            CONSUMER_K_NREG = 168
            CONSUMER_A_NREG = 160
            CONSUMER_S_NREG = 160

            # Prefetch the last chunk of data
            if state_v_first:
                T.copy(
                    h[batch_idx, chunk_start_idx + num_iters - 1, bh, 0:DV, 0:DK//2],
                    h_shared,
                )
            else:
                T.copy(
                    h[batch_idx, chunk_start_idx + num_iters - 1, bh, 0:DK//2, 0:DV],
                    h_shared,
                )
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
            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                if use_dht:
                    if state_v_first:
                        T.copy(dht[bb, bh, 0:DV, 0:DK//2], dh_fragment_0)
                        T.copy(dht[bb, bh, 0:DV, DK//2:DK], dh_fragment_1)
                    else:
                        T.copy(dht[bb, bh, 0:DK//2, 0:DV], dh_fragment_0)
                        T.copy(dht[bb, bh, DK//2:DK, 0:DV], dh_fragment_1)
                else:
                    T.clear(dh_fragment_0)
                    T.clear(dh_fragment_1)
                # K materializes iteration zero directly from the FP32 dht.
                # Subsequent iterations consume the shared state published by
                # S in the preceding iteration.
                # Publish readiness for the direct iteration-zero path.
                T.barrier_arrive(bar_dhtmp_ready)
                for i_s in T.serial(num_iters):
                    cur_idx = chunk_start_idx + num_iters - i_s - 1
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
                    if i_s > 0:
                        T.barrier_wait(bar_k_left_ready, (i_s - 1) % 2)
                        T.copy(dh_fragment_1, tmp_shared_4_1)
                        T.fence_proxy_async()
                        T.barrier_arrive(bar_dh_right_ready)
                    g_last_local_3[0] = g_exp_shared[block_S - 1]
                    # dS0 = g_last * dSt; scale each naturally indexed half fragment.
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            dh_fragment_0[j_v, j_k] *= g_last_local_3[0]
                            dh_fragment_1[j_v, j_k] *= g_last_local_3[0]
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            dh_fragment_0[j_k, j_v] *= g_last_local_3[0]
                            dh_fragment_1[j_k, j_v] *= g_last_local_3[0]
                    T.barrier_arrive(bar_04)

                    # 04, 05, 06, 07
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # dg_last += sum(dS0 * S0)
                    T.clear(reduce_fragment)
                    # W2: dh_fragment_1(right) x h_shared(right), right part
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK//2):
                            reduce_fragment[j_k//32 *64 + 
                            j_v//64*32+ 
                            j_v%8*4 + 
                            j_k%8//2
                            ,j_k%2] += dh_fragment_1[j_v,j_k] * h_shared[j_v, j_k]
                    else:
                        for j_k, j_v in T.Parallel(DK//2, DV):
                            reduce_fragment[j_v//64 *64 + 
                            j_k//32*32+ 
                            j_k%8*4 + 
                            j_v%8//2
                            ,j_v%2] += dh_fragment_1[j_k,j_v] * h_shared[j_k, j_v]
                    T.barrier_arrive(bar_08_2)
                    T.barrier_wait(bar_08_2, (i_s + 0) % 2)
                    # copy left part from hbm
                    if state_v_first:
                        T.copy(h[batch_idx,cur_idx,bh,0:DV,:DK//2],h_shared)
                    else:
                        T.copy(h[batch_idx,cur_idx,bh,:DK//2,0:DV],h_shared)
                    T.barrier_arrive(bar_08_3)
                    T.barrier_wait(bar_08_3,(i_s+0)%2)
                    # W1: dh_fragment_0(left part)
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK//2):
                            reduce_fragment[j_k//32 *64 + 
                            j_v//64*32+ 
                            j_v%8*4 + 
                            j_k%8//2
                            ,j_k%2] += dh_fragment_0[j_v,j_k] * h_shared[j_v, j_k]
                    else:
                        for j_k, j_v in T.Parallel(DK//2, DV):
                            reduce_fragment[j_v//64 *64 + 
                            j_k//32*32+ 
                            j_k%8*4 + 
                            j_v%8//2
                            ,j_v%2] += dh_fragment_0[j_k,j_v] * h_shared[j_k, j_v]

                    T.barrier_arrive(bar_09)
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    T.reduce_sum(T.reshape(reduce_fragment, (128 * 2,)), dg_last_local_3, dim=0, clear=True)
                    dg_shared[block_S - 1] += dg_last_local_3[0]
                    T.barrier_arrive(bar_11)

                    # 11
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    # dS0 += K^T @ dVg, splitting the output DK across two half-sized fragments.
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_2[:, 0:DK//2],
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_2[:, DK//2:DK],
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_2[:, 0:DK//2],
                            tmp_shared_2_3,
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_2[:, DK//2:DK],
                            tmp_shared_2_3,
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
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
                    # dS0 += Q^T @ dOg, with an M-split output.
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_1[:, 0:DK//2],
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_3,
                            tmp_shared_2_1[:, DK//2:DK],
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_1[:, 0:DK//2],
                            tmp_shared_2_3,
                            dh_fragment_0,
                            transpose_A=True,
                            clear_accum=False,
                        )
                        T.gemm(
                            tmp_shared_2_1[:, DK//2:DK],
                            tmp_shared_2_3,
                            dh_fragment_1,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_15)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    T.copy(dh_fragment_0, tmp_shared_4_1)
                    T.fence_proxy_async()
                    T.barrier_arrive(bar_dhtmp_ready)

                if use_dht:
                    if state_v_first:
                        for j_v, j_k in T.Parallel(DV, DK // 2):
                            dh0[bb, bh, j_v, j_k] = dh_fragment_0[j_v, j_k]
                            dh0[bb, bh, j_v, DK // 2 + j_k] = dh_fragment_1[j_v, j_k]
                    else:
                        for j_k, j_v in T.Parallel(DK // 2, DV):
                            dh0[bb, bh, j_k, j_v] = dh_fragment_0[j_k, j_v]
                            dh0[bb, bh, DK // 2 + j_k, j_v] = dh_fragment_1[j_k, j_v]

            elif tx < 256:
                T.set_max_nreg(CONSUMER_K_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 16 == 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # S2[S] dK
                    if i_s > 0:
                        T.copy(dk_frag_l, dqkv_shared[:, :DK//2])
                        T.copy(dk_frag_r, dqkv_shared[:, DK//2:])
                    T.barrier_arrive(bar_01)

                    # 01
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    # K-01
                    T.barrier_wait(bar_dhtmp_ready, (i_s + 0) % 2)
                    # The initial dSt is already available as the fp32 dht
                    # input (or is identically zero).  Materialize it in
                    # shared memory directly for the first reverse chunk
                    # instead of an unnecessary global-memory round trip.
                    if i_s == 0:
                        if state_v_first:
                            for j_v, j_k in T.Parallel(DV, DK // 2):
                                tmp_shared_4_1[j_v, j_k] = (
                                    T.Cast(qkva_dtype, dht[bb, bh, j_v, j_k])
                                    if use_dht else 0
                                )
                        else:
                            for j_k, j_v in T.Parallel(DK // 2, DV):
                                tmp_shared_4_1[j_k, j_v] = (
                                    T.Cast(qkva_dtype, dht[bb, bh, j_k, j_v])
                                    if use_dht else 0
                                )
                        T.fence_proxy_async()
                    # dV' = K @ dSt, reducing over the resident left half.
                    T.copy(tmp_shared_4_1, dh_left_fragment)
                    if state_v_first:
                        T.gemm(
                            k_shared[:, :DK//2],
                            tmp_shared_4_1,
                            dv_fragment,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        if i_s > 0:
                            T.barrier_arrive(bar_k_left_ready)
                        if i_s == 0:
                            for j_v, j_k in T.Parallel(DV, DK // 2):
                                tmp_shared_4_1[j_v, j_k] = (
                                    T.Cast(qkva_dtype, dht[bb, bh, j_v, DK // 2 + j_k])
                                    if use_dht else 0
                                )
                        else:
                            T.barrier_wait(bar_dh_right_ready, (i_s - 1) % 2)
                        T.fence_proxy_async()
                        T.gemm(
                            k_shared[:, DK//2:],
                            tmp_shared_4_1,
                            dv_fragment,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            k_shared[:, :DK//2],
                            tmp_shared_4_1,
                            dv_fragment,
                            clear_accum=True,
                        )
                        if i_s > 0:
                            T.barrier_arrive(bar_k_left_ready)
                        if i_s == 0:
                            for j_k, j_v in T.Parallel(DK // 2, DV):
                                tmp_shared_4_1[j_k, j_v] = (
                                    T.Cast(qkva_dtype, dht[bb, bh, DK // 2 + j_k, j_v])
                                    if use_dht else 0
                                )
                        else:
                            T.barrier_wait(bar_dh_right_ready, (i_s - 1) % 2)
                        T.fence_proxy_async()
                        T.gemm(
                            k_shared[:, DK//2:],
                            tmp_shared_4_1,
                            dv_fragment,
                            clear_accum=False,
                        )
                    # dV' = g_last/g * dV'
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    # dV' += Pg^T @ dO
                    T.gemm(
                        tmp_shared_1_1,
                        do_shared,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_03)

                    # 03 # diff from merge version
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # S2[1] dV'
                    T.copy(dv_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # dV = Ag^T @ dV'
                    T.gemm(
                        tmp_shared_1_2,
                        tmp_shared_2_1,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    # S2[S] dV
                    T.copy(dv_fragment, dqkv_shared)
                    T.barrier_arrive(bar_05)

                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # dVg = -g * dV
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] = (
                            -dv_fragment[j_s, j_v] * g_exp_shared[j_s]
                        )
                    # dg += sum(dVg * U)
                    T.copy(tmp_shared_2_3, odot_fragment_1)
                    for j_s, j_v in T.Parallel(block_S, DV):
                        odot_fragment_1[j_s, j_v] *= dv_fragment[j_s, j_v]
                    T.reduce_sum(odot_fragment_1, dg_fragment_1, dim=1, clear=True)
                    T.copy(dg_fragment_1, dg_shared)
                    T.barrier_arrive(bar_dg_k_init)
                    T.barrier_wait(bar_dg_k_init, i_s % 2)
                    # S2[3] dVg
                    T.copy(dv_fragment, tmp_shared_2_3)
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    # S2[2] K
                    T.copy(k_shared, odot_fragment_1)
                    T.copy(odot_fragment_1, tmp_shared_2_2)
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # dK = dV' @ dSt^T, reducing over the resident right half.
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_frag_r,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_frag_r,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    # Restore the left half retained before tmp_shared_4_1 was
                    # overwritten by the right half for the dV' calculation.
                    T.copy(dh_left_fragment, tmp_shared_4_1)
                    T.fence_proxy_async()
                    # dK = dV' @ dSt^T, reducing over the left half loaded by K.
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_frag_l,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_1,
                            tmp_shared_4_1,
                            dk_frag_l,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    # dK = g_last/g * dK
                    for j_s, j_k in T.Parallel(block_S, DK//2):
                        dk_frag_r[j_s, j_k] *= g_rev_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK//2):
                        dk_frag_l[j_s, j_k] *= g_rev_exp_shared[j_s]
                    # Stage dg -= sum(K * dK) through dqkv_shared to avoid half/full-width layout conflicts.
                    T.copy(dk_frag_l, dqkv_shared[:, :DK//2])
                    T.copy(dk_frag_r, dqkv_shared[:, DK//2:])
                    T.fence_proxy_async()
                    # T.barrier_arrive(bar_s4_merge)
                    # T.barrier_wait(bar_s4_merge, (i_s + 0) % 2)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        odot_fragment_1[j_s, j_k] *= -dqkv_shared[j_s, j_k]
                    T.reduce_sum(odot_fragment_1, dg_fragment_1, dim=1, clear=True)

                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_1[j_s]

                    # dg_last += sum(K * dK)
                    T.reduce_sum(dg_fragment_1, dg_last_local_1, dim=0, clear=True)
                    T.barrier_arrive(bar_dg_k_rows)
                    T.barrier_wait(bar_dg_k_rows, i_s % 2)
                    # Sg[S] dg
                    dg_shared[block_S - 1] -= dg_last_local_1[0]

                    T.barrier_arrive(bar_08_1)
                    T.barrier_wait(bar_08_1,(i_s+0)%2)
                    # Accumulate the resident right half of dK += dVg @ h directly into stage-07-shaped dk_frag_r.
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_frag_r,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_frag_r,
                            transpose_B=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_08_2)
                    T.barrier_wait(bar_08_3,(i_s)%2)
                    # dK += dVg @ h for the left half loaded by S.
                    if state_v_first:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_frag_l,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            tmp_shared_2_3,
                            h_shared,
                            dk_frag_l,
                            transpose_B=True,
                            clear_accum=False,
                        )

                    T.barrier_arrive(bar_10)
                    T.barrier_wait(bar_10, (i_s + 0) % 2)

                    # 12
                    T.barrier_wait(bar_12, (i_s + 0) % 2)
                    # dK += dP^T @ Q for the left and right halves.
                    T.gemm(
                        tmp_shared_1_1,
                        tmp_shared_2_1[:, :DK//2],
                        dk_frag_l,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.gemm(
                        tmp_shared_1_1,
                        tmp_shared_2_1[:, DK//2:],
                        dk_frag_r,
                        transpose_A=True,
                        clear_accum=False,
                    )

                    T.barrier_arrive(bar_13)
                    T.barrier_wait(bar_13, (i_s + 0) % 2)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    # dK += dAs @ K for the left and right halves.
                    T.gemm(
                        tmp_shared_1_2, tmp_shared_2_2[:, :DK//2], dk_frag_l, clear_accum=False
                    )
                    T.gemm(
                        tmp_shared_1_2, tmp_shared_2_2[:, DK//2:], dk_frag_r, clear_accum=False
                    )

                for j_s, j_k in T.Parallel(block_S, DK//2):
                    if seq_start_idx + j_s < seq_end_idx:
                        dk[batch_idx, seq_start_idx + j_s, bh, j_k] = dk_frag_l[j_s, j_k]
                for j_s, j_k in T.Parallel(block_S, DK//2):
                    if seq_start_idx + j_s < seq_end_idx:
                        dk[batch_idx, seq_start_idx + j_s, bh, DK//2 + j_k] = dk_frag_r[j_s, j_k]

            elif tx < 384:
                T.set_max_nreg(CONSUMER_A_NREG, 1)

                for i_s in T.serial(num_iters):
                    cur_idx = chunk_start_idx + num_iters - i_s - 1
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
                    # issue 30 opt1, q_shared -> tmp_shared_2_1
                    # Read Q before bar_03.arrive: K overwrites tmp_shared_2_1 with dV'
                    # immediately after completion. If A arrives last and reads afterward,
                    # K's write races with A's read, so A sees invalid dV' data instead of Q.
                    T.copy(tmp_shared_2_1, odot_fragment_2)
                    T.barrier_arrive(bar_03)
                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # 1st use of h_share,not need to mv, prefretch
                    # U = K @ S0
                    if state_v_first:
                        T.gemm(k_shared[:,:DK//2],h_shared, u_fragment,transpose_B=True,clear_accum=True,)
                    else:
                        T.gemm(k_shared[:,:DK//2], h_shared, u_fragment, clear_accum=True)
                    # egaer implement:copy_immeidately 
                    if state_v_first:
                        T.copy(h[batch_idx,cur_idx,bh,0:DV,DK//2:DK],h_shared)
                        T.gemm(k_shared[:,DK//2:],h_shared, u_fragment,transpose_B=True,clear_accum=False)
                    else:
                        T.copy(h[batch_idx,cur_idx,bh,DK//2:DK,0:DV],h_shared)
                        T.gemm(k_shared[:,DK//2:], h_shared, u_fragment, clear_accum=False)
                    T.barrier_arrive(bar_04)
                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # S2[3] U
                    T.copy(u_fragment, tmp_shared_2_3)
                    # W = V - g * U
                    for j_s, j_v in T.Parallel(block_S, DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, DV):
                        if seq_start_idx + (num_iters - i_s - 1) * block_S + j_s < seq_end_idx:
                            u_fragment[j_s, j_v] += v[
                                batch_idx,
                                seq_start_idx + (num_iters - i_s - 1) * block_S + j_s,
                                bh,
                                j_v,
                            ]
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
                    # dP = s * G * dPg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= scale
                    # S1[1] dP
                    T.copy(dp_fragment, tmp_shared_1_1)
                    # 08_1
                    T.barrier_arrive(bar_08_1)
                    T.barrier_wait(bar_08_1,(i_s+0)%2)
                    # dQ = dO @ h for the right half.
                    if state_v_first:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_frag_r,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_frag_r,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    # 08_2
                    T.barrier_arrive(bar_08_2)

                    T.barrier_wait(bar_08_3, (i_s + 0) % 2)
                    # dQ = dO @ h for the left half loaded by S.
                    if state_v_first:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_frag_l,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            do_shared,
                            h_shared,
                            dq_frag_l,
                            transpose_B=True,
                            clear_accum=True,
                        )
    
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # dQ = s * g * dQ for both the right and left halves.
                    for j_s, j_k in T.Parallel(block_S, DK//2):
                        dq_frag_r[j_s, j_k] *= g_exp_shared[j_s]
                        dq_frag_l[j_s, j_k] *= g_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK//2):
                        dq_frag_r[j_s, j_k] *= scale
                        dq_frag_l[j_s, j_k] *= scale
                    # S2[1] Q
                    T.copy(odot_fragment_2, tmp_shared_2_1)
                    # Move dg += sum(Q * dQ) to stage 10, staged through dqkv_shared.
                    T.barrier_arrive(bar_10)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    T.copy(dq_frag_l, dqkv_shared[:, :DK//2])
                    T.copy(dq_frag_r, dqkv_shared[:, DK//2:])
                    T.barrier_arrive(bar_s4_dot_a)
                    T.barrier_wait(bar_s4_dot_a, (i_s + 0) % 2)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        odot_fragment_2[j_s, j_k] *= dqkv_shared[j_s, j_k]
                    T.reduce_sum(odot_fragment_2, dg_fragment_2, dim=1, clear=True)
                    # dQ += dP @ K for the left and right halves.
                    T.gemm(
                        tmp_shared_1_1, tmp_shared_2_2[:, :DK//2], dq_frag_l, clear_accum=False
                    )
                    T.gemm(
                        tmp_shared_1_1, tmp_shared_2_2[:, DK//2:], dq_frag_r, clear_accum=False
                    )
                    # S2[S] dQ
                    T.copy(dq_frag_l, dqkv_shared[:, :DK//2])
                    T.copy(dq_frag_r, dqkv_shared[:, DK//2:])
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
                    # The reinterpret-transpose loop fixes a_fragment's layout, which cannot
                    # satisfy the stage-10 row partition of dg_fragment_2. Reduce into the
                    # independent dg_fragment_2b target and merge the results when writing Sg.
                    T.reduce_sum(a_fragment, dg_fragment_2b, dim=1, clear=True)
                    # Sg[S] dg.  The two fragments have different row/thread
                    # mappings, so complete the first scatter before the
                    # second read-modify-write phase begins.
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_2[j_s]
                    T.barrier_arrive(bar_dg_a_first)
                    T.barrier_wait(bar_dg_a_first, i_s % 2)
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_2b[j_s]
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

                elif tx < 384 + 64:
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
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_02)
                        T.barrier_wait(bar_02, (i_s + 0) % 2)

                        T.barrier_wait(bar_10, (i_s + 0) % 2)
                        if state_v_first:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DV,
                                    0:DK//2,
                                ],
                                h_shared,
                                barrier=bar_02,
                            )
                        else:
                            T.tma_copy(
                                h[
                                    batch_idx,
                                    chunk_start_idx + chunk_idx,
                                    bh,
                                    0:DK//2,
                                    0:DV,
                                ],
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

    if is_varlen:
        # The persistent kernel covers all real tokens and one trailing tile.
        # Clear the complete padding domain separately without first writing
        # zeros over the much larger valid output domain.
        tilelang_zero_bwd_padding_kernel = tilelang_zero_bwd_padding(
            H,
            K,
            V,
            qkva_dtype=q.dtype,
            g_dtype=g.dtype,
            b_dtype=b.dtype,
            seqlen_dtype=cu_seqlens.dtype,
        )
        tilelang_zero_bwd_padding_kernel(cu_seqlens, dq, dk, dv, dg, db)

    return dq, dk, dv, dg, db, dh0
