# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import tilelang
import tilelang.language as T


@tilelang.jit()
def tilelang_correct_h0_ws(
    H, DK, DV, res_dtype, accum_dtype, h_buffer_dtype, m_buffer_dtype,
    seqlen_dtype, mask_dtype, use_raw_h0, state_v_first,
    reverse=False, transpose_m=False,
):
    """Exclusive state scan with a Hopper-style load warp, sized for SM120.

    With 16-bit M, two M tiles plus single h and previous-state tiles take
    at most 88 KiB. FP32 M uses one stage (96 KiB total). Both fit the 99 KiB
    CTA limit. The math warps store each boundary directly, avoiding another
    shared output buffer. Only input loads overlap; the recurrence is ordered.
    """
    partitions = T.dynamic("partitions")
    sequences = T.dynamic("sequences")
    block_DV = 32
    stages = 2 if m_buffer_dtype.itemsize == 2 else 1
    tile_shape = (block_DV, DK) if state_v_first else (DK, block_DV)
    state_shape = (partitions, H, DV, DK) if state_v_first else (partitions, H, DK, DV)
    seed_shape = (sequences, H, DV, DK) if state_v_first else (sequences, H, DK, DV)
    tiles = T.ceildiv(DV, block_DV)

    @T.macro
    def scan(block, seed, h, m, fallback, seq_map, out):
        bb, bh, bv = block // (H * tiles), block // tiles % H, block % tiles
        lo = bv * block_DV
        start, end = seq_map[bb], seq_map[bb + 1]
        count = end - start - 1
        m_stage = T.alloc_shared((stages, DK, DK), m_buffer_dtype)
        h_shared = T.alloc_shared(tile_shape, h_buffer_dtype)
        previous = T.alloc_shared(tile_shape, m_buffer_dtype)
        state = T.alloc_fragment(tile_shape, accum_dtype)
        ready = T.alloc_barrier(arrive_count=[32] * stages)
        m_free = T.alloc_barrier(arrive_count=[128] * stages)
        h_free = T.alloc_barrier(arrive_count=128)
        tx = T.get_thread_binding()

        if tx < 128:
            T.set_max_nreg(240, 1)
            if use_raw_h0:
                if state_v_first:
                    T.copy(seed[bb, bh, lo:lo + block_DV, :], state)
                else:
                    T.copy(seed[bb, bh, :, lo:lo + block_DV], state)
            else:
                T.clear(state)
            for i in T.serial(count):
                idx = end - 1 - i if reverse else start + i
                stage = i % stages
                if state_v_first:
                    T.copy(state, out[idx, bh, lo:lo + block_DV, :])
                else:
                    T.copy(state, out[idx, bh, :, lo:lo + block_DV])
                if fallback[idx, bh]:
                    T.copy(state, previous)
                T.sync_threads(3, 128)
                T.barrier_wait(ready[stage], (i // stages) % 2)
                T.copy(h_shared, state)
                # Release h independently: its next load can overlap this MMA.
                T.sync_threads(3, 128)
                T.barrier_arrive(h_free)
                if fallback[idx, bh]:
                    if state_v_first:
                        T.gemm(previous, m_stage[stage, :, :], state,
                               transpose_B=not transpose_m, clear_accum=False)
                    else:
                        T.gemm(m_stage[stage, :, :], previous, state,
                               transpose_A=transpose_m, clear_accum=False)
                T.sync_threads(3, 128)
                T.barrier_arrive(m_free[stage])
            last = start if reverse else end - 1
            if state_v_first:
                T.copy(state, out[last, bh, lo:lo + block_DV, :])
            else:
                T.copy(state, out[last, bh, :, lo:lo + block_DV])
        else:
            T.set_max_nreg(24, 0)
            if tx < 160:
                for i in T.serial(count):
                    idx = end - 1 - i if reverse else start + i
                    stage = i % stages
                    T.barrier_wait(m_free[stage], (i // stages + 1) % 2)
                    if fallback[idx, bh]:
                        T.tma_copy(m[idx, bh, :, :], m_stage[stage, :, :],
                                   barrier=ready[stage])
                    T.barrier_wait(h_free, (i + 1) % 2)
                    if state_v_first:
                        T.tma_copy(h[idx, bh, lo:lo + block_DV, :], h_shared,
                                   barrier=ready[stage])
                    else:
                        T.tma_copy(h[idx, bh, :, lo:lo + block_DV], h_shared,
                                   barrier=ready[stage])
                    T.barrier_arrive(ready[stage])

    if use_raw_h0:
        @T.prim_func
        def correct_h0_ws_kernel(
            seed: T.Tensor(seed_shape, res_dtype),
            h: T.Tensor(state_shape, h_buffer_dtype),
            m: T.Tensor((partitions, H, DK, DK), m_buffer_dtype),
            fallback: T.Tensor((partitions, H), mask_dtype),
            seq_map: T.Tensor((sequences + 1,), seqlen_dtype),
            out: T.Tensor(state_shape, res_dtype),
        ):
            with T.Kernel(sequences * H * tiles, threads=256) as (block,):
                scan(block, seed, h, m, fallback, seq_map, out)
    else:
        @T.prim_func
        def correct_h0_ws_kernel(
            h: T.Tensor(state_shape, h_buffer_dtype),
            m: T.Tensor((partitions, H, DK, DK), m_buffer_dtype),
            fallback: T.Tensor((partitions, H), mask_dtype),
            seq_map: T.Tensor((sequences + 1,), seqlen_dtype),
            out: T.Tensor(state_shape, res_dtype),
        ):
            with T.Kernel(sequences * H * tiles, threads=256) as (block,):
                scan(block, h, h, m, fallback, seq_map, out)

    return correct_h0_ws_kernel
