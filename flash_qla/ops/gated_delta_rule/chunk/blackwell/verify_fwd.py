# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Single-chunk Gated DeltaNet forward for DFlash verify blocks of <=16 tokens.

Each CTA handles one (sequence, value head, value tile). Stacking K and Q
fills four MMA warps even for a 16-token block. With c = cumsum(g), compute

    M[i,j] = exp(c[i] - c[j]) * beta[i] * dot(k[i], k[j]), j < i
    P[i,j] = exp(c[i] - c[j]) * scale * dot(q[i], k[j]), j <= i
    W[i]   = beta[i] * (v[i] - exp(c[i]) * (K @ S0)[i])

Solve (I + M) NV = W by forward substitution in registers, then form
O = P @ NV + exp(c) * scale * Q @ S0 and the rank-16 final-state update.
The state update splits its FP32 right operand into two bf16/fp16 pieces;
accumulating both avoids rounding NV to just one low-precision operand.
The state-only specialization omits Q loads, Q projections, and O entirely
for SSM state reconstruction after verify.
Optionally caching FP32 NV and c lets reconstruction apply just the accepted
rank-a state update, reusing the original initial state and verify inputs.
"""

from typing import Optional

import torch
import tilelang
import tilelang.language as T


LOG2E = 1.4426950408889634


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def tilelang_gdn_verify_fwd(
    HQ,
    HV,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    block_DV=128,
    state_only=False,
    store_cache=False,
    use_accepted_lengths=False,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size
    group = HV // HQ
    projection_rows = block_S if state_only else 2 * block_S
    cache_dtype = accum_dtype if store_cache else qkva_dtype
    prefix_dtype = accum_dtype if store_cache else g_dtype
    num_lengths = T.dynamic("num_lengths")

    @T.prim_func
    def tilelang_gdn_verify_fwd_kernel(
        q: T.Tensor((num_tokens, HQ, DK), qkva_dtype),
        k: T.Tensor((num_tokens, HQ, DK), qkva_dtype),
        v: T.Tensor((num_tokens, HV, DV), qkva_dtype),
        g: T.Tensor((num_tokens, HV), g_dtype),
        b: T.Tensor((num_tokens, HV), b_dtype),
        cu_seqlens: T.Tensor((batch_size + 1,), seqlen_dtype),
        h0: T.Tensor((batch_size, HV, DK, DV), h0_dtype),
        o: T.Tensor((num_tokens, HV, DV), o_dtype),
        ht: T.Tensor((batch_size, HV, DK, DV), ht_dtype),
        nv_cache: T.Tensor((num_tokens, HV, DV), cache_dtype),
        gate_cache: T.Tensor((num_tokens, HV), prefix_dtype),
        accepted_lengths: T.Tensor((num_lengths,), seqlen_dtype),
    ):
        with T.Kernel(batch_size * HV * (DV // block_DV), threads=128) as (block,):
            # Prevent ptxas from trading occupancy for excessive live registers.
            T.annotate_min_blocks_per_sm(4)
            bx = block // (DV // block_DV)
            bv = block % (DV // block_DV)
            bb = bx // HV
            bh = bx % HV
            bhg = bh // group
            dv_start = bv * block_DV
            bos = cu_seqlens[bb]
            eos = bos + accepted_lengths[bb] if use_accepted_lengths else cu_seqlens[bb + 1]
            tx = T.get_thread_binding()

            qk_shared = T.alloc_shared((2 * block_S, DK), qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), qkva_dtype)
            s0_shared = T.alloc_shared((DK, block_DV), qkva_dtype)
            b_shared = T.alloc_shared((block_S,), accum_dtype, scope="shared")
            cum_shared = T.alloc_shared((block_S,), accum_dtype, scope="shared")
            cumprod_shared = T.alloc_shared((block_S,), accum_dtype, scope="shared")
            decay_shared = T.alloc_shared((block_S,), accum_dtype, scope="shared")
            m_shared = T.alloc_shared((block_S, block_S), accum_dtype)
            wqk_shared = T.alloc_shared((block_S, block_S), accum_dtype)
            ks_shared = T.alloc_shared((block_S, block_DV), accum_dtype)
            qs_shared = T.alloc_shared((block_S, block_DV), accum_dtype)
            nv_shared = T.alloc_shared((block_S, block_DV), accum_dtype)
            nvd_shared = T.alloc_shared((block_S, block_DV), qkva_dtype)
            nvd_lo_shared = T.alloc_shared((block_S, block_DV), qkva_dtype)

            kkqk_fragment = T.alloc_fragment((2 * block_S, block_S), accum_dtype)
            kqs_fragment = T.alloc_fragment((projection_rows, block_DV), accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), accum_dtype)
            sfin_fragment = T.alloc_fragment((DK, block_DV), accum_dtype)
            gate_local = T.alloc_local((1,), accum_dtype)
            nv_local = T.alloc_local((block_S,), accum_dtype)

            # Zero padding leaves the gate prefix flat and padded updates zero.
            for i, d in T.Parallel(block_S, DK):
                if bos + i < eos:
                    if state_only:
                        # Pad KK to 32 rows so its MMA still uses four warps.
                        qk_shared[block_S + i, d] = 0.0
                    else:
                        qk_shared[block_S + i, d] = q[bos + i, bhg, d]
                    qk_shared[i, d] = k[bos + i, bhg, d]
                else:
                    qk_shared[block_S + i, d] = 0.0
                    qk_shared[i, d] = 0.0
            for i, d in T.Parallel(block_S, block_DV):
                v_shared[i, d] = T.if_then_else(
                    bos + i < eos, v[bos + i, bh, dv_start + d], 0.0
                )

            # Two 16-lane scans in the first warp; only one writes the result.
            if tx < 32:
                gate_local[0] = T.if_then_else(
                    bos + tx % block_S < eos, g[bos + tx % block_S, bh], 0.0
                )
                for step in T.unroll(4):
                    shifted = T.shfl_up(gate_local[0], 1 << step, block_S)
                    gate_local[0] += T.if_then_else(
                        tx % block_S >= (1 << step), shifted, 0.0
                    )
                gate_last = T.shfl_sync(gate_local[0], block_S - 1, block_S)
                if tx < block_S:
                    cum_shared[tx] = gate_local[0]
                    cumprod_shared[tx] = T.exp2(gate_local[0] * LOG2E)
                    decay_shared[tx] = T.exp2((gate_last - gate_local[0]) * LOG2E)
                    b_shared[tx] = T.if_then_else(bos + tx < eos, b[bos + tx, bh], 0.0)
                    if store_cache:
                        # DV tiles share the same prefix; only one may write it.
                        if bv == 0 and bos + tx < eos:
                            gate_cache[bos + tx, bh] = gate_local[0]

            T.gemm(
                qk_shared, qk_shared[:block_S, :], kkqk_fragment,
                transpose_B=True, clear_accum=True,
            )
            for i, j in T.Parallel(2 * block_S, block_S):
                if i < block_S:
                    m_shared[i, j] = T.if_then_else(
                        j < i,
                        T.exp2((cum_shared[i] - cum_shared[j]) * LOG2E)
                        * b_shared[i] * kkqk_fragment[i, j],
                        0.0,
                    )
                elif not state_only:
                    wqk_shared[i - block_S, j] = T.if_then_else(
                        j <= i - block_S,
                        T.exp2((cum_shared[i - block_S] - cum_shared[j]) * LOG2E)
                        * scale * kkqk_fragment[i, j],
                        0.0,
                    )

            if use_initial_state:
                for d, e in T.Parallel(DK, block_DV):
                    s0_shared[d, e] = h0[bb, bh, d, dv_start + e]
                T.gemm(
                    qk_shared[:projection_rows, :], s0_shared, kqs_fragment,
                    clear_accum=True,
                )
                for i, d in T.Parallel(projection_rows, block_DV):
                    if i < block_S:
                        ks_shared[i, d] = kqs_fragment[i, d]
                    else:
                        qs_shared[i - block_S, d] = kqs_fragment[i, d] * scale
            # Each lane owns a value column and all 16 time steps. No matrix
            # inverse, inter-row shared-memory round trips, or TF32 solve.
            for i in T.unroll(block_S):
                if use_initial_state:
                    nv_local[i] = b_shared[i] * (
                        T.cast(v_shared[i, tx % block_DV], accum_dtype)
                        - cumprod_shared[i] * ks_shared[i, tx % block_DV]
                    )
                else:
                    nv_local[i] = b_shared[i] * T.cast(
                        v_shared[i, tx % block_DV], accum_dtype
                    )
            for i in T.unroll(1, block_S):
                for r in T.unroll(i):
                    nv_local[i] -= m_shared[i, r] * nv_local[r]
            for i in T.unroll(block_S):
                if tx < block_DV:
                    nv_shared[i, tx] = nv_local[i]

            if store_cache:
                for i, d in T.Parallel(block_S, block_DV):
                    if bos + i < eos:
                        nv_cache[bos + i, bh, dv_start + d] = nv_shared[i, d]

            if not state_only:
                T.gemm(wqk_shared, nv_shared, o_fragment, clear_accum=True)
                if use_initial_state:
                    for i, d in T.Parallel(block_S, block_DV):
                        o_fragment[i, d] += cumprod_shared[i] * qs_shared[i, d]
                for i, d in T.Parallel(block_S, block_DV):
                    if bos + i < eos:
                        o[bos + i, bh, dv_start + d] = o_fragment[i, d]

            if store_final_state:
                for j, d in T.Parallel(block_S, block_DV):
                    nvd_shared[j, d] = decay_shared[j] * nv_shared[j, d]
                for j, d in T.Parallel(block_S, block_DV):
                    nvd_lo_shared[j, d] = (
                        decay_shared[j] * nv_shared[j, d]
                        - T.cast(nvd_shared[j, d], accum_dtype)
                    )
                T.gemm(
                    qk_shared[:block_S, :], nvd_shared, sfin_fragment,
                    transpose_A=True, clear_accum=True,
                )
                T.gemm(
                    qk_shared[:block_S, :], nvd_lo_shared, sfin_fragment,
                    transpose_A=True,
                )
                cum_last = cumprod_shared[block_S - 1]
                for d, e in T.Parallel(DK, block_DV):
                    if use_initial_state:
                        sfin_fragment[d, e] += cum_last * h0[bb, bh, d, dv_start + e]
                    ht[bb, bh, d, dv_start + e] = sfin_fragment[d, e]

    return tilelang_gdn_verify_fwd_kernel


def gdn_verify_fwd(
    q: Optional[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    final_state: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    state_only: bool = False,
    nv_cache: Optional[torch.Tensor] = None,
    gate_cache: Optional[torch.Tensor] = None,
    accepted_lengths: Optional[torch.Tensor] = None,
):
    """GDN verify forward; every sequence must contain at most 16 tokens.

    Inputs q/k are (total, HQ, 128), v is (total, HV, 128), and g/beta
    are (total, HV). q/k/v use bf16 or fp16; g contains raw log gates.
    cu_seqlens is (B + 1,), with HV a multiple of HQ. State tensors use
    K-major (B, HV, 128, 128) layout. The default scale is 128 ** -0.5.

    Returns (out, final_state). Both outputs may be preallocated, and
    final_state may alias initial_state. Without an initial state, start
    from zero and allocate the final state in fp32 by default.

    Set state_only=True for SSM state reconstruction: returns (None,
    final_state), skips all Q/O work and O allocation, and permits q=None.
    In this mode q, scale, and out are ignored; a supplied out is unchanged.

    To cache verify intermediates for gdn_reconstruct_state, supply both
    nv_cache (total, HV, 128) and gate_cache (total, HV), contiguous fp32
    output buffers that do not alias inputs or other outputs. The returned
    pair is unchanged. Preserve the original initial_state until reconstruction;
    the cache is tied to that state and these K/V/g/beta inputs.

    accepted_lengths optionally selects each sequence's prefix without
    repacking inputs or changing cu_seqlens. It is a CUDA int32 tensor (B,)
    with 0 <= accepted_lengths[b] <= cu_seqlens[b+1] - cu_seqlens[b] <= 16.
    This also permits an uncached state-only replay on the original inputs.
    Only selected prefixes are written to out and cache buffers.
    """
    if state_only:
        # Unused tensor argument; no Q tensor or O allocation is needed.
        q = k
        scale = 1.0
    elif q is None:
        raise ValueError("q is required unless state_only=True")
    total, HQ, DK = q.shape
    HV, DV = v.shape[1:]
    batch = cu_seqlens.shape[0] - 1
    assert HV % HQ == 0, "GVA requires HV % HQ == 0"
    assert DK == DV == 128, f"unsupported geometry DK={DK} DV={DV}"
    assert chunk_size == 16, "GDN verify requires chunk_size=16"
    store_cache = nv_cache is not None
    if store_cache != (gate_cache is not None):
        raise ValueError("nv_cache and gate_cache must be supplied together")
    if store_cache:
        for name, cache, shape in (
            ("nv_cache", nv_cache, (total, HV, DV)),
            ("gate_cache", gate_cache, (total, HV)),
        ):
            if (cache.shape != shape or cache.dtype != torch.float32
                    or cache.device != k.device or not cache.is_contiguous()):
                raise ValueError(f"{name} must be contiguous fp32 {shape} on {k.device}")
    if accepted_lengths is not None:
        _check_accepted_lengths(accepted_lengths, batch, k.device)

    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    if scale is None:
        scale = DK ** -0.5
    use_initial_state = initial_state is not None
    state_dtype = initial_state.dtype if use_initial_state else torch.float32
    if not state_only and out is None:
        out = torch.empty((total, HV, DV), dtype=q.dtype, device=q.device)
    if final_state is None:
        final_state = torch.empty((batch, HV, DK, DV), dtype=state_dtype, device=q.device)
    if initial_state is None:
        # The zero-state specialization never reads this argument.
        initial_state = final_state

    kernel = tilelang_gdn_verify_fwd(
        HQ, HV, DK, DV, chunk_size, scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=v.dtype if state_only else out.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        use_initial_state=use_initial_state,
        store_final_state=True,
        # Small batches need more CTAs; large batches reuse Q/K across all DV.
        block_DV=64 if batch * HV <= 192 else 128,
        state_only=state_only,
        store_cache=store_cache,
        use_accepted_lengths=accepted_lengths is not None,
    )
    # Disabled cache/length specializations ignore the input placeholders.
    kernel(q, k, v, g, beta, cu_seqlens, initial_state,
           v if state_only else out, final_state,
           nv_cache if store_cache else v, gate_cache if store_cache else g,
           cu_seqlens if accepted_lengths is None else accepted_lengths)
    return None if state_only else out, final_state


def _check_accepted_lengths(lengths, batch, device):
    if (lengths.shape != (batch,) or lengths.dtype != torch.int32
            or lengths.device != device or not lengths.is_contiguous()):
        raise ValueError(f"accepted_lengths must be contiguous int32 ({batch},) on {device}")


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def tilelang_gdn_reconstruct_state(
    HQ, HV, DK, DV, io_dtype, h0_dtype, ht_dtype, use_initial_state,
    block_DV=128,
):
    batch = T.dynamic("batch")
    total = T.dynamic("total")
    block_S = 16

    @T.prim_func
    def tilelang_gdn_reconstruct_state_kernel(
        k: T.Tensor((total, HQ, DK), io_dtype),
        nv: T.Tensor((total, HV, DV), "float32"),
        prefix: T.Tensor((total, HV), "float32"),
        cu_seqlens: T.Tensor((batch + 1,), "int32"),
        accepted: T.Tensor((batch,), "int32"),
        h0: T.Tensor((batch, HV, DK, DV), h0_dtype),
        ht: T.Tensor((batch, HV, DK, DV), ht_dtype),
    ):
        with T.Kernel(batch * HV * (DV // block_DV), threads=128) as block:
            T.annotate_min_blocks_per_sm(4)
            bv = block % (DV // block_DV)
            bh = (block // (DV // block_DV)) % HV
            bb = block // ((DV // block_DV) * HV)
            bhg = bh // (HV // HQ)
            dv_start = bv * block_DV
            bos = cu_seqlens[bb]
            length = accepted[bb]
            tx = T.get_thread_binding()
            k_shared = T.alloc_shared((block_S, DK), io_dtype)
            hi_shared = T.alloc_shared((block_S, block_DV), io_dtype)
            lo_shared = T.alloc_shared((block_S, block_DV), io_dtype)
            decay_shared = T.alloc_shared((block_S,), "float32", scope="shared")
            state_fragment = T.alloc_fragment((DK, block_DV), "float32")

            # A prefix of NV depends only on that prefix, never later drafts.
            c_last = T.if_then_else(length > 0, prefix[bos + length - 1, bh], 0.0)
            if tx < block_S:
                decay_shared[tx] = T.if_then_else(
                    tx < length,
                    T.exp2((c_last - prefix[bos + tx, bh]) * LOG2E), 0.0,
                )
            for i, d in T.Parallel(block_S, DK):
                k_shared[i, d] = T.if_then_else(i < length, k[bos + i, bhg, d], 0.0)
            for i, d in T.Parallel(block_S, block_DV):
                scaled_nv = T.if_then_else(
                    i < length, decay_shared[i] * nv[bos + i, bh, dv_start + d], 0.0,
                )
                hi = T.cast(scaled_nv, io_dtype)
                hi_shared[i, d] = hi
                lo_shared[i, d] = scaled_nv - T.cast(hi, "float32")
            T.gemm(k_shared, hi_shared, state_fragment,
                   transpose_A=True, clear_accum=True)
            T.gemm(k_shared, lo_shared, state_fragment, transpose_A=True)
            decay = T.exp2(c_last * LOG2E)
            for d, e in T.Parallel(DK, block_DV):
                if use_initial_state:
                    state_fragment[d, e] += decay * h0[bb, bh, d, dv_start + e]
                ht[bb, bh, d, dv_start + e] = state_fragment[d, e]

    return tilelang_gdn_reconstruct_state_kernel


def gdn_reconstruct_state(
    k: torch.Tensor,
    nv_cache: torch.Tensor,
    gate_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    accepted_lengths: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
):
    """Reconstruct accepted-prefix states from gdn_verify_fwd's fp32 caches.

    k and cu_seqlens describe the original verify input, without repacking.
    accepted_lengths is a CUDA int32 (B,) tensor, bounded by each verified
    sequence's length (0..16). Pass the same initial state used by verify,
    or None for a zero initial state. final_state may alias initial_state.
    Returns final_state in K-major (B, HV, 128, 128) layout.
    """
    total, HQ, DK = k.shape
    HV, DV = nv_cache.shape[1:]
    batch = cu_seqlens.numel() - 1
    assert DK == DV == 128 and HV % HQ == 0
    _check_accepted_lengths(accepted_lengths, batch, k.device)
    for name, cache, shape in (
        ("nv_cache", nv_cache, (total, HV, DV)),
        ("gate_cache", gate_cache, (total, HV)),
    ):
        if (cache.shape != shape or cache.dtype != torch.float32
                or cache.device != k.device or not cache.is_contiguous()):
            raise ValueError(f"{name} must be contiguous fp32 {shape} on {k.device}")
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    use_initial_state = initial_state is not None
    if final_state is None:
        final_state = torch.empty(
            (batch, HV, DK, DV), device=k.device,
            dtype=initial_state.dtype if use_initial_state else torch.float32,
        )
    if initial_state is None:
        initial_state = final_state
    kernel = tilelang_gdn_reconstruct_state(
        HQ, HV, DK, DV, k.dtype, initial_state.dtype, final_state.dtype,
        use_initial_state, block_DV=64 if batch * HV <= 192 else 128,
    )
    kernel(k, nv_cache, gate_cache, cu_seqlens, accepted_lengths,
           initial_state, final_state)
    return final_state
