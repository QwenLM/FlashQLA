# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import torch
import tilelang
import tilelang.language as T

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_fwd(
    H,
    Hg,
    DK,
    DV,
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
    has_seqlens,
    block_DV=128,
    threads=128,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")  # = q_len (D); D fixed per call
    q_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    g_shape = (batch_size, num_tokens, H)
    s_shape = (batch_size, H, DK, DV)
    n_vt = (DV + block_DV - 1) // block_DV

    @T.prim_func
    def kernel(
        q: T.Tensor(q_shape, qkva_dtype),
        k: T.Tensor(q_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        g: T.Tensor(g_shape, g_dtype),
        b: T.Tensor(g_shape, b_dtype),
        h0: T.Tensor(s_shape, h0_dtype),
        seqlens: T.Tensor([batch_size], seqlen_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ht: T.Tensor(s_shape, ht_dtype),
    ):
        # Gemm-free, memory-bound decode recurrence. One CTA owns (sequence bb, V-head bh,
        # V-column-tile bv). State S is kept [block_DV, DK] (V-major rows) in an fp32 fragment;
        # the two GEMVs are reductions over the last (DK) dim and the rank-1 is a T.Parallel
        # outer product. No tensor-core gemm -> no M-padding, no warp-partition constraint.
        with T.Kernel(n_vt * batch_size * H, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt
            bv = bbhv % n_vt
            bb = bbh // H
            bh = bbh % H
            bhg = bh // (H // Hg)
            v0 = bv * block_DV

            L = T.alloc_var("int32")
            L = seqlens[bb] if has_seqlens else num_tokens

            S = T.alloc_fragment((block_DV, DK), accum_dtype)  # state [V-tile, K], fp32
            prod = T.alloc_fragment((block_DV, DK), accum_dtype)
            q_s = T.alloc_shared((1, DK), qkva_dtype)  # 2-D staging (1-D slice copies fail layout)
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            v_s = T.alloc_shared((1, block_DV), qkva_dtype)
            o_sh = T.alloc_shared((1, block_DV), o_dtype)
            kS = T.alloc_fragment((block_DV,), accum_dtype)
            oo = T.alloc_fragment((block_DV,), accum_dtype)
            vnew = T.alloc_fragment((block_DV,), accum_dtype)
            decay = T.alloc_fragment((1,), accum_dtype)
            bt = T.alloc_fragment((1,), accum_dtype)

            if use_initial_state:
                # h0 is K-major [B,H,DK,DV]; load transposed into S[v, dk] = h0[bb,bh,dk,v0+v]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] = h0[bb, bh, j_k, v0 + j_v]
            else:
                T.clear(S)

            for t in T.serial(L):
                T.copy(q[bb, t : t + 1, bhg, 0:DK], q_s)
                T.copy(k[bb, t : t + 1, bhg, 0:DK], k_s)
                T.copy(v[bb, t : t + 1, bh, v0 : v0 + block_DV], v_s)
                decay[0] = T.exp2(g[bb, t, bh] * 1.442695)  # raw g, exp2
                bt[0] = b[bb, t, bh]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] *= decay[0]
                # kS[v] = sum_dk k[dk] * S[v, dk]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = k_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, kS, dim=1)
                # v_new[v] = beta * (v[v] - kS[v])
                for j_v in T.Parallel(block_DV):
                    vnew[j_v] = bt[0] * (v_s[0, j_v] - kS[j_v])
                # rank-1: S[v, dk] += k[dk] * v_new[v]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] += k_s[0, j_k] * vnew[j_v]
                # o[v] = scale * sum_dk q[dk] * S[v, dk]  (post-update read)
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = q_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, oo, dim=1)
                for j_v in T.Parallel(block_DV):
                    o_sh[0, j_v] = oo[j_v] * scale
                T.copy(o_sh, o[bb, t : t + 1, bh, v0 : v0 + block_DV])

            if store_final_state:
                # ht is K-major [B,H,DK,DV]; store transposed ht[bb,bh,dk,v0+v] = S[v, dk]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    ht[bb, bh, j_k, v0 + j_v] = S[j_v, j_k]

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_fwd_hb(
    H,
    Hg,
    DK,
    DV,
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
    has_seqlens,
    block_DV=64,
    threads=256,
):
    """Head-batched GQA specialization of the gemm-free decode kernel. One CTA owns
    (sequence bb, K/Q head-group hg, V-col-tile bv) and processes ALL grp = H//Hg V-heads
    h = hg*grp + i that share K/Q head hg, loading q/k ONCE (dedup). State is row-stacked
    S[grp*block_DV, DK]: row gv -> head-band i = gv//block_DV, v-row jv = gv%block_DV.
    Layout-inference rule learned on H100: every [M] fragment (S/prod/decay_f/.../oo) MUST be
    accessed over the FULL Parallel(M,...) range -- a partial Parallel(block_DV) write at offset
    i*block_DV makes the fragment's affine map non-invertible (TVM InverseAffineIterMap check
    fails). So ALL per-head divergence is routed through GLOBAL reads/writes indexed by the
    derived head hg*grp + gv//block_DV and channel v0 + gv%block_DV (global needs no fragment
    inversion); only q/k stage through shared (loaded once for the group).
    threads = grp*128 keeps per-thread register footprint identical to the per-head kernel."""
    grp = H // Hg
    M = grp * block_DV
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    q_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    g_shape = (batch_size, num_tokens, H)
    s_shape = (batch_size, H, DK, DV)
    n_vt = (DV + block_DV - 1) // block_DV

    @T.prim_func
    def kernel(
        q: T.Tensor(q_shape, qkva_dtype),
        k: T.Tensor(q_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        g: T.Tensor(g_shape, g_dtype),
        b: T.Tensor(g_shape, b_dtype),
        h0: T.Tensor(s_shape, h0_dtype),
        seqlens: T.Tensor([batch_size], seqlen_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ht: T.Tensor(s_shape, ht_dtype),
    ):
        with T.Kernel(n_vt * batch_size * Hg, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt  # flattened (bb, hg)
            bv = bbhv % n_vt
            bb = bbh // Hg
            hg = bbh % Hg
            v0 = bv * block_DV

            L = T.alloc_var("int32")
            L = seqlens[bb] if has_seqlens else num_tokens

            S = T.alloc_fragment((M, DK), accum_dtype)  # row-stacked state [grp*V-tile, K]
            prod = T.alloc_fragment((M, DK), accum_dtype)
            q_s = T.alloc_shared((1, DK), qkva_dtype)  # shared across the grp heads
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            decay_f = T.alloc_fragment((M,), accum_dtype)  # row-aligned bands (full-M access)
            b_f = T.alloc_fragment((M,), accum_dtype)
            kS = T.alloc_fragment((M,), accum_dtype)
            oo = T.alloc_fragment((M,), accum_dtype)
            vnew = T.alloc_fragment((M,), accum_dtype)

            if use_initial_state:
                # full-M gather; K-major h0[B,H,DK,DV] -> S[gv, K] with derived head/channel
                for gv, j_k in T.Parallel(M, DK):
                    S[gv, j_k] = h0[bb, hg * grp + gv // block_DV, j_k, v0 + gv % block_DV]
            else:
                T.clear(S)

            for t in T.serial(L):
                T.copy(q[bb, t : t + 1, hg, 0:DK], q_s)  # loaded ONCE for the whole group
                T.copy(k[bb, t : t + 1, hg, 0:DK], k_s)
                # per-band decay/beta into FULL-M fragments (global g/beta at derived head). A
                # shared [grp] band read by gv//block_DV does NOT lower in TileLang (the Parallel
                # layout inferencer rejects the shared derived-index read), so we materialize the
                # full M; the redundant exp2 is cheap vs the FMA floor on this memory-bound kernel.
                for gv in T.Parallel(M):
                    decay_f[gv] = T.exp2(g[bb, t, hg * grp + gv // block_DV] * 1.442695)
                    b_f[gv] = b[bb, t, hg * grp + gv // block_DV]
                for gv, j_k in T.Parallel(M, DK):
                    S[gv, j_k] *= decay_f[gv]
                for gv, j_k in T.Parallel(M, DK):
                    prod[gv, j_k] = k_s[0, j_k] * S[gv, j_k]
                T.reduce_sum(prod, kS, dim=1)
                # v read straight from global (derived head/channel); no shared staging
                for gv in T.Parallel(M):
                    vnew[gv] = b_f[gv] * (
                        v[bb, t, hg * grp + gv // block_DV, v0 + gv % block_DV] - kS[gv]
                    )
                for gv, j_k in T.Parallel(M, DK):
                    S[gv, j_k] += k_s[0, j_k] * vnew[gv]
                for gv, j_k in T.Parallel(M, DK):
                    prod[gv, j_k] = q_s[0, j_k] * S[gv, j_k]
                T.reduce_sum(prod, oo, dim=1)
                for gv in T.Parallel(M):  # o written straight to global (derived head/channel)
                    o[bb, t, hg * grp + gv // block_DV, v0 + gv % block_DV] = oo[gv] * scale

            if store_final_state:
                for gv, j_k in T.Parallel(M, DK):
                    ht[bb, hg * grp + gv // block_DV, j_k, v0 + gv % block_DV] = S[gv, j_k]

    return kernel


def _fused_recurrent_gdr_fwd_hb(
    q, k, v, g, beta, scale, initial_state, output_final_state, seqlens, grp
):
    """Head-batched dispatch helper. Grid collapses to n_vt*B*Hg; block_DV keys off the
    POST-collapse supply (B*Hg); threads = grp*128 (cap 512)."""
    B, Tq, Hg, K = k.shape
    _, _, H, V = v.shape

    # block_DV from the head-batched grid (B*Hg CTAs, not B*H): the collapse already cost
    # CTAs, so the V-split must work harder to refill. threads scale with grp to hold the
    # per-thread footprint constant (== the per-head kernel).
    grid_base = B * Hg
    block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32
    threads = min(512, grp * 128)

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    final_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    o = torch.empty_like(v)

    has_seqlens = seqlens is not None
    if seqlens is None:
        seqlens = torch.empty((B,), dtype=torch.int32, device=k.device)

    kern = tilelang_fused_recurrent_gdr_fwd_hb(
        H, Hg, K, V, scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlens.dtype,
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        has_seqlens=has_seqlens,
        block_DV=block_DV,
        threads=threads,
    )
    kern(q, k, v, g, beta, initial_state, seqlens, o, final_state)
    return o, (final_state if output_final_state else None)


def fused_recurrent_gdr_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    seqlens=None,
    head_batch=None,
):
    B, Tq, Hg, K = k.shape
    _, _, H, V = v.shape
    assert K == V == 128 and H % Hg == 0
    scale = scale or K ** -0.5

    # Head-batched GQA: one CTA processes all grp = H//Hg V-heads sharing a K/Q head, loading
    # q/k once. Auto OFF here (the host-gated decode path saves only the q/k load ~ sub-1%,
    # below noise); exposed as a forceable flag for benchmarking/tests. Restricted to grp in
    # {2,4} (threads <= 512) -- grp>4 risks register spills / 1024-thread occupancy loss.
    grp = H // Hg
    if head_batch is None:
        head_batch = False
    if head_batch:
        assert grp in (2, 4), f"head_batch supports grp=H//Hg in {{2,4}}, got grp={grp}"
        return _fused_recurrent_gdr_fwd_hb(
            q, k, v, g, beta, scale, initial_state, output_final_state, seqlens, grp
        )

    # Tile selection. When writing the final state (`output_final_state`, the decode default), the
    # K-major transposed store `ht[bb,bh,jk,v0+jv] = S[jv,jk]` is the dominant cost: at block_DV<128
    # it is catastrophically uncoalesced (~0.4 TB/s), but at block_DV=128 (full-V tile, n_vt=1) the
    # transpose coalesces -> measured ~2x faster at EVERY batch size (1.35x @ B=1 .. 3.0x @ B=8-16
    # .. 2.0x @ B=256; benchmark/probe_h2_blockdv_crossover.py, final_state bit-identical). So force
    # block_DV=128 whenever we store the final state. (This inverts the verify kernel's V-major
    # choice of 64 -- that store needs no transpose, so there occupancy wins; here the write does.)
    # No final-state write: keep the occupancy ladder (64/32) -- block_DV is then perf-neutral.
    grid_base = B * H
    if output_final_state:
        block_DV = 128
    else:
        block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    final_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    o = torch.empty_like(v)

    has_seqlens = seqlens is not None
    if seqlens is None:
        seqlens = torch.empty((B,), dtype=torch.int32, device=k.device)
    seqlen_dtype = seqlens.dtype

    kern = tilelang_fused_recurrent_gdr_fwd(
        H,
        Hg,
        K,
        V,
        scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        has_seqlens=has_seqlens,
        block_DV=block_DV,
        threads=128,
    )
    kern(q, k, v, g, beta, initial_state, seqlens, o, final_state)
    return o, (final_state if output_final_state else None)
