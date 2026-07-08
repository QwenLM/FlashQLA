# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""SGLang verify kernel: gemm-free GDN recurrence + paged V-major (bf16) state pool,
per-token intermediate states, no-commit, varlen cu_seqlens. Host-side gating (g/beta
pre-activated, q/k pre-l2normed by the wrapper). CUDA-graph safe (no host sync / no alloc
in the captured entry; all buffers caller-provided)."""
import os

import torch
import tilelang
import tilelang.language as T

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_verify(
    H,
    Hg,
    DK,
    DV,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    pool_dtype,
    o_dtype,
    seqlen_dtype,
    idx_dtype,
    store_intermediate,
    disable_state_update,
    block_DV=128,
    threads=128,
):
    total_tokens = T.dynamic("total_tokens")
    N = T.dynamic("N")  # number of requests
    num_slots = T.dynamic("num_slots")
    num_cache_slots = T.dynamic("num_cache_slots")
    cache_steps = T.dynamic("cache_steps")
    q_shape = (1, total_tokens, Hg, DK)
    v_shape = (1, total_tokens, H, DV)
    g_shape = (1, total_tokens, H)
    pool_shape = (num_slots, H, DV, DK)  # V-major [., H, V, K]
    ibuf_shape = (num_cache_slots, cache_steps, H, DV, DK)  # V-major
    n_vt = (DV + block_DV - 1) // block_DV

    @T.prim_func
    def kernel(
        q: T.Tensor(q_shape, qkva_dtype),
        k: T.Tensor(q_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        g: T.Tensor(g_shape, g_dtype),
        b: T.Tensor(g_shape, b_dtype),
        pool: T.Tensor(pool_shape, pool_dtype),
        state_indices: T.Tensor([N], idx_dtype),
        cu_seqlens: T.Tensor([N + 1], seqlen_dtype),
        intermediate_state_indices: T.Tensor([N], idx_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ibuf: T.Tensor(ibuf_shape, pool_dtype),
    ):
        with T.Kernel(n_vt * N * H, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt
            bv = bbhv % n_vt
            bb = bbh // H  # request index
            bh = bbh % H
            bhg = bh // (H // Hg)
            v0 = bv * block_DV

            slot = T.alloc_var("int32")
            cslot = T.alloc_var("int32")
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            slot = state_indices[bb]
            cslot = intermediate_state_indices[bb]
            seq_start = cu_seqlens[bb]
            seq_end = cu_seqlens[bb + 1]

            S = T.alloc_fragment((block_DV, DK), accum_dtype)  # state [V-tile, K] fp32
            prod = T.alloc_fragment((block_DV, DK), accum_dtype)
            q_s = T.alloc_shared((1, DK), qkva_dtype)
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            v_s = T.alloc_shared((1, block_DV), qkva_dtype)
            o_sh = T.alloc_shared((1, block_DV), o_dtype)
            kS = T.alloc_fragment((block_DV,), accum_dtype)
            oo = T.alloc_fragment((block_DV,), accum_dtype)
            vnew = T.alloc_fragment((block_DV,), accum_dtype)
            decay = T.alloc_fragment((1,), accum_dtype)
            bt = T.alloc_fragment((1,), accum_dtype)

            # gather V-major pool[slot, bh, v0:v0+block_DV, :] directly into S[v, dk]
            T.clear(S)
            with T.If(slot >= 0):
                with T.Then():
                    for j_v, j_k in T.Parallel(block_DV, DK):
                        S[j_v, j_k] = pool[slot, bh, v0 + j_v, j_k]

            for t in T.serial(seq_end - seq_start):
                tt = seq_start + t  # absolute token position in the flattened layout
                T.copy(q[0, tt : tt + 1, bhg, 0:DK], q_s)
                T.copy(k[0, tt : tt + 1, bhg, 0:DK], k_s)
                T.copy(v[0, tt : tt + 1, bh, v0 : v0 + block_DV], v_s)
                decay[0] = T.exp2(g[0, tt, bh] * 1.442695)
                bt[0] = b[0, tt, bh]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] *= decay[0]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = k_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, kS, dim=1)
                for j_v in T.Parallel(block_DV):
                    vnew[j_v] = bt[0] * (v_s[0, j_v] - kS[j_v])
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] += k_s[0, j_k] * vnew[j_v]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = q_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, oo, dim=1)
                for j_v in T.Parallel(block_DV):
                    o_sh[0, j_v] = oo[j_v] * scale
                T.copy(o_sh, o[0, tt : tt + 1, bh, v0 : v0 + block_DV])
                # per-token intermediate (V-major), gated by the POOL slot mask
                if store_intermediate:
                    with T.If(slot >= 0):
                        with T.Then():
                            for j_v, j_k in T.Parallel(block_DV, DK):
                                ibuf[cslot, t, bh, v0 + j_v, j_k] = S[j_v, j_k]

            # commit final state to the pool unless no-commit (verify)
            if not disable_state_update:
                with T.If(slot >= 0):
                    with T.Then():
                        for j_v, j_k in T.Parallel(block_DV, DK):
                            pool[slot, bh, v0 + j_v, j_k] = S[j_v, j_k]

    return kernel


def fused_recurrent_gdr_verify_fwd(
    q,
    k,
    v,
    g,
    beta,
    pool,
    state_indices,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    o,
    scale=None,
    disable_state_update=True,
):
    """Graph-safe low-level verify entry. ALL buffers caller-preallocated (o, pool, ibuf);
    no host sync, no allocation. g/beta pre-activated and q/k pre-l2normed host-side."""
    _, total_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    N = state_indices.shape[0]
    assert K == V == 128 and H % Hg == 0
    scale = scale or K ** -0.5
    store_intermediate = intermediate_states_buffer is not None

    # block_DV=64 (2 V-tiles) @ threads=128 is the bandwidth sweet spot (autotuned, H100);
    # 32 (4 V-tiles) for the low-CTA tail. block_DV=128 is occupancy-starved -> never used.
    grid_base = N * H
    block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32

    kern = tilelang_fused_recurrent_gdr_verify(
        H,
        Hg,
        K,
        V,
        scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        pool_dtype=pool.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        idx_dtype=state_indices.dtype,
        store_intermediate=store_intermediate,
        disable_state_update=disable_state_update,
        block_DV=block_DV,
        threads=128,
    )
    kern(q, k, v, g, beta, pool, state_indices, cu_seqlens,
         intermediate_state_indices, o, intermediate_states_buffer)
    return o


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_verify_gated(
    H, Hg, DK, DV, scale, accum_dtype, qkva_dtype, ab_dtype, gate_dtype, pool_dtype,
    o_dtype, seqlen_dtype, idx_dtype, store_intermediate, disable_state_update,
    l2norm_eps=1e-6, softplus_thr=20.0, allow_neg_eigval=False, block_DV=128, threads=128,
):
    """In-kernel fused-gating verify kernel (req #5): takes raw a,b,A_log,dt_bias and
    computes g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b), and qk-l2norm in-kernel."""
    total_tokens = T.dynamic("total_tokens")
    N = T.dynamic("N")
    num_slots = T.dynamic("num_slots")
    num_cache_slots = T.dynamic("num_cache_slots")
    cache_steps = T.dynamic("cache_steps")
    qk_shape = (1, total_tokens, Hg, DK)
    v_shape = (1, total_tokens, H, DV)
    ab_shape = (1, total_tokens, H)
    pool_shape = (num_slots, H, DV, DK)
    ibuf_shape = (num_cache_slots, cache_steps, H, DV, DK)
    n_vt = (DV + block_DV - 1) // block_DV
    beta_mul = 2.0 if allow_neg_eigval else 1.0

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, qkva_dtype),
        k: T.Tensor(qk_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        a: T.Tensor(ab_shape, ab_dtype),
        b: T.Tensor(ab_shape, ab_dtype),
        A_log: T.Tensor([H], gate_dtype),
        dt_bias: T.Tensor([H], gate_dtype),
        pool: T.Tensor(pool_shape, pool_dtype),
        state_indices: T.Tensor([N], idx_dtype),
        cu_seqlens: T.Tensor([N + 1], seqlen_dtype),
        intermediate_state_indices: T.Tensor([N], idx_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ibuf: T.Tensor(ibuf_shape, pool_dtype),
    ):
        with T.Kernel(n_vt * N * H, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt
            bv = bbhv % n_vt
            bb = bbh // H
            bh = bbh % H
            bhg = bh // (H // Hg)
            v0 = bv * block_DV

            slot = T.alloc_var("int32")
            cslot = T.alloc_var("int32")
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            slot = state_indices[bb]
            cslot = intermediate_state_indices[bb]
            seq_start = cu_seqlens[bb]
            seq_end = cu_seqlens[bb + 1]
            a_log_h = T.alloc_var(accum_dtype)
            dt_b_h = T.alloc_var(accum_dtype)
            a_log_h = A_log[bh]
            dt_b_h = dt_bias[bh]

            S = T.alloc_fragment((block_DV, DK), accum_dtype)
            prod = T.alloc_fragment((block_DV, DK), accum_dtype)
            q_s = T.alloc_shared((1, DK), qkva_dtype)
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            q_n = T.alloc_shared((1, DK), qkva_dtype)
            k_n = T.alloc_shared((1, DK), qkva_dtype)
            v_s = T.alloc_shared((1, block_DV), qkva_dtype)
            o_sh = T.alloc_shared((1, block_DV), o_dtype)
            qsq = T.alloc_fragment((1, DK), accum_dtype)
            ksq = T.alloc_fragment((1, DK), accum_dtype)
            ssq = T.alloc_fragment((1,), accum_dtype)
            kS = T.alloc_fragment((block_DV,), accum_dtype)
            oo = T.alloc_fragment((block_DV,), accum_dtype)
            vnew = T.alloc_fragment((block_DV,), accum_dtype)
            decay = T.alloc_fragment((1,), accum_dtype)
            bt = T.alloc_fragment((1,), accum_dtype)

            T.clear(S)
            with T.If(slot >= 0):
                with T.Then():
                    for j_v, j_k in T.Parallel(block_DV, DK):
                        S[j_v, j_k] = pool[slot, bh, v0 + j_v, j_k]

            for t in T.serial(seq_end - seq_start):
                tt = seq_start + t
                T.copy(q[0, tt : tt + 1, bhg, 0:DK], q_s)
                T.copy(k[0, tt : tt + 1, bhg, 0:DK], k_s)
                T.copy(v[0, tt : tt + 1, bh, v0 : v0 + block_DV], v_s)
                # in-kernel qk l2norm: x_n = x / sqrt(sum(x^2) + eps)
                for _i, j_k in T.Parallel(1, DK):
                    qsq[0, j_k] = q_s[0, j_k] * q_s[0, j_k]
                T.reduce_sum(qsq, ssq, dim=1)
                for _i, j_k in T.Parallel(1, DK):
                    q_n[0, j_k] = q_s[0, j_k] * T.rsqrt(ssq[0] + l2norm_eps)
                for _i, j_k in T.Parallel(1, DK):
                    ksq[0, j_k] = k_s[0, j_k] * k_s[0, j_k]
                T.reduce_sum(ksq, ssq, dim=1)
                for _i, j_k in T.Parallel(1, DK):
                    k_n[0, j_k] = k_s[0, j_k] * T.rsqrt(ssq[0] + l2norm_eps)
                # in-kernel gating: g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b)
                x = a[0, tt, bh] + dt_b_h
                sp = T.if_then_else(x > softplus_thr, x, T.log(1.0 + T.exp(x)))
                decay[0] = T.exp2((-T.exp(a_log_h) * sp) * 1.442695)
                bt[0] = beta_mul * T.sigmoid(b[0, tt, bh])
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] *= decay[0]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = k_n[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, kS, dim=1)
                for j_v in T.Parallel(block_DV):
                    vnew[j_v] = bt[0] * (v_s[0, j_v] - kS[j_v])
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] += k_n[0, j_k] * vnew[j_v]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = q_n[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, oo, dim=1)
                for j_v in T.Parallel(block_DV):
                    o_sh[0, j_v] = oo[j_v] * scale
                T.copy(o_sh, o[0, tt : tt + 1, bh, v0 : v0 + block_DV])
                if store_intermediate:
                    with T.If(slot >= 0):
                        with T.Then():
                            for j_v, j_k in T.Parallel(block_DV, DK):
                                ibuf[cslot, t, bh, v0 + j_v, j_k] = S[j_v, j_k]

            if not disable_state_update:
                with T.If(slot >= 0):
                    with T.Then():
                        for j_v, j_k in T.Parallel(block_DV, DK):
                            pool[slot, bh, v0 + j_v, j_k] = S[j_v, j_k]

    return kernel


def fused_recurrent_gdr_verify_gated_fwd(
    q, k, v, a, b, A_log, dt_bias, pool, state_indices, cu_seqlens,
    intermediate_states_buffer, intermediate_state_indices, o,
    scale=None, disable_state_update=True, allow_neg_eigval=False,
):
    """In-kernel fused-gating verify entry: raw a,b,A_log,dt_bias; computes g/beta + qk-l2norm
    inside the kernel. Graph-safe (all buffers caller-provided)."""
    _, total_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    N = state_indices.shape[0]
    assert K == V == 128 and H % Hg == 0
    scale = scale or K ** -0.5
    store_intermediate = intermediate_states_buffer is not None

    grid_base = N * H  # bandwidth sweet spot (autotuned, H100): block_DV=64 @ threads=128
    block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32

    kern = tilelang_fused_recurrent_gdr_verify_gated(
        H, Hg, K, V, scale,
        accum_dtype="float32", qkva_dtype=q.dtype, ab_dtype=a.dtype, gate_dtype=A_log.dtype,
        pool_dtype=pool.dtype, o_dtype=o.dtype, seqlen_dtype=cu_seqlens.dtype,
        idx_dtype=state_indices.dtype, store_intermediate=store_intermediate,
        disable_state_update=disable_state_update, allow_neg_eigval=allow_neg_eigval,
        block_DV=block_DV, threads=max(128, block_DV * 2),
    )
    kern(q, k, v, a, b, A_log, dt_bias, pool, state_indices, cu_seqlens,
         intermediate_state_indices, o, intermediate_states_buffer)
    return o


# ----------------------------------------------------------------------------------------------
# H1: gating + qk-l2norm DEDUP PRE-PASS.
# The in-kernel-gated kernel above recomputes g/beta + qk-l2norm INSIDE the per-token hot loop,
# once per (token, V-head, V-tile) CTA -> l2norm redundant grp*n_vt times per (token,K-head),
# gating redundant n_vt times per (token,V-head). This pre-pass computes each ONCE (grid =
# total_tokens*Hk, one CTA per (token, K-head)) and feeds the HOST-gated main kernel
# (fused_recurrent_gdr_verify_fwd) whose hot loop then has NO transcendentals. Measured ceiling
# (gated-vs-host-gated, H100): 14-16% at T=12 large batch. Regime-gated (see should_use_prepass):
# at single-request the second-launch tax exceeds the small ceiling, so variant A is kept there.
# ----------------------------------------------------------------------------------------------

# Regime gate tunables. CALIBRATED FOR THE CUDA-GRAPH (production) PATH (H100,
# benchmark/bench_prepass.py bench_graph, _time_graph with 50+ warmup). Prepass wins when (a) drafts
# are long enough that the per-token gating/l2norm recompute it dedups is a meaningful fraction
# (T_avg >= 4; at T=1 it is paid once -> nothing to dedup across tokens, measured neutral/loss even
# under graphs), AND (b) the main-kernel work N*H*(1+T_avg) clears a SMALL floor.
#
# KEY: under CUDA-graph REPLAY the prepass's 2nd launch shrinks to a graph node, so the eager
# second-launch tax (~13-15us) VANISHES and the dedup win dominates down to SINGLE-REQUEST. Measured
# (graph, Hk=16 Hv=32): N=1,T=12 work=416 -> 1.18x WIN (eager was 0.60x LOSS); N=2,T=12 -> 1.08x;
# N=4,T=12 -> 1.18x; every T=12 point N>=1 wins 1.08-1.24x. The only graph losses are work<=320
# (N=1,T=4=160 0.96x; N=2,T=4=320 0.93x). So the gate window for the verify (always T=12) is
# work in (320, 416]; MIN_WORK=384 fires the prepass for the ENTIRE T=12 verify path incl. N=1,
# capturing the 1.08-1.24x the old eager-conservative 3000 was leaving on the table at small batch.
# TRADEOFF: this assumes the CUDA-graph deployment (qwen36 captures all batch sizes). Under EAGER
# (warmup / --disable-cuda-graph), small-N now pays the 2nd-launch tax (N=1,T=12 eager 0.60x) -- but
# steady-state production verify is always captured, and warmup is untimed. Metric scales with H, so
# small-head configs self-raise the batch needed (gate stays safe for untested H<32).
PREPASS_MIN_T = 4.0
# graph-calibrated default 384 (was 3000, eager-conservative); floor below N=1/T=12 work=416.
# Env-overridable for controlled A/B (set FLASHQLA_PREPASS_MIN_WORK=3000 to reproduce the old gate).
PREPASS_MIN_WORK = int(os.environ.get("FLASHQLA_PREPASS_MIN_WORK", "384"))


def should_use_prepass(N, H, total_tokens):
    """Static (capture-safe; shape-only) decision: run the dedup prepass + host-gated main kernel
    (True) vs the single in-kernel-gated kernel A (False). Both produce identical output; this
    only picks the faster path per regime. CALIBRATED FOR THE CUDA-GRAPH (production) path, where
    the prepass's 2nd launch is a free graph node and it wins down to single-request T>=4 (the eager
    second-launch tax that favored variant A at small batch does NOT exist under graph replay)."""
    if N <= 0:
        return False
    t_avg = total_tokens / N
    work = H * (N + total_tokens)  # == N*H*(1 + t_avg), proportional to the main-kernel runtime
    return (t_avg >= PREPASS_MIN_T) and (work >= PREPASS_MIN_WORK)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_gdr_verify_prepass(
    Hk, Hv, DK, accum_dtype, qk_dtype, ab_dtype, gate_dtype, g_out_dtype, b_out_dtype,
    l2norm_eps=1e-6, softplus_thr=20.0, allow_neg_eigval=False, threads=128,
):
    """Dedup pre-pass: ONE CTA per token tt computes q_n/k_n = l2norm(q/k) for all Hk K-heads at
    once via a parallel [Hk,DK] reduce (the proven reduce_sum(...,dim=1) idiom), and
    g = -exp(A_log[h])*softplus(a+dt_bias) (RAW log-decay; the main kernel applies exp2) +
    beta = beta_mul*sigmoid(b) for all Hv V-heads. No recurrence, no cu_seqlens (token-local):
    the main kernel only consumes tokens within cu_seqlens, so normalizing trailing padding is
    harmless.

    q/k are read into a fragment and q_n/k_n written DIRECTLY from a full-range T.Parallel (the
    head-batch global-write idiom) -- no per-row [1,DK] T.copy, so no serial Hk loop (that was a
    measured ~6x slowdown) and no Hopper copy-layout trap. The gate write is the full contiguous
    [1,Hv] row (Hv>=4 -> >=128-bit fp32 extent; a per-K-head [1,grp] write fails layout inference
    for grp<4). Grid = total_tokens balances occupancy (~total_tokens CTAs)."""
    beta_mul = 2.0 if allow_neg_eigval else 1.0
    total_tokens = T.dynamic("total_tokens")
    qk_shape = (1, total_tokens, Hk, DK)
    ab_shape = (1, total_tokens, Hv)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, qk_dtype),
        k: T.Tensor(qk_shape, qk_dtype),
        a: T.Tensor(ab_shape, ab_dtype),
        b: T.Tensor(ab_shape, ab_dtype),
        A_log: T.Tensor([Hv], gate_dtype),
        dt_bias: T.Tensor([Hv], gate_dtype),
        q_n: T.Tensor(qk_shape, qk_dtype),
        k_n: T.Tensor(qk_shape, qk_dtype),
        g_out: T.Tensor(ab_shape, g_out_dtype),
        beta_out: T.Tensor(ab_shape, b_out_dtype),
    ):
        with T.Kernel(total_tokens, threads=threads) as (tt,):
            xf = T.alloc_fragment((Hk, DK), accum_dtype)   # raw q/k rows (reused q then k)
            sq = T.alloc_fragment((Hk, DK), accum_dtype)   # squared, reduce input
            ssq = T.alloc_fragment((Hk,), accum_dtype)     # per-K-head sum-of-squares
            g_f = T.alloc_fragment((1, Hv), accum_dtype)
            b_f = T.alloc_fragment((1, Hv), accum_dtype)
            g_sh = T.alloc_shared((1, Hv), g_out_dtype)
            b_sh = T.alloc_shared((1, Hv), b_out_dtype)

            # q l2norm: load [Hk,DK] -> square -> reduce over DK -> normalize + write (all parallel)
            for i, j in T.Parallel(Hk, DK):
                xf[i, j] = q[0, tt, i, j]
            for i, j in T.Parallel(Hk, DK):
                sq[i, j] = xf[i, j] * xf[i, j]
            T.reduce_sum(sq, ssq, dim=1)
            for i, j in T.Parallel(Hk, DK):
                q_n[0, tt, i, j] = xf[i, j] * T.rsqrt(ssq[i] + l2norm_eps)

            # k l2norm (reuse xf/sq/ssq)
            for i, j in T.Parallel(Hk, DK):
                xf[i, j] = k[0, tt, i, j]
            for i, j in T.Parallel(Hk, DK):
                sq[i, j] = xf[i, j] * xf[i, j]
            T.reduce_sum(sq, ssq, dim=1)
            for i, j in T.Parallel(Hk, DK):
                k_n[0, tt, i, j] = xf[i, j] * T.rsqrt(ssq[i] + l2norm_eps)

            # gating for all Hv V-heads (fragment idiom, inlined exprs, direct global reads),
            # staged through [1,Hv] shared and written as one contiguous row (>=128-bit extent)
            for _i, h in T.Parallel(1, Hv):
                g_f[0, h] = -T.exp(A_log[h]) * T.if_then_else(
                    a[0, tt, h] + dt_bias[h] > softplus_thr,
                    a[0, tt, h] + dt_bias[h],
                    T.log(1.0 + T.exp(a[0, tt, h] + dt_bias[h])),
                )
                b_f[0, h] = beta_mul * T.sigmoid(b[0, tt, h])
            for _i, h in T.Parallel(1, Hv):
                g_sh[0, h] = g_f[0, h]
                b_sh[0, h] = b_f[0, h]
            T.copy(g_sh, g_out[0, tt : tt + 1, 0:Hv])
            T.copy(b_sh, beta_out[0, tt : tt + 1, 0:Hv])

    return kernel


def fused_recurrent_gdr_verify_prepass(
    q, k, a, b, A_log, dt_bias,
    q_n=None, k_n=None, g_out=None, beta_out=None, allow_neg_eigval=False,
):
    """Dedup pre-pass dispatch. Computes q_n=l2norm(q), k_n=l2norm(k), g=raw log-decay,
    beta=beta_mul*sigmoid(b) ONCE, feeding the host-gated verify main kernel. Buffers are
    caller-provided for CUDA-graph capture safety (like o/pool/ibuf); allocate-if-None is an
    EAGER convenience for tests/bench only -- it must NOT run inside a captured region (no alloc
    in capture). g/beta are fp32 (matching gdn_sigmoid_gate); q_n/k_n match q/k dtype."""
    _, total_tokens, Hk, K = q.shape
    Hv = a.shape[2]
    assert Hv % Hk == 0, f"num_v_heads {Hv} must be divisible by num_k_heads {Hk}"
    assert total_tokens > 0
    if q_n is None:
        q_n = torch.empty_like(q)
    if k_n is None:
        k_n = torch.empty_like(k)
    if g_out is None:
        g_out = torch.empty(1, total_tokens, Hv, device=q.device, dtype=torch.float32)
    if beta_out is None:
        beta_out = torch.empty(1, total_tokens, Hv, device=q.device, dtype=torch.float32)

    kern = tilelang_gdr_verify_prepass(
        Hk, Hv, K,
        accum_dtype="float32", qk_dtype=q.dtype, ab_dtype=a.dtype, gate_dtype=A_log.dtype,
        g_out_dtype=g_out.dtype, b_out_dtype=beta_out.dtype,
        allow_neg_eigval=allow_neg_eigval, threads=128,
    )
    kern(q, k, a, b, A_log, dt_bias, q_n, k_n, g_out, beta_out)
    return q_n, k_n, g_out, beta_out


# Module-level UNBOUNDED, never-evicting scratch cache for the prepass outputs. Unbounded is the
# capture-safe choice: each captured graph bakes the device pointers of its own (total_tokens,...)
# entry; an evicting LRU (e.g. utils.tensor_cache) could free a still-referenced storage and make
# a later replay read freed memory. One live entry per captured (N,T); entries are never freed.
_PREPASS_SCRATCH = {}


def get_prepass_scratch(total_tokens, Hk, Hv, device, qk_dtype):
    """Persistent prepass scratch (q_n,k_n bf16 [1,T,Hk,128]; g,beta fp32 [1,T,Hv]). Allocated
    lazily on the first (warmup) call and reused thereafter so addresses are stable across
    CUDA-graph capture/replay. Raises (rather than allocating + aborting capture) on a cold miss
    during capture -- warmup must exercise every (N,T) that will be captured."""
    key = (total_tokens, Hk, Hv, device, qk_dtype)
    buf = _PREPASS_SCRATCH.get(key)
    if buf is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"prepass scratch not warmed for shape {key}; run an eager warmup pass before "
                "CUDA-graph capture (the scratch must be allocated outside the captured region)."
            )
        q_n = torch.empty(1, total_tokens, Hk, 128, device=device, dtype=qk_dtype)
        k_n = torch.empty(1, total_tokens, Hk, 128, device=device, dtype=qk_dtype)
        g_out = torch.empty(1, total_tokens, Hv, device=device, dtype=torch.float32)
        beta_out = torch.empty(1, total_tokens, Hv, device=device, dtype=torch.float32)
        buf = (q_n, k_n, g_out, beta_out)
        _PREPASS_SCRATCH[key] = buf
    return buf
