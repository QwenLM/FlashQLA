# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

# Built-in kernel warmup (ahead-of-time JIT precompilation).
from __future__ import annotations

import math
import warnings

import torch


def _idx(x: list[int], device, dtype=torch.int64) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def _io(nseq, S, H, Hg, K, V, chunk_size, dtype, gate_dtype, beta_dtype, device):
    T = nseq * S
    q = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
    k = torch.randn(1, T, Hg, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)
    a = torch.randn(1, T, H, chunk_size, dtype=dtype, device=device)
    g = torch.randn(1, T, H, dtype=gate_dtype, device=device)
    b = torch.randn(1, T, H, dtype=beta_dtype, device=device)
    do = torch.randn(1, T, H, V, dtype=dtype, device=device)
    return q, k, v, a, g, b, do


def _h0(nseq, H, K, V, state_v_first, device):
    return torch.randn(
        (nseq, H, V, K) if state_v_first else (nseq, H, K, V),
        dtype=torch.float32, device=device,
    )


def _cp_meta(P, S, H, device, idtype):
    cu = _idx([i * S for i in range(P + 1)], device, idtype)  # [P+1] partition bounds
    cp_seq_map = _idx([0] * P, device, idtype)                # [P] every partition -> raw seq 0
    raw_cu = _idx([0, P * S], device, idtype)                 # [2] single raw sequence
    seq_map_r2c = _idx([0, P], device, idtype)                # [raw_batch+1]
    ht_mask = torch.zeros(P, dtype=torch.bool, device=device)
    ht_mask[-1] = True
    ht_mask_bwd = torch.zeros(P, dtype=torch.bool, device=device)
    ht_mask_bwd[0] = True
    return cu, cp_seq_map, raw_cu, seq_map_r2c, ht_mask, ht_mask_bwd


# ---------------------------------------------------------------------------
# non-CP path
# ---------------------------------------------------------------------------
def _layout(cfg, rb, S, is_varlen):
    """Build dummy inputs for ``rb`` sequences/partitions of ``S`` tokens, in the
    requested layout. Returns (q, k, v, a, g, b, do, cu, h0)."""
    H, Hg, K, V, cs = cfg.H, cfg.Hg, cfg.K, cfg.V, cfg.chunk_size
    q, k, v, a, g, b, do = _io(rb, S, H, Hg, K, V, cs, cfg.dtype,
                               cfg.gate_dtype, cfg.beta_dtype, cfg.device)
    h0 = _h0(rb, H, K, V, cfg.state_v_first, cfg.device) if cfg.use_initial_state else None
    if is_varlen:
        cu = _idx([i * S for i in range(rb + 1)], cfg.device, cfg.cu_seqlens_dtype)
    else:
        # dense [rb, S, ...]; cu=None makes the dispatcher build an int32 cu.
        cu = None
        q = q.view(rb, S, Hg, K); k = k.view(rb, S, Hg, K); v = v.view(rb, S, H, V)
        a = a.view(rb, S, H, cs); g = g.view(rb, S, H); b = b.view(rb, S, H)
        do = do.view(rb, S, H, V)
    return q, k, v, a, g, b, do, cu, h0


def _warm_non_cp(cfg):
    from flash_qla.ops.gated_delta_rule.chunk import (
        kkt_solve, fused_gdr_fwd, fused_gdr_h, fused_gdr_bwd,
    )
    cs, svf, vl = cfg.chunk_size, cfg.state_v_first, cfg.is_varlen

    for S in cfg.S_list:
        for rb in cfg.rb_sweep:
            q, k, v, a, g, b, _, cu, h0 = _layout(cfg, rb, S, vl)
            fused_gdr_fwd(
                q=q, k=k, v=v, a=a, g=g, b=b, scale=cfg.scale, initial_state=h0,
                output_final_state=cfg.output_final_state, output_o=True,
                cu_seqlens=cu, state_v_first=svf,
            )

        # Everything else here is independent of block_DV -> run once at rb_rep.
        q, k, v, a, g, b, do, cu, h0 = _layout(cfg, cfg.rb_rep, S, vl)
        kkt_solve(k=k, b=b, chunk_size=cs, cu_seqlens=cu)
        if cfg.training:
            h, _, _ = _h_for_bwd(fused_gdr_h, k, v, a, g, b, h0, cu, svf)
            _warm_bwd_main(fused_gdr_bwd, q, k, v, a, g, b, do, h, cu, cs, svf, cfg)


def _h_for_bwd(fused_gdr_h, k, v, a, g, b, h0, cu, svf):
    # main-backward prepare_h: store_h=True, store_final_state=False, is_cp=False.
    return fused_gdr_h(
        k=k, v=v, a=a, g=g, b=b, initial_state=h0,
        output_final_state=False, output_h=True, cu_seqlens=cu,
        num_warmup_chunks=None, state_v_first=svf,
    )


def _warm_bwd_main(fused_gdr_bwd, q, k, v, a, g, b, do, h, cu, cs, svf, cfg):
    # fused_gdr_bwd keys on (is_varlen, use_dht, state_v_first). Warm use_dht both.
    rb = (len(cu) - 1) if cu is not None else k.shape[0]
    H, K, V = v.shape[-2], k.shape[-1], v.shape[-1]
    dht = torch.randn(
        (rb, H, V, K) if svf else (rb, H, K, V), dtype=torch.float32, device=k.device
    )
    for d in (dht, None):
        fused_gdr_bwd(
            q=q, k=k, v=v, a=a, g=g, b=b, do=do, dht=d, h=h, scale=cfg.scale,
            cu_seqlens=cu, chunk_size=cs, state_v_first=svf,
        )


# ---------------------------------------------------------------------------
# CP path (intra / inter / combined). is_cp=True main kernel comes from intra only;
# inter contributes preprocessing-kernel variants.
# ---------------------------------------------------------------------------
def _warm_cp(cfg):
    from flash_qla.ops.gated_delta_rule.chunk import (
        fused_gdr_fwd, fused_gdr_h, fused_gdr_bwd, fused_gdr_dh, correct_initial_states,
        correct_terminal_states, get_warmup_chunks, get_warmup_chunks_bidi,
    )
    try:
        from flash_qla.ops.gated_delta_rule.chunk import aggregate_card_state
    except ImportError:
        aggregate_card_state = None

    H, K, V = cfg.H, cfg.K, cfg.V
    cs, dtype, device, svf = cfg.chunk_size, cfg.dtype, cfg.device, cfg.state_v_first
    combined = cfg.enable_intra and cfg.enable_inter and aggregate_card_state is not None

    def io(P, S):
        return _io(P, S, H, cfg.Hg, K, V, cs, dtype, cfg.gate_dtype, cfg.beta_dtype, device)

    # (1) Preprocessing / correction / dh / main-backward: all block_DV-independent,
    #     so run once per S (for has_incomplete_tile) at a 2-partition representative.
    P = 2
    for S in cfg.S_list:
        q, k, v, a, g, b, do = io(P, S)
        cu, cp_seq_map, raw_cu, seq_map_r2c, ht_mask, ht_mask_bwd = _cp_meta(P, S, H, device, cfg.cu_seqlens_dtype)
        raw_h0 = _h0(1, H, K, V, svf, device) if cfg.use_initial_state else None

        # --- warmup-chunk kernels ---
        num_warmup, fallback = get_warmup_chunks(
            g=g, cu_seqlens=cu, ht_mask=ht_mask, chunk_size=cs,
        )
        # bidi: force_inter_boundaries False (intra) and/or True (inter).
        for fib in ({False} if cfg.enable_intra else set()) | ({True} if cfg.enable_inter else set()):
            get_warmup_chunks_bidi(
                g=g, cu_seqlens=cu, ht_mask_fwd=ht_mask, ht_mask_bwd=ht_mask_bwd,
                chunk_size=cs, seq_map_r2c=seq_map_r2c, force_inter_boundaries=fib,
            )

        # --- prepare_h (is_cp=True): intra uses initial_state=None (uis=F);
        #     inter uses the caller's initial_state (uis per config). ---
        uis_variants = set()
        if cfg.enable_intra:
            uis_variants.add(False)
        if cfg.enable_inter and cfg.use_initial_state:
            uis_variants.add(True)
        ht = mt = None
        for uis in uis_variants:
            h0_in = raw_h0.expand(P, *raw_h0.shape[1:]).contiguous() if uis else None
            _, ht_i, mt_i = fused_gdr_h(
                k=k, v=v, a=a, g=g, b=b, initial_state=h0_in,
                output_final_state=True, output_h=False, cu_seqlens=cu,
                num_warmup_chunks=num_warmup, state_v_first=svf,
            )
            if ht is None:
                ht, mt = ht_i, mt_i
        if ht is None:  # inter-only without initial_state still needs ht/mt below
            _, ht, mt = fused_gdr_h(
                k=k, v=v, a=a, g=g, b=b, initial_state=None,
                output_final_state=True, output_h=False, cu_seqlens=cu,
                num_warmup_chunks=num_warmup, state_v_first=svf,
            )

        # --- correction (fwd): use_raw_h0 False always; True when h0 present. ---
        cp_h0 = correct_initial_states(
            raw_h0=None, ht_buffer=ht, mt_buffer=mt, fallback_mask=fallback,
            seq_map_r2c=seq_map_r2c, state_v_first=svf,
        )
        if cfg.use_initial_state:
            cp_h0 = correct_initial_states(
                raw_h0=raw_h0, ht_buffer=ht, mt_buffer=mt, fallback_mask=fallback,
                seq_map_r2c=seq_map_r2c, state_v_first=svf,
            )
        if combined:
            aggregate_card_state(
                ht, mt, fallback, seq_map_r2c, raw_h0=None,
                state_v_first=svf, compute_m=True,
            )
            if cfg.use_initial_state:
                aggregate_card_state(
                    ht, mt, fallback, seq_map_r2c, raw_h0=raw_h0,
                    state_v_first=svf, compute_m=True,
                )

        if cfg.training and cfg.enable_intra:
            h_main, _, _ = fused_gdr_h(
                k=k, v=v, a=a, g=g, b=b, initial_state=cp_h0,
                output_final_state=False, output_h=True, cu_seqlens=cu,
                num_warmup_chunks=None, state_v_first=svf,
            )
            dht_main = _h0(P, H, K, V, svf, device)  # cp_dht is a tensor -> use_dht=True
            fused_gdr_bwd(
                q=q, k=k, v=v, a=a, g=g, b=b, do=do, dht=dht_main, h=h_main,
                scale=cfg.scale, cu_seqlens=cu, chunk_size=cs, state_v_first=svf,
            )

        # --- backward CP preprocessing ---
        if cfg.training and fused_gdr_dh is not None:
            dht_variants = {None}
            if cfg.enable_inter and cfg.output_final_state:
                dht_variants.add("dht")  # pure-inter dh gets a real dht (use_dht=T)
            dh = None
            for dv in dht_variants:
                dht = _h0(P, H, K, V, svf, device) if dv == "dht" else None
                _, dh_i = fused_gdr_dh(
                    q=q, k=k, a=a, g=g, b=b, do=do, dht=dht,
                    output_dh0=True, output_dh=False, scale=cfg.scale, cu_seqlens=cu,
                    num_warmup_chunks=num_warmup, state_v_first=svf,
                )
                if dh is None:
                    dh = dh_i
            correct_terminal_states(
                raw_dht=None, dht_buffer=dh, mt_buffer=mt, fallback_mask=fallback,
                seq_map_r2c=seq_map_r2c, state_v_first=svf,
            )
            if cfg.output_final_state:
                correct_terminal_states(
                    raw_dht=_h0(1, H, K, V, svf, device), dht_buffer=dh, mt_buffer=mt,
                    fallback_mask=fallback, seq_map_r2c=seq_map_r2c, state_v_first=svf,
                )
            if combined:
                aggregate_card_state(
                    dh, mt, fallback, seq_map_r2c, raw_h0=None,
                    state_v_first=svf, reverse=True, transpose_m=True, compute_m=False,
                )
                if cfg.output_final_state:
                    aggregate_card_state(
                        dh, mt, fallback, seq_map_r2c, raw_h0=_h0(1, H, K, V, svf, device),
                        state_v_first=svf, reverse=True, transpose_m=True, compute_m=False,
                    )

    # (2) CP main forward (is_cp=True, intra only): block_DV depends on grid, so sweep
    #     real_batch_size (partition count) and let the dispatcher pick+cache block_DV.
    if cfg.enable_intra:
        for S in cfg.S_list:
            for rb in cfg.rb_sweep:
                q, k, v, a, g, b, _ = io(rb, S)
                cu, cp_seq_map, raw_cu, *_ = _cp_meta(rb, S, H, device, cfg.cu_seqlens_dtype)
                cp_h0 = _h0(rb, H, K, V, svf, device)  # dummy; CP main fwd uis=True always
                fused_gdr_fwd(
                    q=q, k=k, v=v, a=a, g=g, b=b, scale=cfg.scale, initial_state=cp_h0,
                    output_final_state=cfg.output_final_state, output_o=True,
                    cu_seqlens=cu, cp_seq_map=cp_seq_map, raw_cu_seqlens=raw_cu,
                    state_v_first=svf,
                )


# ---------------------------------------------------------------------------
# auxiliary tilelang ops on the chunk op's path
# ---------------------------------------------------------------------------
def _warm_aux(cfg):
    from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector

    H, Hg, K = cfg.H, cfg.Hg, cfg.K
    cs, dtype, device = cfg.chunk_size, cfg.dtype, cfg.device
    S = cfg.S_list[0]
    nseq = 2
    T = nseq * S
    # Only the configured layout (varlen cu / dense cu=None), matching is_varlen.
    if cfg.is_varlen:
        cu = _idx([i * S for i in range(nseq + 1)], device, cfg.cu_seqlens_dtype)
        g_shape = (1, T, H)
    else:
        cu = None
        g_shape = (nseq, S, H)

    # forward cumsum: g in the gate dtype, reverse=False, zero_last_chunk=False.
    g_fwd = torch.randn(*g_shape, dtype=cfg.gate_dtype, device=device)
    chunk_local_cumsum(g=g_fwd, chunk_size=cs, cu_seqlens=cu)
    # backward cumsum: dg in fp32, reverse=True, zero_last_chunk=True.
    dg = torch.randn(*g_shape, dtype=torch.float32, device=device)
    chunk_local_cumsum(g=dg, chunk_size=cs, cu_seqlens=cu, reverse=True, zero_last_chunk=True)

    # group reduce only runs for GVA (Hg < H), on dq/dk-shaped buffers.
    if cfg.training and Hg < H:
        buf = torch.randn(1, T, H, K, dtype=dtype, device=device)
        group_reduce_vector(buf, Hg)


def _warm_l2norm(cfg):
    from flash_qla.utils import l2norm_fwd, l2norm_bwd

    S = cfg.S_list[0]
    x = torch.randn(1, 2 * S, cfg.Hg, cfg.K, dtype=cfg.dtype, device=cfg.device)
    y, rstd = l2norm_fwd(x)
    if cfg.training:
        l2norm_bwd(x, rstd, torch.randn_like(y))


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
class _Cfg:
    __slots__ = (
        "H", "Hg", "K", "V", "chunk_size", "dtype", "gate_dtype", "beta_dtype",
        "cu_seqlens_dtype", "device", "scale", "state_v_first", "use_initial_state",
        "output_final_state", "training", "is_varlen", "enable_intra", "enable_inter",
        "rb_sweep", "rb_rep", "S_list",
    )


def warmup(
    num_v_heads: int,
    num_qk_heads: int,
    *,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    gate_dtype: torch.dtype = torch.float32,
    beta_dtype: torch.dtype = torch.float32,
    cu_seqlens_dtype: torch.dtype = torch.int64,
    device: torch.device | int | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    use_initial_state: bool = True,
    output_final_state: bool = True,
    training: bool = True,
    is_varlen: bool = True,
    enable_intra: bool = True,
    enable_inter: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
) -> None:
    """Precompile the CP and non-CP kernels for a given model config.

    Drives every kernel dispatcher once per static-arg combination the config
    will hit (``is_cp``, ``is_varlen``, ``has_incomplete_tile``, all forward
    ``block_DV`` buckets, and the CP preprocessing/correction kernels), populating
    tilelang's JIT cache so the first real step does not stall on compilation.

    This is compilation warmup; it is unrelated to the CP "warmup chunks"
    (``get_warmup_chunks``), which are an algorithmic quantity.

    Args:
        num_v_heads: number of value heads ``H``.
        num_qk_heads: number of query/key heads ``Hg`` (``Hg < H`` enables GVA).
        head_dim: ``K == V`` head dim (must be 128 on sm90/sm100/sm103).
        dtype: q/k/v/a dtype — ``bfloat16`` or ``float16``, must match the model.
        gate_dtype: dtype of ``g`` (and thus of ``dg``). Training requires ``float32``
            (the backward asserts an fp32 ``dg``); it is a compile key.
        beta_dtype: dtype of ``beta``; also a compile key (fp32 in the reference setup).
        cu_seqlens_dtype: dtype of ``cu_seqlens`` for the varlen/CP paths — a compile key.
            Defaults to ``int64`` to match the ``LongTensor`` API contract. (The dense,
            non-varlen path always uses an internally-built int32 and is unaffected.)
        device: CUDA device (defaults to the current one).
        scale: softmax scale; ``None`` -> ``head_dim ** -0.5`` (the production default).
        state_v_first / use_initial_state / output_final_state / training:
            must match the production call; only the hit variants are compiled.
        is_varlen: the non-CP layout — ``True`` for packed/varlen inputs with
            ``cu_seqlens`` (the usual training layout; fixed per run), ``False`` for
            a dense ``[B, T, ...]`` batch. CP always runs varlen regardless.
        enable_intra / enable_inter: which CP paths to precompile. The non-CP path
            is always precompiled; ``enable_intra and enable_inter`` additionally
            compiles the inter+intra ``aggregate_card_state`` variants (where the
            architecture provides them).
        use_qk_l2norm_in_kernel: also warm the l2norm ``torch.compile`` helpers.
    """
    from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE

    assert dtype in (torch.bfloat16, torch.float16), (
        "FlashQLA only supports bfloat16 and float16."
    )
    assert num_v_heads % num_qk_heads == 0, "num_v_heads must be divisible by num_qk_heads."
    if device is None:
        device = torch.cuda.current_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    cfg = _Cfg()
    cfg.H, cfg.Hg = num_v_heads, num_qk_heads
    cfg.K = cfg.V = head_dim
    cfg.chunk_size = CHUNK_SIZE
    cfg.dtype, cfg.device, cfg.scale = dtype, device, scale
    cfg.gate_dtype, cfg.beta_dtype = gate_dtype, beta_dtype
    cfg.cu_seqlens_dtype = cu_seqlens_dtype
    cfg.state_v_first = state_v_first
    cfg.use_initial_state = use_initial_state
    cfg.output_final_state = output_final_state
    cfg.training = training
    cfg.is_varlen = is_varlen
    cfg.enable_intra, cfg.enable_inter = enable_intra, enable_inter

    sm = torch.cuda.get_device_properties(device).multi_processor_count
    rb_max = max(1, math.ceil(sm / num_v_heads))
    cfg.rb_sweep = list(range(1, rb_max + 1))
    cfg.rb_rep = min(2, rb_max)
    base = 2 * CHUNK_SIZE
    cfg.S_list = [base, base + CHUNK_SIZE // 2]

    _warm_non_cp(cfg)
    if enable_intra or enable_inter:
        if cfg.chunk_size != 64:
            warnings.warn(
                f"CP warmup skipped: the CP warmup-chunk kernels require chunk_size==64 "
                f"but this architecture uses chunk_size={cfg.chunk_size}.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            _warm_cp(cfg)
    _warm_aux(cfg)
    if use_qk_l2norm_in_kernel:
        _warm_l2norm(cfg)
    torch.cuda.synchronize(device)
