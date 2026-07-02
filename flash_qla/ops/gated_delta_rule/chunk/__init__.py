# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang

from flash_qla.utils import l2norm_fwd, l2norm_bwd, prepare_chunk_offsets
from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector

if tilelang.contrib.nvcc.get_target_compute_version() == "9.0":
    from .hopper import fused_gdr_fwd, fused_gdr_bwd, fused_gdr_h, kkt_solve
    from .hopper import get_warmup_chunks, get_warmup_chunks_bidi, correct_initial_states, correct_terminal_states
    from .hopper.cp_bwd import fused_gdr_dh_ws as fused_gdr_dh
elif tilelang.contrib.nvcc.get_target_compute_version() == "10.0":
    from .blackwell import fused_gdr_fwd, fused_gdr_bwd, fused_gdr_h, kkt_solve
    from .blackwell import get_warmup_chunks, get_warmup_chunks_bidi, correct_initial_states, correct_terminal_states
    from .blackwell.cp_bwd import fused_gdr_dh_ws as fused_gdr_dh
elif tilelang.contrib.nvcc.get_target_compute_version() == "12.0":
    from .blackwell_sm120 import fused_gdr_fwd, fused_gdr_h, kkt_solve
    from .blackwell_sm120 import get_warmup_chunks, get_warmup_chunks_bidi, correct_initial_states, correct_terminal_states
    fused_gdr_bwd = None
    fused_gdr_dh = None
else:
    raise ValueError("FlashQLA now support sm90 and sm100 only.")
from .cp_context import intra_card_cp_preprocess, intra_card_cp_preprocess_bwd, _calc_cp_seqs, _create_cu_seqlens

from flash_qla.utils import input_guard

def _get_default_chunk_size() -> int:
    """Return the optimal default chunk_size based on the detected GPU compute capability."""
    version = tilelang.contrib.nvcc.get_target_compute_version()
    # SM90 (9.0) and SM100 (10.0) use 64; SM120 (12.0) uses 32
    if version == "12.0":
        return 32
    # Default for SM90, SM100, and any fallback
    return 64

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = True,
    state_v_first: bool = False,
    enable_fwd_cp_cache: bool = False,
):
    chunk_size = _get_default_chunk_size()
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    A = kkt_solve(
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
    )
    cp_cache = None
    if auto_cp:
        if enable_fwd_cp_cache:
            initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens, cached_mt, cached_fallback_bwd, cached_num_warmup_bwd = (
                intra_card_cp_preprocess(
                    k=k, v=v, a=A, g=g, b=beta,
                    raw_h0=initial_state,
                    raw_cu_seqlens=cu_seqlens,
                    state_v_first=state_v_first,
                    enable_fwd_cp_cache=True,
                )
            )
            if cached_mt is not None:
                cp_cache = (initial_state, cached_mt, cached_fallback_bwd, cached_num_warmup_bwd)
        else:
            initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
                intra_card_cp_preprocess(
                    k=k, v=v, a=A, g=g, b=beta,
                    raw_h0=initial_state,
                    raw_cu_seqlens=cu_seqlens,
                    state_v_first=state_v_first,
                )
            )
    else:
        cp_seq_map = None
        raw_cu_seqlens = None
    o, h, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=output_h,
        output_o=True,
        cu_seqlens=cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
        state_v_first=state_v_first,
    )
    return g, A, o, h, final_state, cp_cache


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    state_v_first: bool = False,
    auto_cp: bool = True,
    cp_cache: tuple | None = None,
):
    if fused_gdr_bwd is None:
        raise NotImplementedError(
            "Backward pass is not implemented for SM120 (Blackwell)."
            "Only forward pass is supported on this architecture."
        )

    batch_size, num_tokens, num_k_heads, _ = k.shape
    _, _, H, _ = v.shape
    chunk_size = A.shape[-1]

    if auto_cp and fused_gdr_dh is not None:
        h_initial_state, h_cu_seqlens, bwd_dht, bwd_cu_seqlens, seq_map_r2c, use_cp = (
            intra_card_cp_preprocess_bwd(
                k=k, v=v, a=A, g=g, b=beta, raw_h0=initial_state,
                q=q, do=do, dht=dht, scale=scale,
                raw_cu_seqlens=cu_seqlens,
                state_v_first=state_v_first,
                cp_cache=cp_cache,
            )
        )
    else:
        h_initial_state = initial_state
        h_cu_seqlens = cu_seqlens
        bwd_dht = dht
        bwd_cu_seqlens = cu_seqlens
        seq_map_r2c = None
        use_cp = False

    h, _, _ = fused_gdr_h(
        k=k, v=v, a=A, g=g, b=beta,
        initial_state=h_initial_state,
        output_final_state=False,
        output_h=True,
        cu_seqlens=h_cu_seqlens,
        state_v_first=state_v_first,
    )
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q, k=k, v=v, a=A, g=g, b=beta,
        do=do, dht=bwd_dht, h=h, scale=scale,
        cu_seqlens=bwd_cu_seqlens,
        state_v_first=state_v_first,
    )

    if use_cp:  # TODO store dh0 in fused_bwd kernel
        dh0 = dh0[seq_map_r2c[:-1].long()] if dh0 is not None else None
    elif initial_state is None:
        dh0 = None

    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq = group_reduce_vector(dq, Hg)
        dk = group_reduce_vector(dk, Hg)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        state_v_first: bool = False,
        auto_cp: bool = True,
        use_qk_l2norm_in_kernel: bool = False,
        enable_fwd_cp_cache: bool = True,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g, A, o, _, final_state, cp_cache = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=False,
            cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
            auto_cp=auto_cp,
            enable_fwd_cp_cache=enable_fwd_cp_cache,
        )

        if cp_cache is not None:
            cached_cp_h0, cached_mt, cached_fallback_bwd, cached_num_warmup_bwd = cp_cache
            ctx.save_for_backward(
                q, k, q_rstd, k_rstd, v, g, beta, A, initial_state, cu_seqlens,
                cached_cp_h0, cached_mt, cached_fallback_bwd, cached_num_warmup_bwd,
            )
            ctx._cp_cache_count = 4
        else:
            ctx.save_for_backward(q, k, q_rstd, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
            ctx._cp_cache_count = 0
        ctx.scale = scale
        ctx.state_v_first = state_v_first
        ctx.autocp = auto_cp
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        if ctx._cp_cache_count == 4:
            q, k, q_rstd, k_rstd, v, g, beta, A, initial_state, cu_seqlens, \
                cached_cp_h0, cached_mt, cached_fallback_bwd, cached_num_warmup_bwd = ctx.saved_tensors
            cp_cache = (cached_cp_h0, cached_mt, cached_fallback_bwd, cached_num_warmup_bwd)
        else:
            q, k, q_rstd, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
            cp_cache = None

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            state_v_first=ctx.state_v_first,
            auto_cp=ctx.autocp,
            cp_cache=cp_cache,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0 if initial_state is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    state_v_first: bool = False,
    auto_cp: bool = True,
    enable_fwd_cp_cache: bool = True,
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16, (
        "FlashQLA only supports bfloat16 and float16."
    )
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_v_first,
        auto_cp,
        use_qk_l2norm_in_kernel,
        enable_fwd_cp_cache,
    )

    return o, final_state