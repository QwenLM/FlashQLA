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
    CHUNK_SIZE = 64
elif tilelang.contrib.nvcc.get_target_compute_version() in ["10.0", "10.3"]:
    from .blackwell import fused_gdr_fwd, fused_gdr_bwd, fused_gdr_h, kkt_solve
    from .blackwell import get_warmup_chunks, get_warmup_chunks_bidi, correct_initial_states, correct_terminal_states
    from .blackwell.cp_bwd import fused_gdr_dh_ws as fused_gdr_dh
    CHUNK_SIZE = 64
elif tilelang.contrib.nvcc.get_target_compute_version() == "12.0":
    from .blackwell_sm120 import fused_gdr_fwd, fused_gdr_h, kkt_solve
    from .blackwell_sm120 import get_warmup_chunks, get_warmup_chunks_bidi, correct_initial_states, correct_terminal_states
    fused_gdr_bwd = None
    fused_gdr_dh = None
    CHUNK_SIZE = 32
else:
    raise ValueError(f"FlashQLA now support sm90, sm100, sm103 and sm120 only. Found compute version: {tilelang.contrib.nvcc.get_target_compute_version()}")
from .cp_context import intra_card_cp_preprocess, intra_card_cp_preprocess_bwd, _calc_cp_seqs, _create_cu_seqlens

from flash_qla.utils import input_guard


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
    g = chunk_local_cumsum(
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )
    A = kkt_solve(
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )
    cp_cache = None
    if auto_cp:
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens, cp_cache = intra_card_cp_preprocess(
            k=k, v=v, a=A, g=g, b=beta,
            raw_h0=initial_state,
            raw_cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
            enable_fwd_cp_cache=enable_fwd_cp_cache,
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
        h0, cu_seqlens_fwd, dht, cu_seqlens_bwd, seq_map_r2c, use_cp = intra_card_cp_preprocess_bwd(
            k=k, v=v, a=A, g=g, b=beta, raw_h0=initial_state,
            q=q, do=do, dht=dht, scale=scale,
            raw_cu_seqlens=cu_seqlens,
            state_v_first=state_v_first,
            cp_cache=cp_cache,
        )
    else:
        h0 = initial_state
        cu_seqlens_fwd = cu_seqlens
        dht = dht
        cu_seqlens_bwd = cu_seqlens
        seq_map_r2c = None
        use_cp = False

    h, _, _ = fused_gdr_h(
        k=k, v=v, a=A, g=g, b=beta,
        initial_state=h0,
        output_final_state=False,
        output_h=True,
        cu_seqlens=cu_seqlens_fwd,
        state_v_first=state_v_first,
    )
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q, k=k, v=v, a=A, g=g, b=beta,
        do=do, dht=dht, h=h, scale=scale,
        cu_seqlens=cu_seqlens_bwd,
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
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens)
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
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA (Grouped Value Attention) is applied if `HV > H`, where `HV` must be divisible by `H`.
        g (torch.Tensor):
            (forget) gating tensor of shape `[B, T, HV]`.
            `g` should be in log space (pre-computed decay).
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first ``[V, K]`` layout instead of the default ``[K, V]``. Default: ``False``.
        auto_cp (Optional[bool]):
            Whether to enable automatic intra-card CP. Default: `True`.
        enable_fwd_cp_cache (Optional[bool]):
            Whether to cache CP related variables during the forward pass. Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Notes:
        The TVM host code does not accept `strides == nullptr` even for compact
        tensors. You must explicitly set `strides` to a valid array when constructing
        the DLTensor. This limitation applies to any manual DLPack construction.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, HV, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
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
