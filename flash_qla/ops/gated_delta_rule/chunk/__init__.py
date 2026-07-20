# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang

from flash_qla.utils import l2norm_fwd, l2norm_bwd
from flash_qla.ops.utils import chunk_local_cumsum

if tilelang.contrib.nvcc.get_target_compute_version() == "9.0":
    from .hopper import fused_gdr_fwd, kkt_solve
    BACKWARD_SUPPORTED = True
    CHUNK_SIZE = 64
elif tilelang.contrib.nvcc.get_target_compute_version() == "10.0":
    from .blackwell import fused_gdr_fwd, kkt_solve
    BACKWARD_SUPPORTED = True
    CHUNK_SIZE = 64
elif tilelang.contrib.nvcc.get_target_compute_version() == "12.0":
    from .blackwell_sm120 import fused_gdr_fwd, kkt_solve
    BACKWARD_SUPPORTED = False
    CHUNK_SIZE = 32
else:
    raise ValueError("FlashQLA now support sm90 and sm100 only.")
from .cp_context import intra_card_cp_preprocess

from flash_qla.utils import input_guard


def _zero_sequence_start_grad(
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
):
    if cu_seqlens is None:
        dg[:, 0] = 0
    else:
        dg[0, cu_seqlens[:-1].long()] = 0


def _zero_padded_token_grads(
    cu_seqlens: torch.LongTensor | None,
    *grads: torch.Tensor,
):
    if cu_seqlens is not None:
        # FLA's varlen kernels leave the token tail unwritten.
        num_valid_tokens = int(cu_seqlens[-1].item())
        for grad in grads:
            grad[:, num_valid_tokens:] = 0


def _require_backward_support():
    if not BACKWARD_SUPPORTED:
        raise NotImplementedError(
            "Backward pass is not implemented for SM120 (Blackwell)."
            " Only forward pass is supported on this architecture."
        )


def _fla_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    state_v_first: bool,
):
    """Run FLA's stable backward against FlashQLA forward activations."""
    _require_backward_support()

    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import prepare_chunk_indices
    from fla.ops.utils.constant import RCP_LN2

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        if cu_seqlens is not None
        else None
    )
    # FlashQLA accumulates natural-log gates; FLA kernels consume log2 gates.
    fla_g = g * RCP_LN2
    # FlashQLA's A is ungated, while FLA's backward expects a gated A.
    _, _, fla_A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=fla_g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=CHUNK_SIZE,
    )
    dq, dk, dv, db, dg, dh0, _, _ = chunk_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=fla_g,
        beta=beta,
        A=fla_A,
        scale=scale,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=CHUNK_SIZE,
    )
    if initial_state is None:
        dh0 = None
        _zero_sequence_start_grad(dg, cu_seqlens)
    _zero_padded_token_grads(cu_seqlens, dq, dk, dv, db, dg)
    return dq, dk, dv, db, dg, dh0


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
    # These arguments are retained for API compatibility with the fused path.
    del A, auto_cp, cp_cache
    return _fla_gated_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        do=do,
        dht=dht,
        scale=scale if scale is not None else q.shape[-1] ** -0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        state_v_first=state_v_first,
    )


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
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        # The stable backward recomputes its intermediates and cannot consume
        # the FlashQLA-specific CP cache.
        g, _, o, _, final_state, _ = chunk_gated_delta_rule_fwd(
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
            enable_fwd_cp_cache=False,
        )

        ctx.save_for_backward(
            q, k, q_rstd, k_rstd, v, g, beta, initial_state, cu_seqlens
        )
        ctx.scale = scale
        ctx.state_v_first = state_v_first
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        (
            q, k, q_rstd, k_rstd, v, g, beta, initial_state, cu_seqlens
        ) = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0 = _fla_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            state_v_first=ctx.state_v_first,
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
            dh0,
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
            Retained for compatibility. Stable backward recomputes its state. Default: `True`.

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
    )

    return o, final_state
