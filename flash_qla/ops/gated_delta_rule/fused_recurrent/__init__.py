# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import torch
import torch.nn.functional as F
import tilelang

from flash_qla.utils import l2norm

if tilelang.contrib.nvcc.get_target_compute_version() == "9.0":
    from .hopper import fused_recurrent_gdr_fwd  # noqa: F401
    from .hopper.fused_recurrent_verify import (  # noqa: F401
        fused_recurrent_gdr_verify_fwd,
        fused_recurrent_gdr_verify_gated_fwd,
        fused_recurrent_gdr_verify_prepass,
        get_prepass_scratch,
        should_use_prepass,
    )
else:
    raise ValueError("FlashQLA now support sm90 only.")

__all__ = [
    "fused_recurrent_gdr_fwd",
    "recurrent_gated_delta_rule",
    "fused_recurrent_gdr_verify_fwd",
    "fused_recurrent_gdr_verify_gated_fwd",
    "recurrent_gated_delta_rule_verify",
]


def recurrent_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
    seqlens=None,
    head_first=False,
    head_batch=None,
):
    assert q.dtype == k.dtype == v.dtype and q.dtype != torch.float32
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0 and q.shape[-1] == v.shape[-1] == 128
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)
    o, final_state = fused_recurrent_gdr_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        seqlens=seqlens,
        head_batch=head_batch,
    )
    return o.to(q.dtype), final_state


def gdn_sigmoid_gate(A_log, a, dt_bias, b, allow_neg_eigval=False):
    """Host-side GDN gating (the sigmoid_gating family): g = -exp(A_log)*softplus(a+dt_bias),
    beta = sigmoid(b) (x2 if allow_neg_eigval). A_log,dt_bias:[H]; a,b:[...,H]. Returns fp32."""
    g = -torch.exp(A_log.float())[(None,) * (a.dim() - 1)] * F.softplus(
        a.float() + dt_bias.float()[(None,) * (a.dim() - 1)]
    )
    beta = torch.sigmoid(b.float())
    if allow_neg_eigval:
        beta = beta * 2
    return g, beta


def recurrent_gated_delta_rule_verify(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    ssm_states,
    cache_indices,
    query_start_loc,
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps=None,
    o=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    disable_state_update=True,
    allow_neg_eigval=False,
    fuse_gating=False,
    prepass=None,  # H1 dedup gating+l2norm pre-pass: None=auto (regime-gated), True/False=force
    retrieve_parent_token=None,  # accepted-and-IGNORED (DFlash width-1; tree path not built)
):
    """High-level SGLang DFlash verify entry. q,k:[1,T,Hk,128] v:[1,T,Hv,128]; a,b:[1,T,Hv];
    A_log,dt_bias:[Hv]; ssm_states pool V-major [num_slots,Hv,128,128].

    CUDA-graph note: for capture, use ``fuse_gating=True`` (computes g/beta + qk-l2norm INSIDE
    the kernel from raw a,b,A_log,dt_bias -- no PyTorch gating/l2norm, no allocation when ``o``
    is provided -> fully capture-safe). The default ``fuse_gating=False`` path computes g/beta +
    qk-l2norm in PyTorch (l2norm is ``@torch.compile``'d) and allocates them; run it OUTSIDE
    capture or prefer ``fuse_gating=True`` inside it.

    H1 (``fuse_gating=True``): in the large-batch / multi-draft-token regime the gating + qk-l2norm
    are split into a tiny dedup PRE-PASS kernel (computed ONCE per (token,K-head) instead of once
    per (token,V-head,V-tile) in the hot loop) feeding the host-gated main kernel -- measured
    ~12-15% faster at T=12 large batch. Regime-gated by ``should_use_prepass`` (the single
    in-kernel-gated kernel A is kept for latency-bound single-request, where a 2nd launch is a net
    loss). The prepass scratch is a persistent, never-evicting cache -> capture-safe after warmup.
    ``prepass`` forces the choice (None=auto, True=prepass+main, False=single gated kernel A)."""
    assert q.dtype == k.dtype == v.dtype and q.dtype != torch.float32
    assert q.shape[-1] == v.shape[-1] == 128 and v.shape[2] % k.shape[2] == 0
    if cache_steps is not None:  # static shape check (capture-safe; no value read)
        assert intermediate_states_buffer.shape[1] >= cache_steps, (
            f"intermediate_states_buffer cache-steps dim {intermediate_states_buffer.shape[1]} "
            f"< cache_steps {cache_steps}"
        )
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    if o is None:
        o = torch.empty(1, q.shape[1], v.shape[2], v.shape[-1], device=q.device, dtype=q.dtype)

    if fuse_gating:
        N, total_tokens, Hk = cache_indices.shape[0], q.shape[1], k.shape[2]
        H = v.shape[2]
        use_prepass = should_use_prepass(N, H, total_tokens) if prepass is None else prepass
        if use_prepass:
            # H1: dedup gating + qk-l2norm into a run-once pre-pass, then the host-gated main
            # kernel (no in-loop transcendentals). Persistent scratch -> capture-safe after warmup.
            q_n, k_n, g_pp, beta_pp = get_prepass_scratch(total_tokens, Hk, H, q.device, q.dtype)
            fused_recurrent_gdr_verify_prepass(
                q, k, a, b, A_log, dt_bias, q_n, k_n, g_pp, beta_pp,
                allow_neg_eigval=allow_neg_eigval,
            )
            fused_recurrent_gdr_verify_fwd(
                q_n, k_n, v, g_pp, beta_pp, ssm_states, cache_indices, query_start_loc,
                intermediate_states_buffer, intermediate_state_indices, o,
                scale=scale, disable_state_update=disable_state_update,
            )
            return o
        fused_recurrent_gdr_verify_gated_fwd(
            q, k, v, a, b, A_log, dt_bias, ssm_states, cache_indices, query_start_loc,
            intermediate_states_buffer, intermediate_state_indices, o,
            scale=scale, disable_state_update=disable_state_update, allow_neg_eigval=allow_neg_eigval,
        )
        return o

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)
    g, beta = gdn_sigmoid_gate(A_log, a, dt_bias, b, allow_neg_eigval)
    fused_recurrent_gdr_verify_fwd(
        q, k, v, g, beta, ssm_states, cache_indices, query_start_loc,
        intermediate_states_buffer, intermediate_state_indices, o,
        scale=scale, disable_state_update=disable_state_update,
    )
    return o
