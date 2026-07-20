# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import pytest
import torch

from flash_qla import chunk_gated_delta_rule
from flash_qla.utils import l2norm


DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 128
RTOL = 0.02


def _recurrent_gated_delta_rule(q, k, v, g, beta, cu_seqlens, initial_state=None):
    """Small, stable reference for strong-decay gradient tests."""
    num_value_heads = v.shape[2]
    q = q.repeat_interleave(num_value_heads // q.shape[2], dim=2)
    k = k.repeat_interleave(num_value_heads // k.shape[2], dim=2)
    boundaries = cu_seqlens.tolist() if cu_seqlens is not None else [0, q.shape[1]]
    outputs = []
    final_states = []
    for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (
            initial_state[sequence]
            if initial_state is not None
            else q.new_zeros(num_value_heads, q.shape[-1], v.shape[-1])
        )
        for token in range(begin, end):
            state = state * g[0, token, :, None, None].exp()
            prediction = torch.einsum("hk,hkv->hv", k[0, token], state)
            residual = beta[0, token, :, None] * (v[0, token] - prediction)
            state = state + torch.einsum("hk,hv->hkv", k[0, token], residual)
            outputs.append(torch.einsum("hk,hkv->hv", q[0, token], state))
        final_states.append(state)
    output = torch.stack(outputs).unsqueeze(0) * q.shape[-1] ** -0.5
    return output, torch.stack(final_states)


def _assert_relative_l2(actual, expected, name):
    error = torch.linalg.vector_norm(actual.double() - expected.double()).item()
    reference = torch.linalg.vector_norm(expected.double()).item()
    assert error <= reference * RTOL, (
        f"{name}: error={error:.6f}, reference={reference:.6f}, "
        f"relative={error / reference:.6f} > rtol={RTOL}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "cu_seqlens_list",
    [None, pytest.param([0, 47, 128], id="packed")],
)
def test_qwen_decay_parameter_gradients(cu_seqlens_list):
    """Strong Qwen decay transforms need stable parameter gradients."""
    torch.manual_seed(20260719)
    q = l2norm(torch.randn(1, 128, 2, HEAD_DIM, device=DEVICE, dtype=DTYPE))
    k = l2norm(torch.randn(1, 128, 2, HEAD_DIM, device=DEVICE, dtype=DTYPE))
    v = torch.randn(1, 128, 4, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    beta = torch.randn(1, 128, 4, device=DEVICE).sigmoid()
    gate_input = torch.randn(1, 128, 4, device=DEVICE) * 0.5
    A_log = torch.tensor([0.5, 2.0, 8.0, 16.0], device=DEVICE).log()
    dt_bias = torch.ones(4, device=DEVICE)
    do = torch.randn_like(v)
    cu_seqlens = (
        torch.tensor(cu_seqlens_list, device=DEVICE, dtype=torch.int32)
        if cu_seqlens_list is not None
        else None
    )

    qla_params = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in (gate_input, A_log, dt_bias)
    ]
    qla_g = -qla_params[1].exp() * torch.nn.functional.softplus(
        qla_params[0] + qla_params[2]
    )
    qla_g.retain_grad()
    qla_o, _ = chunk_gated_delta_rule(
        q, k, v, qla_g, beta, cu_seqlens=cu_seqlens, auto_cp=True
    )
    (qla_o.float() * do.float()).sum().backward()

    ref_params = [
        tensor.detach().double().requires_grad_(True)
        for tensor in (gate_input, A_log, dt_bias)
    ]
    ref_g = -ref_params[1].exp() * torch.nn.functional.softplus(
        ref_params[0] + ref_params[2]
    )
    ref_o, _ = _recurrent_gated_delta_rule(
        q.double(), k.double(), v.double(), ref_g, beta.double(), cu_seqlens
    )
    (ref_o * do.double()).sum().backward()

    for name, actual, expected in zip(
        ("gate_input", "A_log", "dt_bias"), qla_params, ref_params, strict=True
    ):
        _assert_relative_l2(actual.grad, expected.grad, name)

    starts = (
        cu_seqlens[:-1] if cu_seqlens is not None else torch.tensor([0], device=DEVICE)
    )
    assert torch.count_nonzero(qla_g.grad[0, starts]).item() == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "cu_seqlens_list,state_v_first,use_qk_l2norm_in_kernel",
    [
        pytest.param(None, False, False, id="fixed-kv"),
        pytest.param([0, 47, 117], True, True, id="padded-vk-l2norm"),
    ],
)
def test_public_initial_and_final_state_gradients(
    cu_seqlens_list, state_v_first, use_qk_l2norm_in_kernel
):
    """The public API must propagate output and final-state gradients."""
    torch.manual_seed(20260720)
    num_sequences = 1 if cu_seqlens_list is None else len(cu_seqlens_list) - 1
    q = torch.randn(1, 128, 2, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    if not use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    v = torch.randn(1, 128, 4, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    g = torch.nn.functional.logsigmoid(torch.randn(1, 128, 4, device=DEVICE)) / 16
    beta = torch.randn(1, 128, 4, device=DEVICE).sigmoid()
    h0 = torch.randn(num_sequences, 4, HEAD_DIM, HEAD_DIM, device=DEVICE)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0) / 8
    cu_seqlens = (
        torch.tensor(cu_seqlens_list, device=DEVICE, dtype=torch.int32)
        if cu_seqlens_list is not None
        else None
    )

    qla_inputs = [
        tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)
    ]
    qla_h0 = h0.transpose(-1, -2) if state_v_first else h0
    qla_h0 = qla_h0.contiguous().detach().requires_grad_(True)
    qla_dht = dht.transpose(-1, -2) if state_v_first else dht
    qla_o, qla_ht = chunk_gated_delta_rule(
        *qla_inputs,
        initial_state=qla_h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        state_v_first=state_v_first,
        auto_cp=True,
    )
    num_valid_tokens = 128 if cu_seqlens is None else int(cu_seqlens[-1].item())
    (
        (qla_o[:, :num_valid_tokens].float() * do[:, :num_valid_tokens].float()).sum()
        + (qla_ht * qla_dht).sum()
    ).backward()

    ref_inputs = [
        tensor.detach().double().requires_grad_(True) for tensor in (q, k, v, g, beta)
    ]
    ref_h0 = h0.detach().double().requires_grad_(True)
    ref_q, ref_k, ref_v, ref_g, ref_beta = ref_inputs
    if use_qk_l2norm_in_kernel:
        ref_q = torch.nn.functional.normalize(ref_q, dim=-1)
        ref_k = torch.nn.functional.normalize(ref_k, dim=-1)
    ref_o, ref_ht = _recurrent_gated_delta_rule(
        ref_q, ref_k, ref_v, ref_g, ref_beta, cu_seqlens, ref_h0
    )
    (
        (ref_o * do[:, :num_valid_tokens].double()).sum()
        + (ref_ht * dht.double()).sum()
    ).backward()

    for name, actual, expected in zip(
        ("q", "k", "v", "g", "beta"), qla_inputs, ref_inputs, strict=True
    ):
        _assert_relative_l2(actual.grad, expected.grad, name)
    qla_dh0 = qla_h0.grad.transpose(-1, -2) if state_v_first else qla_h0.grad
    _assert_relative_l2(qla_dh0, ref_h0.grad, "initial_state")
