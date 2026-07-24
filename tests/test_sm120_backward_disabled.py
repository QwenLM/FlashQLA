# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Verify that the backward pass is properly disabled on SM120 (Blackwell).

On SM120, only forward pass is supported. Any attempt to invoke the backward
path — whether through the low-level ``chunk_gated_delta_rule_bwd`` API, the
autograd ``.backward()`` path, or the CP ``intra_card_cp_preprocess_bwd``
helper — must raise ``NotImplementedError`` with a clear message.
"""

import pytest
import torch
import tilelang

from flash_qla import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_qla
from flash_qla import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_qla
from flash_qla import chunk_gated_delta_rule
from flash_qla.utils import l2norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM_K = 128
HEAD_DIM_V = 128


def _make_inputs(batch_size=1, num_tokens=256, num_k_heads=4, num_v_heads=4,
                 use_h0=False, state_v_first=False, device="cuda", dtype=torch.bfloat16):
    """Create a minimal set of inputs suitable for triggering backward."""
    torch.manual_seed(42)

    q = l2norm(torch.randn(
        batch_size, num_tokens, num_k_heads, HEAD_DIM_K,
        device=device, dtype=dtype,
    ))
    k = l2norm(torch.randn(
        batch_size, num_tokens, num_k_heads, HEAD_DIM_K,
        device=device, dtype=dtype,
    ))
    v = torch.randn(
        batch_size, num_tokens, num_v_heads, HEAD_DIM_V,
        device=device, dtype=dtype,
    )
    g = torch.nn.functional.logsigmoid(torch.randn(
        batch_size, num_tokens, num_v_heads,
        device=device, dtype=torch.float32,
    )) / 16
    beta = torch.randn(
        batch_size, num_tokens, num_v_heads,
        device=device, dtype=torch.float32,
    ).sigmoid()

    do = torch.randn_like(v)

    h0 = None
    dht = None
    if use_h0:
        h0 = torch.randn(
            batch_size, num_v_heads, HEAD_DIM_V, HEAD_DIM_K,
            device=device, dtype=torch.float32,
        ) if state_v_first else torch.randn(
            batch_size, num_v_heads, HEAD_DIM_K, HEAD_DIM_V,
            device=device, dtype=torch.float32,
        )
        dht = torch.randn_like(h0) / 8

    scale = HEAD_DIM_K ** (-0.5)
    return q, k, v, g, beta, do, h0, dht, scale


def _assert_not_implemented_error(exc_info, context: str):
    """Assert that the raised error is NotImplementedError with SM120 mention."""
    assert isinstance(exc_info.value, NotImplementedError), (
        f"{context}: expected NotImplementedError, got {type(exc_info.value).__name__}"
    )
    msg = str(exc_info.value).lower()
    assert "sm120" in msg or "blackwell" in msg or "not implemented" in msg, (
        f"{context}: error message should mention SM120/Blackwell/not implemented, "
        f"got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# SM120-only tests: backward is disabled
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.sm120
def test_forward_still_works_on_sm120():
    """Forward pass must succeed on SM120 — only backward is gated."""
    q, k, v, g, beta, do, h0, dht, scale = _make_inputs(
        use_h0=False, state_v_first=False,
    )

    # Forward should *not* raise
    _, _, o, _, final_state, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=None,
        cu_seqlens=None,
        output_final_state=True,
        output_h=True,
        auto_cp=False,
        state_v_first=False,
    )
    assert o is not None
    assert o.shape == v.shape


@pytest.mark.gpu
@pytest.mark.sm120
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_bwd_direct_call_raises_not_implemented(state_v_first, use_h0):
    """``chunk_gated_delta_rule_bwd`` must raise NotImplementedError on SM120."""
    q, k, v, g, beta, do, h0, dht, scale = _make_inputs(
        use_h0=use_h0, state_v_first=state_v_first,
    )

    # Run forward to get g, A
    g_qla, A_qla, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=None,
        output_final_state=False,
        output_h=False,
        auto_cp=False,
        state_v_first=state_v_first,
    )

    with pytest.raises(NotImplementedError) as exc_info:
        chunk_gated_delta_rule_bwd_qla(
            q, k, v, g_qla, beta, A_qla, do, dht,
            scale, h0, None, state_v_first, auto_cp=False,
        )
    _assert_not_implemented_error(exc_info, "chunk_gated_delta_rule_bwd")


@pytest.mark.gpu
@pytest.mark.sm120
def test_autograd_backward_raises_not_implemented():
    """Autograd ``.backward()`` through the Function must raise NotImplementedError."""
    q, k, v, g, beta, do, h0, dht, scale = _make_inputs(
        use_h0=False, state_v_first=False,
    )

    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    g = g.detach().requires_grad_(True)
    beta = beta.detach().requires_grad_(True)

    o, _ = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        state_v_first=False,
        auto_cp=False,
    )

    loss = o.sum()

    with pytest.raises(NotImplementedError) as exc_info:
        loss.backward()
    _assert_not_implemented_error(exc_info, "autograd backward")


@pytest.mark.gpu
@pytest.mark.sm120
def test_cp_preprocess_bwd_raises_not_implemented():
    """``intra_card_cp_preprocess_bwd`` must raise NotImplementedError on SM120."""
    from flash_qla.ops.gated_delta_rule.chunk.cp_context import intra_card_cp_preprocess_bwd

    q, k, v, g, beta, do, h0, dht, scale = _make_inputs(
        use_h0=True, state_v_first=False,
    )

    # h0 / dht in reference layout [B, H, K, V]
    h0_ref = h0
    dht_ref = dht
    if h0_ref is not None:
        h0_ref = h0_ref.transpose(-1, -2).contiguous() if h0_ref.shape[-1] == HEAD_DIM_K else h0_ref
        dht_ref = dht_ref.transpose(-1, -2).contiguous() if dht_ref.shape[-1] == HEAD_DIM_K else dht_ref

    # Run forward to get A
    _, A_qla, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=None,
        output_final_state=False,
        output_h=False,
        auto_cp=False,
        state_v_first=False,
    )

    with pytest.raises(NotImplementedError) as exc_info:
        intra_card_cp_preprocess_bwd(
            k=k, v=v, a=A_qla, g=g, b=beta,
            raw_h0=h0_ref,
            q=q, do=do, dht=dht_ref,
            scale=scale,
            raw_cu_seqlens=None,
            state_v_first=False,
            force_cp=1,
        )
    _assert_not_implemented_error(exc_info, "intra_card_cp_preprocess_bwd")


# ---------------------------------------------------------------------------
# Smoke test: on non-SM120 architectures, backward still works
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_bwd_works_on_non_sm120(state_v_first):
    """Sanity check: backward must *not* raise on supported architectures."""
    if tilelang.contrib.nvcc.get_target_compute_version() == "12.0":
        pytest.skip("Backward is disabled on SM120 — this test only runs on SM90/SM100/SM103")
    q, k, v, g, beta, do, h0, dht, scale = _make_inputs(
        use_h0=True, state_v_first=state_v_first,
    )

    g_qla, A_qla, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=None,
        output_final_state=False,
        output_h=False,
        auto_cp=False,
        state_v_first=state_v_first,
    )

    # This must *not* raise
    dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht,
        scale, h0, None, state_v_first, auto_cp=False,
    )
    assert dq.shape == v.shape
    assert dk.shape == v.shape
    assert dv.shape == v.shape


if __name__ == "__main__":
    # Allow running directly for quick smoke check
    _cv = tilelang.contrib.nvcc.get_target_compute_version()
    print(f"Detected compute version: {_cv}")
    if _cv == "12.0":
        print("SM120 detected — running backward-disabled tests...")
        test_forward_still_works_on_sm120()
        print("  ✓ test_forward_still_works_on_sm120 passed")
        test_bwd_direct_call_raises_not_implemented(state_v_first=False, use_h0=False)
        print("  ✓ test_bwd_direct_call_raises_not_implemented[no_h0-kv] passed")
        test_bwd_direct_call_raises_not_implemented(state_v_first=False, use_h0=True)
        print("  ✓ test_bwd_direct_call_raises_not_implemented[h0-kv] passed")
        test_autograd_backward_raises_not_implemented()
        print("  ✓ test_autograd_backward_raises_not_implemented passed")
        test_cp_preprocess_bwd_raises_not_implemented()
        print("  ✓ test_cp_preprocess_bwd_raises_not_implemented passed")
        print("All SM120 backward-disabled tests passed.")
    else:
        print(f"Not SM120 (arch={_cv}) — test_bwd_works_on_non_sm120...")
        test_bwd_works_on_non_sm120(state_v_first=False)
        print("  ✓ test_bwd_works_on_non_sm120[kv] passed")
        test_bwd_works_on_non_sm120(state_v_first=True)
        print("  ✓ test_bwd_works_on_non_sm120[vk] passed")
        print("OK")
