# tests/test_prepass_gdr.py
# H1: the gating + qk-l2norm PRE-PASS kernel must produce g/beta/q_n/k_n bit-equivalent (bf16
# noise) to the reference gdn_sigmoid_gate + l2norm that the host-gated main verify kernel
# consumes. This is the correctness gate for the dedup pre-pass (replaces the in-hot-loop
# recompute of variant A).
import pytest
import torch

from flash_qla.utils import l2norm
from flash_qla.ops.gated_delta_rule.fused_recurrent import gdn_sigmoid_gate

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


@CUDA
@pytest.mark.parametrize("Hk,Hv", [(8, 8), (2, 8), (16, 32), (4, 16)])
@pytest.mark.parametrize("neg", [False, True])
def test_prepass_matches_reference(Hk, Hv, neg):
    from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
        fused_recurrent_gdr_verify_prepass,
    )
    N, D = 4, 5
    total = N * D
    torch.manual_seed(7)
    q = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(Hv, device="cuda").abs().log()
    dt_bias = torch.randn(Hv, device="cuda")

    q_n, k_n, g, beta = fused_recurrent_gdr_verify_prepass(
        q, k, a, b, A_log, dt_bias, allow_neg_eigval=neg)
    g_ref, beta_ref = gdn_sigmoid_gate(A_log, a, dt_bias, b, allow_neg_eigval=neg)

    assert _rel(q_n, l2norm(q)) <= 0.02, "q l2norm mismatch"
    assert _rel(k_n, l2norm(k)) <= 0.02, "k l2norm mismatch"
    assert _rel(g, g_ref) <= 0.02, "g (raw log-decay) mismatch"
    assert _rel(beta, beta_ref) <= 0.02, "beta mismatch"


@CUDA
def test_prepass_negctrl_raw_g_not_exp2():
    # discriminating: g must be the RAW log-decay (g<=0), NOT pre-exp2'd (which would be in (0,1]).
    # The main kernel applies decay=exp2(g*1.442695); a pre-exp2'd g would be silently wrong.
    from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
        fused_recurrent_gdr_verify_prepass,
    )
    N, D, Hk, Hv = 3, 4, 4, 8
    total = N * D
    torch.manual_seed(11)
    q = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(Hv, device="cuda").abs().log()
    dt_bias = torch.randn(Hv, device="cuda")
    _, _, g, _ = fused_recurrent_gdr_verify_prepass(q, k, a, b, A_log, dt_bias)
    assert (g <= 1e-4).all(), "g must be raw log-decay (<=0), not pre-exp2'd"


@CUDA
@pytest.mark.parametrize("Hk,Hv", [(16, 32), (2, 8)])
def test_prepass_plus_hostgated_matches_in_kernel_gated(Hk, Hv):
    # end-to-end: prepass + host-gated main kernel must match the in-kernel-gated kernel on the
    # SAME raw inputs (both compute the identical recurrence; only WHERE gating/l2norm happen
    # differs). Within in-kernel-gating tolerance (0.03).
    from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
        fused_recurrent_gdr_verify_prepass,
        fused_recurrent_gdr_verify_fwd,
        fused_recurrent_gdr_verify_gated_fwd,
    )
    N, D = 4, 6
    total = N * D
    torch.manual_seed(13)
    q = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, total, Hv, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(Hv, device="cuda").abs().log()
    dt_bias = torch.randn(Hv, device="cuda")
    pool = torch.randn(N, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, total + 1, D, dtype=torch.int32, device="cuda")
    si = torch.arange(N, dtype=torch.int32, device="cuda")
    ci = torch.arange(N, dtype=torch.int32, device="cuda")
    ibuf_a = torch.zeros(N + 1, D, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)
    ibuf_b = torch.zeros(N + 1, D, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)
    o_gated = torch.empty(1, total, Hv, 128, device="cuda", dtype=torch.bfloat16)
    o_pp = torch.empty(1, total, Hv, 128, device="cuda", dtype=torch.bfloat16)

    # in-kernel-gated (the current path)
    fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool, si, cu, ibuf_a, ci, o_gated, disable_state_update=True)
    # prepass + host-gated (the H1 path)
    q_n, k_n, g, beta = fused_recurrent_gdr_verify_prepass(q, k, a, b, A_log, dt_bias)
    fused_recurrent_gdr_verify_fwd(
        q_n, k_n, v, g, beta, pool, si, cu, ibuf_b, ci, o_pp, disable_state_update=True)

    assert _rel(o_pp, o_gated) <= 0.03, "prepass+host-gated o must match in-kernel-gated o"
    assert _rel(ibuf_b, ibuf_a) <= 0.03, "prepass+host-gated ibuf must match in-kernel-gated ibuf"


@CUDA
def test_prepass_path_cuda_graph():
    # the auto-prepass path (large-batch regime) must be CUDA-graph capturable: the persistent
    # scratch is warmed by the eager warmup runs, so capture sees no allocation, and replay
    # reuses the same buffers -> must match eager bit-for-bit.
    from flash_qla import recurrent_gated_delta_rule_verify
    N, D, Hk, Hv = 64, 4, 8, 8  # N*Hv*n_vt comfortably saturates SMs -> should_use_prepass True
    total = N * D
    torch.manual_seed(5)
    q = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, total, Hv, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, total, Hv, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(Hv, device="cuda").abs().log()
    dt_bias = torch.randn(Hv, device="cuda")
    pool = torch.randn(N, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, total + 1, D, dtype=torch.int32, device="cuda")
    si = torch.arange(N, dtype=torch.int32, device="cuda")
    ci = torch.arange(N, dtype=torch.int32, device="cuda")
    ibuf = torch.zeros(N + 1, D, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(1, total, Hv, 128, device="cuda", dtype=torch.bfloat16)

    def run():
        recurrent_gated_delta_rule_verify(
            A_log, a, dt_bias, q, k, v, b, pool, si, cu, ibuf, ci,
            o=o, fuse_gating=True, prepass=True, disable_state_update=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # warmup: JIT-compile prepass+main and allocate persistent scratch
            run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):  # would raise if the prepass allocated/synced inside capture
        run()
    graph.replay()
    torch.cuda.synchronize()
    o_graph = o.clone()

    o_eager = torch.empty_like(o)
    recurrent_gated_delta_rule_verify(
        A_log, a, dt_bias, q, k, v, b, pool, si, cu, ibuf, ci,
        o=o_eager, fuse_gating=True, prepass=True, disable_state_update=True)
    assert torch.equal(o_graph, o_eager), "prepass-path graph replay must match eager"
