# tests/test_verify_gdr.py
import pytest
import torch

from ref_gdr import verify_ref
from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
    fused_recurrent_gdr_verify_fwd,
)
from flash_qla.utils import l2norm

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mk(N, D, Hk, Hv, num_slots, seed=0, ragged=False, distinct_gate=False):
    torch.manual_seed(seed)
    Ls = (torch.randint(1, D + 1, (N,)) if ragged else torch.full((N,), D)).to(torch.int32)
    cu = torch.zeros(N + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(Ls, 0)
    total = int(cu[-1])
    q = l2norm(torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(1, total, Hk, 128, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(1, total, Hv, 128, device="cuda", dtype=torch.bfloat16)
    if distinct_gate:  # per-head distinct decay (catches a V-major transpose bug)
        g = (torch.nn.functional.logsigmoid(torch.randn(1, total, Hv, device="cuda")) / 16
             * (1 + torch.arange(Hv, device="cuda").float()[None, None, :]))
    else:
        g = torch.nn.functional.logsigmoid(torch.randn(1, total, Hv, device="cuda")) / 16
    beta = torch.randn(1, total, Hv, device="cuda").sigmoid()
    pool = torch.randn(num_slots, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)  # V-major
    ibuf = torch.zeros(N + 1, D, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)
    si = torch.arange(N, dtype=torch.int32, device="cuda")
    ci = torch.arange(N, dtype=torch.int32, device="cuda")
    return q, k, v, g, beta, pool, si, cu.cuda(), ibuf, ci


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


@CUDA
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("Hk,Hv", [(8, 8), (2, 8)])
def test_verify_nocommit(ragged, Hk, Hv):
    N, D = 4, 4
    q, k, v, g, beta, pool, si, cu, ibuf, ci = _mk(N, D, Hk, Hv, num_slots=N, ragged=ragged)
    o = torch.empty(1, q.shape[1], Hv, 128, device="cuda", dtype=torch.bfloat16)
    pool0 = pool.clone()
    o_ref, pool_ref, ibuf_ref = verify_ref(q, k, v, g, beta, pool, si, cu, ibuf, ci, disable_state_update=True)
    fused_recurrent_gdr_verify_fwd(q, k, v, g, beta, pool, si, cu, ibuf, ci, o, disable_state_update=True)
    assert _rel(o, o_ref) <= 0.02
    assert _rel(ibuf, ibuf_ref) <= 0.02
    assert torch.equal(pool, pool0), "no-commit must not touch the pool"


@CUDA
def test_verify_commit():
    N, D, H = 4, 4, 8
    q, k, v, g, beta, pool, si, cu, ibuf, ci = _mk(N, D, H, H, num_slots=N)
    o = torch.empty(1, q.shape[1], H, 128, device="cuda", dtype=torch.bfloat16)
    pool_k = pool.clone()
    o_ref, pool_ref, ibuf_ref = verify_ref(q, k, v, g, beta, pool, si, cu, ibuf, ci, disable_state_update=False)
    fused_recurrent_gdr_verify_fwd(q, k, v, g, beta, pool_k, si, cu, ibuf, ci, o, disable_state_update=False)
    assert _rel(o, o_ref) <= 0.02
    assert _rel(pool_k, pool_ref) <= 0.02, "committed final state mismatch"


@CUDA
def test_verify_distinct_gate_vmajor():
    # per-head-distinct gates: a wrong V-major store would diverge from the reference
    N, D, H = 4, 4, 8
    q, k, v, g, beta, pool, si, cu, ibuf, ci = _mk(N, D, H, H, num_slots=N, distinct_gate=True)
    o = torch.empty(1, q.shape[1], H, 128, device="cuda", dtype=torch.bfloat16)
    o_ref, pool_ref, ibuf_ref = verify_ref(q, k, v, g, beta, pool, si, cu, ibuf, ci, disable_state_update=True)
    fused_recurrent_gdr_verify_fwd(q, k, v, g, beta, pool, si, cu, ibuf, ci, o, disable_state_update=True)
    assert _rel(o, o_ref) <= 0.02
    assert _rel(ibuf, ibuf_ref) <= 0.02


@CUDA
def test_verify_wrapper_gating():
    # high-level wrapper: host-side sigmoid gating + l2norm must match a hand-built reference
    from flash_qla import recurrent_gated_delta_rule_verify
    from flash_qla.ops.gated_delta_rule.fused_recurrent import gdn_sigmoid_gate

    N, D, H = 4, 4, 8
    total = N * D
    torch.manual_seed(1)
    q = torch.randn(1, total, H, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, H, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, total, H, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, total, H, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, total, H, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda").abs().log()
    dt_bias = torch.randn(H, device="cuda")
    pool = torch.randn(N, H, 128, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, total + 1, D, dtype=torch.int32, device="cuda")
    si = torch.arange(N, dtype=torch.int32, device="cuda")
    ci = torch.arange(N, dtype=torch.int32, device="cuda")
    ibuf = torch.zeros(N + 1, D, H, 128, 128, device="cuda", dtype=torch.bfloat16)

    g_ref, beta_ref = gdn_sigmoid_gate(A_log, a, dt_bias, b)
    o_ref, _, ibuf_ref = verify_ref(
        l2norm(q), l2norm(k), v, g_ref, beta_ref, pool, si, cu, ibuf, ci, disable_state_update=True)
    o = recurrent_gated_delta_rule_verify(
        A_log, a, dt_bias, q, k, v, b, pool, si, cu, ibuf, ci, disable_state_update=True)
    assert _rel(o, o_ref) <= 0.02
    assert _rel(ibuf, ibuf_ref) <= 0.02


@CUDA
@pytest.mark.parametrize("Hk,Hv", [(8, 8), (2, 8)])
def test_verify_in_kernel_gating(Hk, Hv):
    # in-kernel g/beta/l2norm must match the host-gating reference on the same raw inputs
    from flash_qla.ops.gated_delta_rule.fused_recurrent import gdn_sigmoid_gate
    from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
        fused_recurrent_gdr_verify_gated_fwd,
    )

    N, D = 4, 4
    total = N * D
    torch.manual_seed(2)
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

    g_ref, beta_ref = gdn_sigmoid_gate(A_log, a, dt_bias, b)
    o_ref, _, ibuf_ref = verify_ref(
        l2norm(q), l2norm(k), v, g_ref, beta_ref, pool, si, cu, ibuf, ci, disable_state_update=True)
    fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool, si, cu, ibuf, ci, o, disable_state_update=True)
    assert _rel(o, o_ref) <= 0.03
    assert _rel(ibuf, ibuf_ref) <= 0.03


@CUDA
def test_verify_cuda_graph():
    # the low-level entry must be CUDA-graph capturable (no host sync / no alloc) and replay correctly
    N, D, H = 4, 4, 8
    q, k, v, g, beta, pool, si, cu, ibuf, ci = _mk(N, D, H, H, num_slots=N)
    o = torch.empty(1, q.shape[1], H, 128, device="cuda", dtype=torch.bfloat16)

    def run():
        fused_recurrent_gdr_verify_fwd(q, k, v, g, beta, pool, si, cu, ibuf, ci, o, disable_state_update=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()

    o_eager = torch.empty_like(o)
    fused_recurrent_gdr_verify_fwd(q, k, v, g, beta, pool, si, cu, ibuf, ci, o_eager, disable_state_update=True)
    assert torch.equal(o, o_eager), "graph replay must match eager"


@CUDA
def test_verify_negctrl_vmajor_transpose():
    # discriminating: the V-major ibuf must DIFFER from a transposed-major reference
    N, D, H = 4, 4, 8
    q, k, v, g, beta, pool, si, cu, ibuf, ci = _mk(N, D, H, H, num_slots=N, distinct_gate=True)
    o = torch.empty(1, q.shape[1], H, 128, device="cuda", dtype=torch.bfloat16)
    o_ref, _, ibuf_ref = verify_ref(q, k, v, g, beta, pool, si, cu, ibuf, ci, disable_state_update=True)
    fused_recurrent_gdr_verify_fwd(q, k, v, g, beta, pool, si, cu, ibuf, ci, o, disable_state_update=True)
    assert _rel(ibuf, ibuf_ref) <= 0.02
    ibuf_wrong = ibuf_ref.transpose(-1, -2).contiguous()  # K/V swapped (the silent-bug case)
    assert _rel(ibuf, ibuf_wrong) > 0.2  # the test discriminates a wrong major-order


@CUDA
def test_verify_gated_cuda_graph():
    # the fully in-kernel gated path (no PyTorch gating/l2norm) is the capture-safe SGLang entry
    from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
        fused_recurrent_gdr_verify_gated_fwd,
    )
    N, D, H = 4, 4, 8
    total = N * D
    torch.manual_seed(3)
    q = torch.randn(1, total, H, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, H, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, total, H, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, total, H, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, total, H, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="cuda").abs().log()
    dt_bias = torch.randn(H, device="cuda")
    pool = torch.randn(N, H, 128, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, total + 1, D, dtype=torch.int32, device="cuda")
    si = torch.arange(N, dtype=torch.int32, device="cuda")
    ci = torch.arange(N, dtype=torch.int32, device="cuda")
    ibuf = torch.zeros(N + 1, D, H, 128, 128, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(1, total, H, 128, device="cuda", dtype=torch.bfloat16)

    def run():
        fused_recurrent_gdr_verify_gated_fwd(
            q, k, v, a, b, A_log, dt_bias, pool, si, cu, ibuf, ci, o, disable_state_update=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    o_eager = torch.empty_like(o)
    fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool, si, cu, ibuf, ci, o_eager, disable_state_update=True)
    assert torch.equal(o, o_eager), "in-kernel gated graph replay must match eager"


@CUDA
def test_verify_skip_slot():
    # a -1 pool slot must skip gather/commit/intermediate writes for that request
    N, D, H = 3, 4, 8
    q, k, v, g, beta, pool, si, cu, ibuf, ci = _mk(N, D, H, H, num_slots=N)
    si[1] = -1  # request 1 has no pool slot
    o = torch.empty(1, q.shape[1], H, 128, device="cuda", dtype=torch.bfloat16)
    ibuf0 = ibuf.clone()
    o_ref, pool_ref, ibuf_ref = verify_ref(q, k, v, g, beta, pool, si, cu, ibuf, ci, disable_state_update=True)
    fused_recurrent_gdr_verify_fwd(q, k, v, g, beta, pool, si, cu, ibuf, ci, o, disable_state_update=True)
    # request 1's tokens: o still produced (from zero state), ibuf row left untouched
    s0, s1 = int(cu[1]), int(cu[2])
    assert _rel(o[:, s0:s1], o_ref[:, s0:s1]) <= 0.02
    assert torch.equal(ibuf[int(ci[1])], ibuf0[int(ci[1])]), "skipped slot must not write ibuf"
