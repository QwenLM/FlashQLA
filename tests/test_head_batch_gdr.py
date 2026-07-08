# tests/test_head_batch_gdr.py
# Head-batched GQA specialization of the gemm-free decode kernel: one CTA processes all
# grp = Hv//Hk V-heads sharing a K/Q head. Validates correctness + a within-group band-swap
# negative control + direct equality with the per-head path. Forces head_batch=True so the
# new code is actually exercised (auto stays OFF for the host-gated decode path).
import pytest
import torch

from ref_gdr import decode_recur
from flash_qla.utils import l2norm

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mk(B, T, Hk, Hv, seed=0, distinct_gate=True):
    """bf16 inputs. distinct_gate gives each V-head its own decay magnitude so a within-group
    head-band swap produces a far-off result (mandatory for the band-swap negative control)."""
    torch.manual_seed(seed)
    q = l2norm(torch.randn(B, T, Hk, 128, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(B, T, Hk, 128, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(B, T, Hv, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, device="cuda")) / 16
    if distinct_gate:
        g = g * (1 + torch.arange(Hv, device="cuda").float()[None, None, :])
    beta = torch.randn(B, T, Hv, device="cuda").sigmoid()
    return q, k, v, g, beta


def _rel(a, b):
    return (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)


@CUDA
@pytest.mark.parametrize("Hk,Hv", [(4, 8), (2, 8)])  # grp = 2, 4
def test_head_batch_compiles_and_runs(Hk, Hv):
    # smoke: the factory must JIT-build for both grp values (isolates a lowering failure).
    from flash_qla import recurrent_gated_delta_rule
    q, k, v, g, beta = _mk(1, 8, Hk, Hv)
    o, s = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, output_final_state=True, head_batch=True)
    assert o.shape == (1, 8, Hv, 128) and s.shape == (1, Hv, 128, 128)
    assert torch.isfinite(o.float()).all()


@CUDA
@pytest.mark.parametrize("D", [1, 8, 12])
@pytest.mark.parametrize("Hk,Hv", [(4, 8), (2, 8)])  # grp = 2, 4
@pytest.mark.parametrize("use_h0", [False, True])
def test_head_batch_matches_reference(D, Hk, Hv, use_h0):
    from flash_qla import recurrent_gated_delta_rule
    B = 1
    q, k, v, g, beta = _mk(B, D, Hk, Hv)
    h0 = (torch.randn(B, Hv, 128, 128, device="cuda", dtype=torch.float32) if use_h0 else None)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, initial_state=h0)
    o_hb, s_hb = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, initial_state=h0,
        output_final_state=True, head_batch=True)
    assert _rel(o_hb, o_ref) <= 0.02
    assert _rel(s_hb, s_ref) <= 0.02


@CUDA
@pytest.mark.parametrize("Hk,Hv", [(4, 8), (2, 8)])
def test_head_batch_equals_per_head(Hk, Hv):
    # the two paths must agree on identical inputs (cheap regression catch; no fp32 ref).
    from flash_qla import recurrent_gated_delta_rule
    q, k, v, g, beta = _mk(1, 8, Hk, Hv)
    o_hb, s_hb = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, output_final_state=True, head_batch=True)
    o_ph, s_ph = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, output_final_state=True, head_batch=False)
    assert _rel(o_hb, o_ph) <= 0.02
    assert _rel(s_hb, s_ph) <= 0.02


@CUDA
def test_head_batch_low_occupancy_block_dv32():
    # B*Hg small -> head-batch wrapper picks block_DV=32 (the low-CTA tail); must still match.
    from flash_qla import recurrent_gated_delta_rule
    q, k, v, g, beta = _mk(1, 8, 2, 8)  # Hg=2 -> grid_base=2 -> block_DV=32
    o_ref, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_hb, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5, head_batch=True)
    assert _rel(o_hb, o_ref) <= 0.02


@CUDA
def test_head_batch_high_occupancy_block_dv64():
    # B*Hg large -> head-batch wrapper picks block_DV=64; must still match.
    from flash_qla import recurrent_gated_delta_rule
    B, Hk, Hv = 16, 4, 8  # Hg=4 -> grid_base=64 -> block_DV=64, grp=2
    q, k, v, g, beta = _mk(B, 4, Hk, Hv)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_hb, s_hb = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, output_final_state=True, head_batch=True)
    assert _rel(o_hb, o_ref) <= 0.02
    assert _rel(s_hb, s_ref) <= 0.02


@CUDA
def test_head_batch_ragged_seqlens():
    from flash_qla import recurrent_gated_delta_rule
    B, Hk, Hv = 3, 2, 8
    q, k, v, g, beta = _mk(B, 8, Hk, Hv)
    seqlens = torch.tensor([1, 5, 8], device="cuda", dtype=torch.int32)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, seqlens=seqlens)
    o_hb, s_hb = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, seqlens=seqlens,
        output_final_state=True, head_batch=True)
    for b in range(B):
        L = int(seqlens[b])
        assert _rel(o_hb[b, :L], o_ref[b, :L]) <= 0.02
        assert _rel(s_hb[b], s_ref[b]) <= 0.02


@CUDA
def test_negctrl_head_band_swap():
    # discriminating: head-batch must match the correct band layout and DIFFER from a
    # within-group V-head swap. gqa_mod can't express this (it only re-routes the K/Q source).
    from flash_qla import recurrent_gated_delta_rule
    B, D, Hk, Hv = 1, 8, 2, 8  # grp=4
    q, k, v, g, beta = _mk(B, D, Hk, Hv)  # distinct per-head gate (mandatory)
    o_ok, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_swap, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, band_perm=True)
    o_hb, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5, head_batch=True)
    assert _rel(o_hb, o_ok) <= 0.02
    assert _rel(o_hb, o_swap) > 0.2  # must NOT match a swapped-band layout


@CUDA
def test_head_batch_rejects_unsupported_grp():
    # grp not in {2,4} when forced must raise (thread/register cap), not silently mis-run.
    from flash_qla import recurrent_gated_delta_rule
    q, k, v, g, beta = _mk(1, 4, 1, 8)  # grp=8
    with pytest.raises(AssertionError):
        recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5, head_batch=True)
