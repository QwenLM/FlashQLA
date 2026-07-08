# tests/test_decode_gdr.py
import pytest
import torch

from ref_gdr import decode_recur
from ref_gdr import chunk_gated_delta_rule_fwd as chunk_fwd_ref
from flash_qla.utils import l2norm

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mk(B, T, Hk, Hv, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    q = l2norm(torch.randn(B, T, Hk, 128, device="cuda", dtype=dtype))
    k = l2norm(torch.randn(B, T, Hk, 128, device="cuda", dtype=dtype))
    v = torch.randn(B, T, Hv, 128, device="cuda", dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, device="cuda")) / 16
    beta = torch.randn(B, T, Hv, device="cuda").sigmoid()
    return q, k, v, g, beta


def _ref_bf16_inputs(B, T, Hk, Hv, seed=0):
    return _mk(B, T, Hk, Hv, seed=seed, dtype=torch.bfloat16)


@CUDA
def test_decode_recur_matches_chunk_at_cs64():
    # A length-L single sequence: decode_recur must match the chunk reference on the L-prefix.
    B, T, H = 1, 50, 8
    q, k, v, g, beta = _mk(B, T, H, H, dtype=torch.float32)
    o_dec, s_dec = decode_recur(q, k, v, g, beta)
    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_fwd_ref(
        q=q.double(), k=k.double(), v=v.double(), g=g.double(), beta=beta.double(),
        scale=128 ** -0.5, initial_state=None, cu_seqlens=None)
    assert (o_dec - o_ref.float()).abs().max() / o_ref.abs().max() < 1e-3
    assert (s_dec - s_ref.float()).abs().max() / s_ref.abs().max() < 1e-3


@CUDA
@pytest.mark.parametrize("D", [1, 8])
@pytest.mark.parametrize("Hk,Hv", [(8, 8), (2, 8), (1, 8)])
@pytest.mark.parametrize("use_h0", [False, True])
def test_kernel_matches_reference(D, Hk, Hv, use_h0):
    from flash_qla import recurrent_gated_delta_rule
    B = 1
    q, k, v, g, beta = _ref_bf16_inputs(B, D, Hk, Hv)
    h0 = (torch.randn(B, Hv, 128, 128, device="cuda", dtype=torch.float32) if use_h0 else None)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, initial_state=h0)
    o_qla, s_qla = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, initial_state=h0, output_final_state=True)
    assert (o_qla.float() - o_ref).abs().max() / o_ref.abs().max().clamp_min(1e-6) <= 0.02
    assert (s_qla - s_ref).abs().max() / s_ref.abs().max().clamp_min(1e-6) <= 0.02


@CUDA
def test_kernel_g0_swa_heads():
    from flash_qla import recurrent_gated_delta_rule
    B, D, H = 1, 8, 8
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    g[:, :, :H // 2] = 0.0  # half the heads have no decay
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_qla, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5)
    assert (o_qla.float() - o_ref).abs().max() / o_ref.abs().max() <= 0.02


@CUDA
def test_kernel_ragged_seqlens():
    from flash_qla import recurrent_gated_delta_rule
    B, D, H = 3, 8, 8
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    seqlens = torch.tensor([1, 5, 8], device="cuda", dtype=torch.int32)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, seqlens=seqlens)
    o_qla, s_qla = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, seqlens=seqlens, output_final_state=True)
    for b in range(B):
        L = int(seqlens[b])
        assert (o_qla[b, :L].float() - o_ref[b, :L]).abs().max() / o_ref[b, :L].abs().max() <= 0.02
        assert (s_qla[b] - s_ref[b]).abs().max() / s_ref[b].abs().max() <= 0.02


@CUDA
def test_kernel_low_occupancy_vsplit():
    from flash_qla import recurrent_gated_delta_rule
    # B*H small => wrapper picks block_DV in {64,32}; result must still match.
    B, D, H = 1, 4, 4
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    o_ref, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_qla, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5)
    assert (o_qla.float() - o_ref).abs().max() / o_ref.abs().max() <= 0.02


@CUDA
@pytest.mark.parametrize("B,H", [(8, 8), (16, 8)])  # B*H=64 -> block_DV=64; 128 -> block_DV=128
def test_kernel_high_occupancy(B, H):
    from flash_qla import recurrent_gated_delta_rule
    D = 4
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_qla, s_qla = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, output_final_state=True)
    assert (o_qla.float() - o_ref).abs().max() / o_ref.abs().max() <= 0.02
    assert (s_qla - s_ref).abs().max() / s_ref.abs().max() <= 0.02


@CUDA
def test_negctrl_step_order():
    # discriminating: kernel matches the post-update reference and DIFFERS from a pre-update one
    from flash_qla import recurrent_gated_delta_rule
    B, D, H = 1, 6, 8
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    o_post, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_pre, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, read_pre_update=True)
    o_qla, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5)
    assert (o_qla.float() - o_post).abs().max() / o_post.abs().max() <= 0.02
    assert (o_qla.float() - o_pre).abs().max() / o_pre.abs().max() > 0.2  # must NOT match wrong order


@CUDA
def test_negctrl_gqa_mapping():
    # discriminating: kernel uses hg=h//grp and must DIFFER from the hg=h%Hk mapping
    from flash_qla import recurrent_gated_delta_rule
    B, D, Hk, Hv = 1, 6, 2, 8
    q, k, v, g, beta = _ref_bf16_inputs(B, D, Hk, Hv)
    o_div, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_mod, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, gqa_mod=True)
    o_qla, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5)
    assert (o_qla.float() - o_div).abs().max() / o_div.abs().max() <= 0.02
    assert (o_qla.float() - o_mod).abs().max() / o_mod.abs().max() > 0.2  # must NOT match wrong GQA


def test_signature_contract():
    import inspect
    from flash_qla import recurrent_gated_delta_rule
    sig = inspect.signature(recurrent_gated_delta_rule)
    for p in ["q", "k", "v", "g", "beta", "scale", "initial_state",
              "output_final_state", "use_qk_l2norm_in_kernel", "seqlens"]:
        assert p in sig.parameters
