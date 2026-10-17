# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import os
import sys

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context
from flash_qla.ops.gated_delta_rule.chunk.cp.context import (
    _build_intra_cp_context,
    _calc_inter_cp_seqs,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.comm import pack_hm, unpack_hm

from ref_context import ref_intra_context, ref_inter_context

DEV = "cuda:0"


def _assert_intra(ctx, ref):
    assert ctx.intra_cp_cu_seqlens.tolist() == ref["cp_cu"]
    assert ctx.seq_map_r2c.tolist() == ref["r2c"]
    assert ctx.seq_map_c2r.tolist() == ref["c2r"]
    assert ctx.ht_mask.tolist() == ref["ht"]
    assert ctx.ht_mask_bwd.tolist() == ref["ht_bwd"]


def _assert_inter(ctx, ref):
    assert ctx.cu_seqlens_cpu.tolist() == ref["local_cu"]
    assert ctx.pre_num_ranks == ref["pre_num_ranks"]
    assert ctx.post_num_ranks == ref["post_num_ranks"]
    assert ctx.is_first_rank == ref["is_first_rank"]
    assert ctx.is_last_rank == ref["is_last_rank"]
    assert ctx.pre_num_conv_tokens == ref["pre_num_conv_tokens"]


# ===========================================================================
# intra-card context
# ===========================================================================
_INTRA_LAYOUTS = [
    [0, 8192],
    [0, 2048, 8192],
    [0, 1024, 1536, 8192],
    [0, 4096, 8192],
    [0, 2048, 4096, 6144, 8192],
    [0, 1031, 8192],                       # ragged: boundary off the chunk grid
    [0, 64, 8192],                         # single-chunk leading sequence
    [0, 8192, 8256],                       # single-chunk trailing sequence
    [0, 100, 8192],                        # sub-chunk leading sequence
]


@pytest.mark.gpu
@pytest.mark.parametrize("cu_list", _INTRA_LAYOUTS, ids=lambda c: f"n{len(c) - 1}-T{c[-1]}")
@pytest.mark.parametrize("max_local_chunks", [4, 5, 7, 12])
def test_intra_context_matches_ref(cu_list, max_local_chunks):
    cu = torch.tensor(cu_list, dtype=torch.int32, device=DEV)
    num_chunks = [-(-(cu_list[i + 1] - cu_list[i]) // CHUNK_SIZE) for i in range(len(cu_list) - 1)]
    ctx = _build_intra_cp_context(cu, CHUNK_SIZE, num_chunks, max_local_chunks)
    _assert_intra(ctx, ref_intra_context(cu_list, CHUNK_SIZE, max_local_chunks))
    assert ctx.is_intra and not ctx.is_inter
    assert not ctx.inter_intra_requested
    assert ctx.cu_seqlens.tolist() == cu_list


# ===========================================================================
# inter-card context
# ===========================================================================
# (cu_list, world_size): explicit global partitions. card edge = total / world_size,
# so each id names where the seq boundaries fall relative to the card edges.
_INTER_CASES = [
    pytest.param([0, 2048], 2, id="single-ws2"),                    # one seq spanning all cards
    pytest.param([0, 3072], 3, id="single-ws3"),
    pytest.param([0, 8192], 4, id="single-ws4"),
    pytest.param([0, 1024, 2048, 3072], 3, id="per-card-ws3"),      # boundary on every card edge
    pytest.param([0, 512, 4096], 2, id="head-inside-card0-ws2"),    # first boundary inside card 0
    pytest.param([0, 512, 700, 2048], 2, id="mid-inside-card0-ws2"),  # short seq wholly inside card 0
    pytest.param([0, 1024, 3072, 4096], 2, id="tail-inside-card1-ws2"),  # short tail seq in last card
    pytest.param([0, 519, 3072], 3, id="odd-boundary-ws3"),         # boundary off any round grid
]


@pytest.mark.gpu
@pytest.mark.parametrize("cu_list, world_size", _INTER_CASES)
def test_inter_context_matches_ref(cu_list, world_size):
    cu = torch.tensor(cu_list, dtype=torch.int32, device=DEV)
    for rank in range(world_size):
        ctx = _calc_inter_cp_seqs(cu, world_size=world_size, rank=rank, group=None)
        _assert_inter(ctx, ref_inter_context(cu_list, world_size, rank))
        assert ctx.is_inter and not ctx.is_intra
        assert not ctx.inter_intra_requested


@pytest.mark.parametrize("total, world_size", [(1000, 3), (1000, 7), (100, 3)])
def test_inter_cp_indivisible_raises(total, world_size):
    assert total % world_size != 0, "test setup: total must be indivisible"
    cu = torch.tensor([0, total], dtype=torch.int32)
    with pytest.raises(AssertionError, match="divisible by"):
        _calc_inter_cp_seqs(cu, world_size=world_size, rank=0, group=None)


# ===========================================================================
# combined inter+intra context
# ===========================================================================
class _StubGroup:
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank


@pytest.fixture
def stub_group(monkeypatch):
    # Building a combined context asks the group only for its size/rank (no collectives),
    # so a stub covers the merge path in one process. Real multi-proc is in test_cp_e2e.py.
    def make(world_size: int, rank: int) -> _StubGroup:
        group = _StubGroup(world_size, rank)
        real_ws, real_rank = dist.get_world_size, dist.get_rank
        monkeypatch.setattr(
            dist, "get_world_size",
            lambda group=None: group.world_size if isinstance(group, _StubGroup) else real_ws(group),
        )
        monkeypatch.setattr(
            dist, "get_rank",
            lambda group=None: group.rank if isinstance(group, _StubGroup) else real_rank(group),
        )
        return group

    return make


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_inter_intra_matches_both_refs(stub_group, world_size):
    # The merge must carry the inter topology *and* an intra split built over the
    # card-local sequence, and copy_for_backward must preserve both.
    total = world_size * 32768
    cu_list = [0, total]
    cu = torch.tensor(cu_list, dtype=torch.int32, device=DEV)

    for rank in range(world_size):
        group = stub_group(world_size, rank)
        ctx = build_cp_context(
            cu, enable_inter=True, enable_intra=True, group=group,
            num_v_heads=4, chunk_size=CHUNK_SIZE, force_intra_cp=True,
        )
        assert ctx.is_inter and ctx.is_intra and ctx.group is group
        assert ctx.inter_intra_requested

        inter_ref = ref_inter_context(cu_list, world_size, rank)
        _assert_inter(ctx, inter_ref)

        # intra split refines the local view; recover the split's max_local_chunks
        # from the first full segment, then compare the whole partition.
        local_cu = ctx.cu_seqlens.tolist()
        cp_cu = ctx.intra_cp_cu_seqlens.tolist()
        max_local_chunks = (cp_cu[1] - cp_cu[0]) // CHUNK_SIZE
        intra_ref = ref_intra_context(local_cu, CHUNK_SIZE, max_local_chunks)
        _assert_intra(ctx, intra_ref)

        copied = ctx.copy_for_backward()
        assert copied.group is group
        assert copied.inter_intra_requested
        _assert_inter(copied, inter_ref)
        _assert_intra(copied, intra_ref)


@pytest.mark.gpu
def test_mixed_dual_backward_rejected_before_work(stub_group, monkeypatch):
    from flash_qla.ops.gated_delta_rule import chunk
    from flash_qla.ops.gated_delta_rule.chunk.cp import context
    from flash_qla.ops.gated_delta_rule.chunk.cp.preprocess import (
        _assert_inter_intra_supported, cp_preprocess_bwd,
    )

    # Fix the local decisions so this regression does not depend on GPU/model tuning.
    decisions = iter([(False, None), (True, 4)])
    monkeypatch.setattr(context, "AUTOCP_MODEL", True)
    monkeypatch.setattr(context, "autocp_decide", lambda **kwargs: next(decisions))
    monkeypatch.setattr(chunk, "DUAL_CP_BWD_SUPPORTED", False)
    cu = torch.tensor(list(range(0, 1025, 32)) + [2048], dtype=torch.int32, device=DEV)
    for rank in range(2):
        ctx = build_cp_context(
            cu, enable_inter=True, enable_intra=True, group=stub_group(2, rank),
            num_v_heads=4, chunk_size=32, is_train=True,
        )
        assert ctx.is_inter and ctx.is_intra == bool(rank)
        for bwd_ctx in (ctx, ctx.copy_for_backward()):
            # None inputs ensure rejection precedes tensor work and any collective.
            with pytest.raises(NotImplementedError, match="Dual-CP.*Backward"):
                cp_preprocess_bwd(
                    bwd_ctx, q=None, k=None, v=None, a=None, g=None, beta=None,
                    do=None, dht=None, scale=1.0, initial_state=None,
                    state_v_first=False, cp_cache=None,
                )

    # The capability restriction must still allow explicitly requested pure modes.
    _assert_inter_intra_supported(True, False, backward=True)
    _assert_inter_intra_supported(False, True, backward=True)


# ===========================================================================
# (h, M) pack & unpack
# ===========================================================================
_HM_DTYPES = [
    (torch.bfloat16, torch.bfloat16),   # fwd, pure inter: `ht` / `mt`, both k.dtype
    (torch.float32, torch.float32),     # fwd+bwd, inter+intra: aggregate outputs
    (torch.float32, torch.bfloat16),    # bwd, pure inter: fp32 `dh` + bf16 `mt`
]


@pytest.mark.gpu
@pytest.mark.parametrize("h_dtype, m_dtype", _HM_DTYPES,
                         ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_pack_unpack_hm(h_dtype, m_dtype, state_v_first):
    H, K, V = 4, 8, 8
    n_cards = 3
    h = torch.randn((H, V, K) if state_v_first else (H, K, V),
                    device=DEV).to(h_dtype)
    m = torch.randn((H, K, K), device=DEV).to(m_dtype)

    packed = pack_hm(h, m)
    assert packed.dtype is torch.uint8
    assert packed.numel() == h.numel() * h.element_size() + m.numel() * m.element_size()

    gathered = torch.stack([packed] * n_cards)      # stands in for the all_gather
    h_buf, m_buf = unpack_hm(gathered, h, m)

    assert h_buf.shape == (n_cards, *h.shape) and h_buf.dtype is h_dtype
    assert m_buf.shape == (n_cards, *m.shape) and m_buf.dtype is m_dtype
    assert h_buf.is_contiguous() and m_buf.is_contiguous()
    for i in range(n_cards):
        assert torch.equal(h_buf[i], h)
        assert torch.equal(m_buf[i], m)
