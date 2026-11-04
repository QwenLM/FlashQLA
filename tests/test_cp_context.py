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
    _calc_inter_cp_seqs_zigzag,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.comm import pack_hm, pack_states, unpack_hm

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
            cu, group=group,
            enable_inter=True, enable_intra=True,
            num_v_heads=4, is_train=True, force_intra_cp=True,
        )
        assert ctx.is_inter and ctx.is_intra and ctx.group is group

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
        _assert_inter(copied, inter_ref)
        _assert_intra(copied, intra_ref)


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


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_zigzag_fold_slots_on_cpu(world_size, dtype):
    num_docs, nc = 3, 2 * world_size
    cu = torch.tensor([0, nc * 3, nc * 8, nc * 15], dtype=dtype)

    # Start in document/time order, with a unique value for every chunk.
    chronological = torch.arange(num_docs * nc).reshape(num_docs, nc)
    expected_h = chronological.flatten()
    expected_m = expected_h + expected_h.numel()

    # Zigzag assigns columns from opposite ends to each rank; local storage is
    # all documents' early chunks followed by all documents' late chunks.
    h_by_rank = [torch.cat((chronological[:, rank], chronological[:, -1 - rank]))
                 for rank in range(world_size)]
    m_by_rank = [h + expected_h.numel() for h in h_by_rank]

    # Simulate rank-ordered all-gather, using the real packer for packed storage.
    h_split = torch.cat(h_by_rank)
    m_split = torch.cat(m_by_rank)
    packed = torch.cat([pack_states(h, m) for h, m in zip(h_by_rank, m_by_rank)])

    for rank in range(world_size):
        ctx = _calc_inter_cp_seqs_zigzag.__wrapped__(
            cu, world_size=world_size, rank=rank,
        )
        # Both storage layouts must recover the original document/time order.
        assert torch.equal(packed[ctx.fold_h_slot.long()], expected_h)
        assert torch.equal(packed[ctx.fold_m_slot.long()], expected_m)
        assert torch.equal(h_split[ctx.fold_split_slot.long()], expected_h)
        assert torch.equal(m_split[ctx.fold_split_slot.long()], expected_m)

        # Output slots must select this rank's chunks and restore local order.
        local = torch.full_like(h_by_rank[rank], -1)
        owned = ctx.fold_out_slot >= 0
        assert owned.sum().item() == local.numel()
        local[ctx.fold_out_slot[owned].long()] = expected_h[owned]
        assert torch.equal(local, h_by_rank[rank])
