# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""L3: end-to-end multi-card correctness for the ZIGZAG inter-card CP layout.

Companion to ``test_cp_e2e.py`` (contiguous inter / inter+intra). Here rank ``r`` owns
chunk ``r`` and chunk ``NC-1-r`` (NC = 2*world_size) of every document, fed in the agreed
``[early halves d=0..D-1, late halves d=0..D-1]`` local layout. Each rank runs its owned
chunks through the public autograd API under a zigzag CP context and compares the output
and input gradients against the float64 oracle restricted to the global token positions it
owns. The backward comparison genuinely exercises the reverse (suffix) fold: a token's
gradient includes contributions from future chunks that live on other cards.

Stage A scope: pure zigzag-inter (no intra split, no short-conv, no h0/dht). Every document
length must be divisible by NC.

Debug one config::

    pytest tests/test_cp_zigzag_e2e.py -k "single-ws2 and kv" -s
"""
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdr_common import (
    RTOL,  # noqa: F401
    REF_DTYPE,
    CHUNK_SIZE,
    make_inputs,
    assert_relative,
    chunk_gated_delta_rule_fwd_ref,
    chunk_gated_delta_rule_bwd_ref,
)
from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context

# (cu_seqlens, world_size, num_k_heads, num_v_heads, g_scale); every doc length must be
# divisible by NC = 2*world_size. Includes short-doc / large-cp corners that stress the
# per-chunk fold (errors scale with 2*cp/L).
CONFIGS = [
    pytest.param([0, 4096], 2, 8, 8, 1 / 16, id="single-ws2"),
    pytest.param([0, 8192], 4, 8, 8, 1 / 16, id="single-ws4"),
    pytest.param([0, 2048, 6144], 2, 8, 8, 1 / 16, id="two-ws2"),
    pytest.param([0, 1024, 2048], 2, 8, 8, 1.0, id="short-two-ws2"),
    pytest.param([0, 2048], 4, 8, 8, 1.0, id="short-ws4"),
    pytest.param([0, 4096], 2, 8, 16, 1 / 16, id="gva-ws2"),
    pytest.param([0, 3072, 6144, 9216], 2, 4, 4, 1.0, id="three-ws2"),
]

_GRAD_LEAF = {"dq": "q", "dk": "k", "dv": "v", "dg": "g", "db": "beta"}


def _zigzag_owned_idx(cu_list, world_size, rank, device):
    """Global token indices this rank owns, in local ``[early halves, late halves]`` order
    -- the slice to feed the kernel and the restriction to compare the oracle against."""
    NC, D = 2 * world_size, len(cu_list) - 1
    early, late = [], []
    for d in range(D):
        s, e = cu_list[d], cu_list[d + 1]
        L = e - s
        assert L % NC == 0, f"doc {d} length {L} not divisible by NC={NC}"
        cs = L // NC
        early.append(torch.arange(s + rank * cs, s + (rank + 1) * cs))
        late.append(torch.arange(s + (NC - 1 - rank) * cs, s + (NC - rank) * cs))
    return torch.cat(early + late).to(device)


def _init_distributed(rank, world_size, port):
    os.environ.update(
        MASTER_ADDR="localhost", MASTER_PORT=port,
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return dist.group.WORLD


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _free_port() -> str:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def _reference(q, k, v, g, beta, cu_g, scale):
    """Global float64 oracle over the whole sequence (no CP), do = 1 everywhere."""
    q64, k64, v64 = (t.to(REF_DTYPE, copy=True) for t in (q, k, v))
    g64, beta64 = g.to(REF_DTYPE, copy=True), beta.to(REF_DTYPE, copy=True)
    g_r, o_r, A_r, _h, s_r = chunk_gated_delta_rule_fwd_ref(
        q=q64, k=k64, v=v64, g=g64, beta=beta64, scale=scale,
        initial_state=None, cu_seqlens=cu_g, chunk_size=CHUNK_SIZE,
    )
    do = torch.ones_like(o_r)
    dq, dk, dv, db, dg, _dh0 = chunk_gated_delta_rule_bwd_ref(
        q64, k64, v64, g_r, beta64, A_r, scale, None, do, None, cu_g, chunk_size=CHUNK_SIZE,
    )
    return dict(o=o_r, dq=dq, dk=dk, dv=dv, db=db, dg=dg)


def _run_and_check(rank, world_size, cfg, state_v_first, group):
    cu_list, _, Hk, Hv, g_scale = cfg
    T = cu_list[-1]

    (q, k, v, g, beta, _do, _h0r, _dhtr, _h0q, _dhtq, cu_g, scale) = make_inputs(
        1, T, Hk, Hv, varlen=True, cu_seqlens_list=cu_list,
        use_h0=False, state_v_first=state_v_first,
    )
    g = g * g_scale
    ref = _reference(q, k, v, g, beta, cu_g, scale)

    ctx = build_cp_context(
        cu_g, group=group, enable_inter=True, enable_intra=False,
        num_v_heads=Hv, is_train=True, zigzag=True,
    )

    idx = _zigzag_owned_idx(cu_list, world_size, rank, q.device)
    leaves = {
        name: t[:, idx].detach().clone().requires_grad_(True)
        for name, t in dict(q=q, k=k, v=v, g=g, beta=beta).items()
    }

    o, _ = chunk_gated_delta_rule(
        leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
        scale=scale, initial_state=None, output_final_state=False,
        cu_seqlens=None, state_v_first=state_v_first, cp_context=ctx,
    )
    o.float().sum().backward()
    dist.barrier()

    tag = f"[rank{rank}/{world_size} zigzag]"
    assert_relative(o, ref["o"][:, idx], f"{tag} o")
    for gname, lname in _GRAD_LEAF.items():
        assert_relative(leaves[lname].grad, ref[gname][:, idx], f"{tag} {gname}")


def _cp_worker(rank, world_size, cfg, state_v_first, port):
    try:
        group = _init_distributed(rank, world_size, port)
        _run_and_check(rank, world_size, cfg, state_v_first, group)
    finally:
        _cleanup_distributed()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.needs_bwd
@pytest.mark.multigpu
@pytest.mark.parametrize("cu_list, world_size, num_k_heads, num_v_heads, g_scale", CONFIGS)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_cp_zigzag(cu_list, world_size, num_k_heads, num_v_heads, g_scale, state_v_first):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs >= {world_size} GPUs, found {torch.cuda.device_count()}")

    cfg = (cu_list, world_size, num_k_heads, num_v_heads, g_scale)
    mp.start_processes(
        _cp_worker,
        args=(world_size, cfg, state_v_first, _free_port()),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
