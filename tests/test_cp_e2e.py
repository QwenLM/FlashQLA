# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""L3: end-to-end multi-card CP correctness -- the multi-card extension of
``test_gdr_unit.py``.

Each config is one global varlen sequence with an explicit ``(cu_seqlens, world_size)``;
the total is divisible by the world size (inter-card CP requires it) and the interior
boundaries fall at chosen offsets relative to the card edge at ``total / world_size``.
One process per rank runs its token slice through the public autograd API under a built
CP context, then compares against the float64 reference (:mod:`ref_gdr`, the same oracle
``test_gdr_unit.py`` uses) restricted to what that rank owns.

Single-card modes (``none`` / ``intra``) are *not* here: ``none`` is ``test_gdr_unit.py``
and forced ``intra`` is ``test_cp_features.py``; this file is only about the cards talking
to each other, so it needs >= 2 GPUs to run at all.

Debug one config::

    pytest tests/test_cp_e2e.py -k "single-ws2 and inter and kv" -s
"""
from contextlib import nullcontext
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdr_common import (
    RTOL,  # noqa: F401  (re-exported so a debug run can print the tolerance)
    REF_DTYPE,
    CHUNK_SIZE,
    make_inputs,
    assert_relative,
    chunk_gated_delta_rule_fwd_ref,
    chunk_gated_delta_rule_bwd_ref,
)
from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context

# (cu_seqlens, world_size, num_k_heads, num_v_heads, g_scale, use_h0, use_dht)
# Interior boundaries are placed relative to the card edge at total/world_size.
CONFIGS = [
    pytest.param([0, 4096], 2, 8, 8, 1 / 16, False, False, id="single-ws2"),
    pytest.param([0, 6144], 3, 8, 8, 1 / 16, False, False, id="single-ws3"),
    pytest.param([0, 8192], 4, 8, 8, 1 / 16, False, False, id="single-ws4"),
    pytest.param([0, 4096], 2, 8, 8, 0.0, True, True, id="single-h0-ws2"),
    pytest.param([0, 6144], 3, 8, 8, 0.0, True, True, id="single-h0-ws3"),
    pytest.param([0, 8192], 4, 8, 8, 0.0, True, True, id="single-h0-ws4"),
    pytest.param([0, 2048, 4096, 6144], 3, 8, 8, 1 / 16, True, True, id="per-card-ws3"),
    pytest.param([0, 1024, 4096], 2, 8, 8, 1 / 16, False, False, id="offset-ws2"),
    pytest.param([0, 1024, 1536, 4096], 2, 8, 8, 1 / 16, True, True, id="three-ws2"),
    pytest.param([0, 3072, 4096], 2, 8, 8, 1 / 16, False, False, id="tail-ws2"),
    pytest.param([0, 1031, 4096], 2, 8, 8, 1.0, False, False, id="offset-ragged-ws2"),
    pytest.param([0, 4096], 2, 8, 16, 1 / 16, False, False, id="gva-ws2"),
    pytest.param([0, 519, 4096], 2, 4, 4, 1.0, True, True, id="allflags-ws2"),
]

# Both modes are inter-card; inter_intra additionally splits each card's local sequence
# for intra-card CP (needs the combined-CP capability, gated in conftest). The bool is
# `enable_intra`, threaded straight into the context builder.
MODES = [
    pytest.param(False, id="inter"),
    pytest.param(True, id="inter_intra", marks=pytest.mark.cp_inter_intra),
]

_GRAD_LEAF = {"dq": "q", "dk": "k", "dv": "v", "dg": "g", "db": "beta"}


def _make_ctx(cu_g, num_v_heads, group, enable_intra, *, is_train=True):
    return build_cp_context(
        cu_g, enable_inter=True, enable_intra=enable_intra, group=group,
        num_v_heads=num_v_heads, chunk_size=CHUNK_SIZE,
        is_train=is_train, force_intra_cp=enable_intra,
    )


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


def _reference(q, k, v, g, beta, h0_ref, cu_g, scale, use_dht, *, forward_only=False):
    """Global float64 oracle: whole sequence, no CP, differentiating the same loss the
    CP run uses (do = 1 everywhere; dht = 1 on the final state iff ``use_dht``)."""
    q64, k64, v64 = (t.to(REF_DTYPE, copy=True) for t in (q, k, v))
    g64, beta64 = g.to(REF_DTYPE, copy=True), beta.to(REF_DTYPE, copy=True)
    h0_64 = h0_ref.to(REF_DTYPE, copy=True) if h0_ref is not None else None

    g_r, o_r, A_r, _h, s_r = chunk_gated_delta_rule_fwd_ref(
        q=q64, k=k64, v=v64, g=g64, beta=beta64, scale=scale,
        initial_state=h0_64, cu_seqlens=cu_g, chunk_size=CHUNK_SIZE,
    )
    if forward_only:
        return dict(o=o_r, s=s_r)
    do = torch.ones_like(o_r)
    dht = torch.ones_like(s_r) if use_dht else None
    dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd_ref(
        q64, k64, v64, g_r, beta64, A_r, scale, h0_64, do, dht, cu_g, chunk_size=CHUNK_SIZE,
    )
    return dict(o=o_r, s=s_r, dq=dq, dk=dk, dv=dv, db=db, dg=dg, dh0=dh0)


def _run_and_check(rank, world_size, enable_intra, cfg, state_v_first, group,
                   *, forward_only=False):
    cu_list, _, Hk, Hv, g_scale, use_h0, use_dht = cfg
    T = cu_list[-1]

    # One deterministic global input set, shared by every rank and the oracle.
    (q, k, v, g, beta, _do, h0_ref, _dhtr, h0_qla, _dhtq, cu_g, scale) = make_inputs(
        1, T, Hk, Hv, varlen=True, cu_seqlens_list=cu_list,
        use_h0=use_h0, state_v_first=state_v_first,
    )
    g = g * g_scale
    ref = _reference(q, k, v, g, beta, h0_ref, cu_g, scale, use_dht,
                     forward_only=forward_only)

    ctx = _make_ctx(cu_g, Hv, group, enable_intra, is_train=not forward_only)
    if forward_only:
        # Do not silently test pure inter when a Dual-CP test failed to split.
        assert ctx.is_inter and ctx.is_intra
        assert ctx.intra_cp_cu_seqlens.numel() > ctx.cu_seqlens.numel()

    part = T // world_size
    lo, hi = rank * part, (rank + 1) * part
    start_seq = int(torch.searchsorted(cu_g[1:], lo, side="right").item())
    n_local = ctx.num_seqs

    leaves = {
        name: t[:, lo:hi].detach().clone().requires_grad_(not forward_only)
        for name, t in dict(q=q, k=k, v=v, g=g, beta=beta).items()
    }
    h0_leaf = None
    if h0_qla is not None:
        h0_leaf = h0_qla[start_seq: start_seq + n_local].detach().clone().requires_grad_(not forward_only)

    with torch.no_grad() if forward_only else nullcontext():
        o, final_state = chunk_gated_delta_rule(
            leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
            scale=scale, initial_state=h0_leaf, output_final_state=True,
            cu_seqlens=None, state_v_first=state_v_first, cp_context=ctx,
        )
    if not forward_only:
        loss = o.float().sum()
        if use_dht and final_state is not None:
            loss = loss + final_state.float().sum()
        loss.backward()

    # All collectives are done; sync so a failed assert on one rank cannot hang peers.
    dist.barrier()

    tag = f"[rank{rank}/{world_size} {'inter_intra' if enable_intra else 'inter'}]"
    assert_relative(o, ref["o"][:, lo:hi], f"{tag} o")
    if not forward_only:
        for gname, lname in _GRAD_LEAF.items():
            assert_relative(leaves[lname].grad, ref[gname][:, lo:hi], f"{tag} {gname}")

    def _fix(t):
        return t.transpose(-1, -2) if state_v_first else t

    if forward_only:
        assert final_state is not None, "output_final_state=True must return states"
    if final_state is not None:
        # A sequence's final state is the true global one only on the card where the
        # sequence ends; elsewhere it is a partial (card-local) state -- skip it.
        for si in range(n_local):
            gs = start_seq + si
            if lo < cu_list[gs + 1] <= hi:
                assert_relative(_fix(final_state[si]), ref["s"][gs], f"{tag} final_state[{gs}]")

    if h0_leaf is not None and h0_leaf.grad is not None:
        # dh0 for a sequence belongs to the card where the sequence starts.
        for si in range(n_local):
            gs = start_seq + si
            if lo <= cu_list[gs] < hi:
                assert_relative(_fix(h0_leaf.grad[si]), ref["dh0"][gs], f"{tag} dh0[{gs}]")


def _cp_worker(rank, world_size, enable_intra, cfg, state_v_first, port):
    try:
        group = _init_distributed(rank, world_size, port)
        _run_and_check(rank, world_size, enable_intra, cfg, state_v_first, group)
    finally:
        _cleanup_distributed()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.needs_bwd
@pytest.mark.multigpu
@pytest.mark.parametrize(
    "cu_list, world_size, num_k_heads, num_v_heads, g_scale, use_h0, use_dht", CONFIGS,
)
@pytest.mark.parametrize("enable_intra", MODES)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_cp(cu_list, world_size, num_k_heads, num_v_heads, g_scale, use_h0, use_dht,
            enable_intra, state_v_first):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs >= {world_size} GPUs, found {torch.cuda.device_count()}")

    cfg = (cu_list, world_size, num_k_heads, num_v_heads, g_scale, use_h0, use_dht)
    mp.start_processes(
        _cp_worker,
        args=(world_size, enable_intra, cfg, state_v_first, _free_port()),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


# Forward-only Dual-CP: each case uses two visible devices. This
# exercises the public API without autograd or the backward reference/kernel.
# Run on two SM120 GPUs:
# CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/test_cp_e2e.py -k dual_cp_forward -v
_FORWARD_CONFIGS = [
    pytest.param([0, 4096], 0.0, id="single-full-fallback"),
    pytest.param([0, 4096], 1 / 16, id="single-decay"),
    pytest.param([0, 2048, 4096], 1 / 16, id="card-aligned"),
    pytest.param([0, 1031, 4096], 1 / 16, id="ragged-boundary"),
    pytest.param([0, 519, 1536, 3079, 4096], 1 / 16, id="multiple-sequences"),
]


def _cp_forward_worker(rank, world_size, cfg, state_v_first, port):
    try:
        group = _init_distributed(rank, world_size, port)
        _run_and_check(rank, world_size, True, cfg, state_v_first, group,
                       forward_only=True)
    finally:
        _cleanup_distributed()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.multigpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("cu_list, g_scale", _FORWARD_CONFIGS)
@pytest.mark.parametrize("use_h0", [False, True], ids=["zero-seed", "raw-h0"])
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_dual_cp_forward(cu_list, g_scale, use_h0, state_v_first):
    # GVA as well as both layouts; compare each rank's output and every sequence
    # final state owned by that rank against the global float64 reference.
    cfg = (cu_list, 2, 2, 4, g_scale, use_h0, False)
    mp.start_processes(
        _cp_forward_worker,
        args=(2, cfg, state_v_first, _free_port()),
        nprocs=2, join=True, start_method="spawn",
    )
