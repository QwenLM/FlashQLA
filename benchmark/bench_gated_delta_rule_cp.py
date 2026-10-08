# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Context-parallel benchmark for FlashQLA Gated Delta Rule.
#
# Usage::
#
#     torchrun --nproc_per_node=8 benchmark/bench_gated_delta_rule_cp.py --mode infer
#     torchrun --nproc_per_node=8 benchmark/bench_gated_delta_rule_cp.py --mode train

import argparse
import gc
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

import tilelang

from flash_qla import chunk_gated_delta_rule as qla
from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE, build_cp_context
from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector
from flash_qla.ops.gated_delta_rule.chunk import (
    kkt_solve,
    fused_gdr_fwd,
    fused_gdr_bwd,
    fused_gdr_h,
    fused_gdr_dh,
    correct_initial_states,
    correct_terminal_states,
    get_warmup_chunks_bidi,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.comm import (
    all_gather_into_tensor,
    pack_hm,
    unpack_hm,
)
from flash_qla.utils import l2norm

try:
    import fla as _fla
    from fla.ops.cp import build_cp_context as build_cp_context_fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_kcp_fn
    HAS_FLA = True
except ImportError:
    HAS_FLA = False

HEAD_DIM = 128

MAX_CARD_LOAD = 131072 * 64


@dataclass
class ModelConfig:
    label: str
    h_qk: int
    h_v: int


@dataclass
class SeqLenConfig:
    label: str
    #: global cu_seqlens offsets; [0, T] is one sequence spanning every card.
    cu_seqlens: List[int]


# h_v from 16 to 64, GQA and symmetric.
MODEL_CONFIGS = [
    ModelConfig("h16/64 GQA", 16, 64),
    ModelConfig("h16/32 GQA", 16, 32),
    ModelConfig("h16/16", 16, 16),
    ModelConfig("h32/32", 32, 32),
    ModelConfig("h64/64", 64, 64),
]

# global T from 16k to 1M, one sequence spanning every card.
SEQLEN_CONFIGS = [
    SeqLenConfig("1x32768", [0, 32768]),
    SeqLenConfig("1x65536", [0, 65536]),
    SeqLenConfig("1x131072", [0, 131072]),
    SeqLenConfig("1x262144", [0, 262144]),
    SeqLenConfig("1x524288", [0, 524288]),
    SeqLenConfig("1x1048576", [0, 1048576]),
]


def cleanup_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# naive_dual composition
# ---------------------------------------------------------------------------

@dataclass
class NaiveFwd:
    g_c: torch.Tensor
    a: torch.Tensor
    o: torch.Tensor
    cp_h0: Optional[torch.Tensor] = None
    mt: Optional[torch.Tensor] = None
    mt_card: Optional[torch.Tensor] = None
    num_warmup_bwd: Optional[torch.Tensor] = None
    fallback_bwd: Optional[torch.Tensor] = None


def _inter_warmup_counts(ctx, num_v_heads: int, device, *, reverse: bool):
    cu_cpu = ctx.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    counts = torch.zeros((N, num_v_heads), dtype=ctx.cu_seqlens.dtype, device=device)

    def full(i):
        return (cu_cpu[i + 1] - cu_cpu[i] + CHUNK_SIZE - 1) // CHUNK_SIZE

    if reverse:
        counts[0, :] = full(0)
    else:
        counts[-1, :] = full(N - 1)
        if N > 1:
            counts[0, :] = full(0)
    return counts


def naive_fwd(d: dict, ctx, scale: float, *, initial_state=None,
              state_v_first: bool = False) -> NaiveFwd:
    q, k, v, g, beta = d["q"], d["k"], d["v"], d["g"], d["beta"]
    assert ctx.is_inter and ctx.is_intra
    Hv = v.shape[2]
    cu = ctx.cu_seqlens
    cp_cu = ctx.intra_cp_cu_seqlens

    g_c = chunk_local_cumsum(g, cu_seqlens=cu, chunk_size=CHUNK_SIZE)
    a = kkt_solve(k=k, b=beta, cu_seqlens=cu, chunk_size=CHUNK_SIZE)

    # No boundary forcing: the card state comes from the card scan below, so this scan
    # is free to truncate on decay like pure intra does.
    num_warmup, num_warmup_bwd, fallback, fallback_bwd = get_warmup_chunks_bidi(
        g=g_c, cu_seqlens=cp_cu, ht_mask_fwd=ctx.ht_mask,
        ht_mask_bwd=ctx.ht_mask_bwd, chunk_size=CHUNK_SIZE,
    )

    # level 1: inter -- a card-serial state scan, then gather + correct.
    _, ht_card, mt_card = fused_gdr_h(
        k=k, v=v, a=a, g=g_c, b=beta, initial_state=initial_state,
        output_final_state=True, output_h=False, cu_seqlens=cu,
        num_warmup_chunks=_inter_warmup_counts(ctx, Hv, k.device, reverse=False),
        state_v_first=state_v_first,
    )
    hm = pack_hm(ht_card[-1], mt_card[-1])
    ag_hm, _ = all_gather_into_tensor(hm, group=ctx.group)
    rank = dist.get_rank(group=ctx.group)
    h_buf, m_buf = unpack_hm(
        ag_hm[rank - ctx.pre_num_ranks: rank + 1], ht_card[-1], mt_card[-1]
    )
    card_h0 = initial_state
    if not ctx.is_first_rank:
        seq_map_scan, fb_scan = ctx.get_fwd_scan_tensors(Hv)
        cp_h0_all = correct_initial_states(
            raw_h0=None, ht_buffer=h_buf, mt_buffer=m_buf,
            fallback_mask=fb_scan, seq_map_r2c=seq_map_scan,
            state_v_first=state_v_first,
        )
        card_h0 = (
            torch.zeros(ht_card.shape, dtype=torch.float32, device=ht_card.device)
            if initial_state is None else initial_state.clone()
        )
        card_h0[0] = cp_h0_all[h_buf.shape[0] - 1]

    # level 2: intra -- a second state scan, at partition granularity.
    _, ht, mt = fused_gdr_h(
        k=k, v=v, a=a, g=g_c, b=beta, initial_state=None,
        output_final_state=True, output_h=False, cu_seqlens=cp_cu,
        num_warmup_chunks=num_warmup, state_v_first=state_v_first,
    )
    cp_h0 = correct_initial_states(
        raw_h0=card_h0, ht_buffer=ht, mt_buffer=mt,
        fallback_mask=fallback, seq_map_r2c=ctx.seq_map_r2c,
        state_v_first=state_v_first,
    )

    o, _, _ = fused_gdr_fwd(
        q=q, k=k, v=v, a=a, g=g_c, b=beta, scale=scale,
        initial_state=cp_h0, output_final_state=False, output_h=False,
        output_o=True, cu_seqlens=cp_cu,
        cp_seq_map=ctx.seq_map_c2r, raw_cu_seqlens=cu,
        state_v_first=state_v_first,
    )
    return NaiveFwd(g_c=g_c, a=a, o=o, cp_h0=cp_h0, mt=mt, mt_card=mt_card,
                    num_warmup_bwd=num_warmup_bwd, fallback_bwd=fallback_bwd)


def naive_bwd(d: dict, ctx, scale: float, fwd: NaiveFwd, *, do: torch.Tensor,
              dht=None, initial_state=None, state_v_first: bool = False) -> dict:
    q, k, v, beta = d["q"], d["k"], d["v"], d["beta"]
    g_c, a = fwd.g_c, fwd.a
    Hg, Hv = k.shape[2], v.shape[2]
    cu = ctx.cu_seqlens
    cp_cu = ctx.intra_cp_cu_seqlens

    # level 1: inter -- card-granularity reverse scan.
    _, dh_card = fused_gdr_dh(
        q=q, k=k, a=a, g=g_c, b=beta, do=do, dht=dht,
        output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cu,
        num_warmup_chunks=_inter_warmup_counts(ctx, Hv, k.device, reverse=True),
        state_v_first=state_v_first,
    )
    N = dh_card.shape[0]
    hm = pack_hm(dh_card[0], fwd.mt_card[0])
    ag_hm, _ = all_gather_into_tensor(hm, group=ctx.group)
    rank = dist.get_rank(group=ctx.group)
    dh_buf, m_buf = unpack_hm(
        ag_hm[rank: rank + 1 + ctx.post_num_ranks], dh_card[0], fwd.mt_card[0]
    )
    if dht is None and ctx.is_last_rank:
        card_dht = None
    else:
        card_dht = (
            torch.zeros(dh_card.shape, dtype=torch.float32, device=dh_card.device)
            if dht is None else dht.clone()
        )
        if not ctx.is_last_rank:
            seq_map_scan, fb_scan = ctx.get_bwd_scan_tensors(Hv)
            cp_dht_all = correct_terminal_states(
                raw_dht=None, dht_buffer=dh_buf, mt_buffer=m_buf,
                fallback_mask=fb_scan, seq_map_r2c=seq_map_scan,
                state_v_first=state_v_first,
            )
            card_dht[N - 1] = cp_dht_all[0]

    # level 2: intra.
    _, dh = fused_gdr_dh(
        q=q, k=k, a=a, g=g_c, b=beta, do=do, dht=None,
        output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cp_cu,
        num_warmup_chunks=fwd.num_warmup_bwd, state_v_first=state_v_first,
    )
    cp_dht = correct_terminal_states(
        raw_dht=card_dht, dht_buffer=dh, mt_buffer=fwd.mt,
        fallback_mask=fwd.fallback_bwd, seq_map_r2c=ctx.seq_map_r2c,
        state_v_first=state_v_first,
    )

    h, _, _ = fused_gdr_h(
        k=k, v=v, a=a, g=g_c, b=beta, initial_state=fwd.cp_h0,
        output_final_state=False, output_h=True, cu_seqlens=cp_cu,
        state_v_first=state_v_first,
    )
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q, k=k, v=v, a=a, g=g_c, b=beta, do=do, dht=cp_dht, h=h,
        scale=scale, cu_seqlens=cp_cu, state_v_first=state_v_first,
    )
    if dh0 is None or initial_state is None:
        dh0 = None
    else:
        dh0 = dh0[ctx.seq_map_r2c[:-1].long()]
        if not ctx.is_first_rank:
            dh0[0] = 0
    if Hg < Hv:
        dq = group_reduce_vector(dq, Hg)
        dk = group_reduce_vector(dk, Hg)
    dg = chunk_local_cumsum(dg, chunk_size=CHUNK_SIZE, reverse=True, cu_seqlens=cu)
    return dict(dq=dq, dk=dk, dv=dv, dg=dg, db=db, dh0=dh0)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
VARIANT_NAMES = ("fla_kcp", "qla_inter", "naive_dual", "qla_dual")
OURS = "qla_dual"
BASELINES = ("fla_kcp", "qla_inter", "naive_dual")


def build_context(name: str, cu_g: torch.Tensor, *, num_v_heads: int, is_train: bool):
    """The context each variant is measured on. Every variant is inter; the inter split
    derives each card's local view from the global ``cu_g``."""
    if name == "fla_kcp":
        return build_cp_context_fla(cu_g, group=dist.group.WORLD)
    if name == "qla_inter":
        return build_cp_context(
            cu_g, group=dist.group.WORLD,
            enable_inter=True, enable_intra=False,
        )
    # naive and fused share one (True, True) context.
    return build_cp_context(
        cu_g, group=dist.group.WORLD,
        enable_inter=True, enable_intra=True,
        num_v_heads=num_v_heads, is_train=is_train,
        force_intra_cp=True,
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def generate_inputs(cu_seqlens: List[int], h_qk: int, h_v: int, head_dim: int = HEAD_DIM,
                    swa_ratio: float = 0.75, seed: int = 42) -> Dict:
    """Global (all-card) inputs. ``swa_ratio`` of the value heads get g == 0 (a
    sliding-window head to these kernels)."""
    dev = f"cuda:{torch.cuda.current_device()}"
    torch.manual_seed(seed)
    T = cu_seqlens[-1]

    q = l2norm(torch.randn(1, T, h_qk, head_dim, device=dev, dtype=torch.bfloat16))
    k = l2norm(torch.randn(1, T, h_qk, head_dim, device=dev, dtype=torch.bfloat16))
    v = torch.randn(1, T, h_v, head_dim, device=dev, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, h_v, device=dev, dtype=torch.float32)) / 16
    beta = torch.randn(1, T, h_v, device=dev, dtype=torch.float32).sigmoid()

    swa = torch.zeros(h_v, dtype=torch.bool, device=dev)
    swa[:math.ceil(swa_ratio * h_v)] = 1
    swa = swa[torch.randperm(h_v, device=dev)]
    g[:, :, ~swa] = 0.0

    cu_g = torch.tensor(cu_seqlens, device=dev, dtype=torch.int32)
    return dict(q=q, k=k, v=v, g=g, beta=beta, cu_g=cu_g, scale=head_dim ** -0.5)


def slice_card(inputs: dict, rank: int, world_size: int) -> Tuple[dict, int]:
    """Card ``rank`` owns the contiguous token range ``[r*T/W, (r+1)*T/W)``."""
    part = inputs["cu_g"][-1].item() // world_size
    lo, hi = rank * part, (rank + 1) * part
    d = {n: inputs[n][:, lo:hi] for n in ("q", "k", "v", "g", "beta")}
    return d, hi - lo


def _leaves(d: dict) -> dict:
    return {n: t.detach().clone().requires_grad_(True) for n, t in d.items()}


# ---------------------------------------------------------------------------
# End-to-end calls
# ---------------------------------------------------------------------------
def call_e2e(name: str, d: dict, ctx, scale: float):
    if name == "fla_kcp":
        # FLA's KCP rejects initial_state / output_final_state.
        return fla_kcp_fn(d["q"], d["k"], d["v"], d["g"], d["beta"],
                          scale=scale, cp_context=ctx)
    if name == "naive_dual":
        fwd = naive_fwd(d, ctx, scale)
        return fwd.o, None
    return qla(d["q"], d["k"], d["v"], d["g"], d["beta"],
               scale=scale, output_final_state=False, cp_context=ctx)


def make_infer_call(name: str, d: dict, ctx, scale: float) -> Callable:
    """Forward only -- the inference path."""
    if name == "naive_dual":
        return lambda: naive_fwd(d, ctx, scale)
    return lambda: call_e2e(name, d, ctx, scale)


def make_train_call(name: str, d: dict, ctx, scale: float) -> Optional[Callable]:
    """A single fwd+bwd step -- the training path.

    fla_kcp's bwd runs for symmetric heads but has no train column under GQA: FLA's
    tilelang bwd backend rejects GQA (v has more heads than k), and its fallback Triton
    path hard-``raise``s on sm90 + Triton>=3.4.0 (see fla #640). L20X reports sm90, so
    the GQA configs would hit that raise -- skip them rather than error the whole table."""
    if name == "fla_kcp" and d["k"].shape[2] != d["v"].shape[2]:
        return None
    if name == "naive_dual":
        def step():
            fwd = naive_fwd(d, ctx, scale)
            naive_bwd(d, ctx, scale, fwd, do=torch.ones_like(fwd.o))
        return step

    def step():
        leaves = _leaves(d)
        o, _ = call_e2e(name, leaves, ctx, scale)
        o.sum().backward()
    return step


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _reduce_max(ms: float) -> float:
    """Worst-rank time -- a rank-0-only number would miss a boundary card's stall."""
    buf = torch.tensor([ms if ms == ms else -1.0],
                       device=f"cuda:{torch.cuda.current_device()}", dtype=torch.float64)
    dist.all_reduce(buf, op=dist.ReduceOp.MAX)
    val = buf.item()
    return float("nan") if val < 0 else val


def time_call(fn: Optional[Callable], warmup: int, repeats: int, backend: str) -> float:
    if fn is None:
        return float("nan")
    try:
        ms = tilelang.profiler.do_bench(
            fn, _n_warmup=warmup, _n_repeat=repeats, backend=backend
        )
    except RuntimeError as e:
        print(f"\n[WARN] timing failed: {e}", flush=True)
        cleanup_cuda()
        ms = float("nan")
    return _reduce_max(ms)


def bench_config(cu_seqlens: List[int], h_qk: int, h_v: int, *, rank: int,
                 world_size: int, directions: Tuple[str, ...], warmup: int,
                 repeats: int, backend: str, swa_ratio: float, seed: int,
                 is_train: bool) -> Dict[str, Dict[str, float]]:
    """Returns {variant: {"infer": ms, "train": ms, "tokens": n}} (worst-rank ms), timing
    only the requested ``directions``."""
    cleanup_cuda()
    inputs = generate_inputs(cu_seqlens, h_qk, h_v, swa_ratio=swa_ratio, seed=seed)
    d, tokens = slice_card(inputs, rank, world_size)
    scale = inputs["scale"]

    out: Dict[str, Dict[str, float]] = {}
    for name in VARIANT_NAMES:
        if name == "fla_kcp" and not HAS_FLA:
            continue
        ctx = build_context(name, inputs["cu_g"], num_v_heads=h_v, is_train=is_train)
        # An inter+intra context whose intra split declined is measuring pure inter --
        # not the comparison this column claims. force_intra_cp=True prevents it, but
        # guard anyway.
        if name in ("naive_dual", "qla_dual") and not ctx.is_intra:
            if rank == 0:
                print(f"[warn] {name}: intra split declined; skipping", flush=True)
            continue
        rec = {"tokens": float(tokens)}
        if "infer" in directions:
            rec["infer"] = time_call(make_infer_call(name, d, ctx, scale),
                                     warmup, repeats, backend)
        if "train" in directions:
            rec["train"] = time_call(make_train_call(name, d, ctx, scale),
                                     warmup, repeats, backend)
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_time(ms: float) -> str:
    return "     N/A  " if ms != ms else f"{ms:>8.3f}ms"


def fmt_ratio(base: float, ours: float) -> str:
    if base != base or ours != ours or ours == 0:
        return "   N/A  "
    return f"{base / ours:>6.2f}x"


# How each measured direction is labelled in the report.
DIRECTION_LABELS = {"infer": "INFER  (fwd)", "train": "TRAIN  (fwd + bwd)"}


def print_header(direction: str):
    hdr = (
        f"{'Model':<12} {'Seqlens':<14} {'h_qk':>4} {'h_v':>4} {'tok/card':>8}    "
        f"{'fla_kcp':>10} {'qla_inter':>10} {'naive_dual':>10} {'qla_dual(ours)':>11}   "
        f"{'vs_fla':>7} {'vs_inter':>8} {'vs_naive':>8}"
    )
    label = DIRECTION_LABELS.get(direction, direction.upper())
    print(f"\n>>> {label}  (ratio = baseline / ours; > 1 means ours is faster)")
    print(hdr)
    print("-" * len(hdr))


def print_row(model: ModelConfig, sl: SeqLenConfig, res: Dict[str, Dict[str, float]],
              direction: str):
    def get(name):
        return res.get(name, {}).get(direction, float("nan"))

    fla, inter = get("fla_kcp"), get("qla_inter")
    naive, ours = get("naive_dual"), get(OURS)
    tokens = res.get(OURS, res.get("qla_inter", {})).get("tokens", float("nan"))
    print(
        f"{model.label:<12} {sl.label:<14} {model.h_qk:>4} {model.h_v:>4} "
        f"{tokens:>8.0f}    "
        f"{fmt_time(fla)} {fmt_time(inter)} {fmt_time(naive)} {fmt_time(ours)}   "
        f"{fmt_ratio(fla, ours)} {fmt_ratio(inter, ours)} {fmt_ratio(naive, ours)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def card_load_ok(cu_seqlens: List[int], h_v: int, world_size: int, rank: int) -> bool:
    """Skip before allocating: a config that would OOM one card hangs the group at the
    next collective while the survivors wait."""
    load = (cu_seqlens[-1] // world_size) * h_v
    if load > MAX_CARD_LOAD:
        if rank == 0:
            print(f"[skip] T={cu_seqlens[-1]} h_v={h_v}: card load {load} > "
                  f"MAX_CARD_LOAD={MAX_CARD_LOAD}", flush=True)
        return False
    return True


def valid(cu_seqlens: List[int], world_size: int, rank: int) -> bool:
    T = cu_seqlens[-1]
    if T % world_size != 0:
        if rank == 0:
            print(f"[skip] global T={T} not divisible by world_size={world_size}")
        return False
    if (T // world_size) % CHUNK_SIZE != 0:
        if rank == 0:
            print(f"[skip] per-card T={T // world_size} not a multiple of "
                  f"chunk_size={CHUNK_SIZE}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="CP benchmark: fla_kcp / qla_inter / naive vs fused qla_dual")
    parser.add_argument("--mode", choices=["infer", "train"], default="train",
                        help="infer = forward only; train = forward + backward")
    parser.add_argument("--warmup", type=int, default=25,
                        help="fixed warmup iterations per timed step (same on every rank)")
    parser.add_argument("--repeats", type=int, default=100,
                        help="fixed timed iterations per step (same on every rank)")
    parser.add_argument("--backend", choices=["event", "cudagraph"], default="event",
                        help="event is safest with NCCL collectives inside the step")
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # train adds the fwd+bwd (train) table on top of infer, and drives the intra decision's
    # cost regime.
    is_train = args.mode == "train"

    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        print("CP benchmark needs world_size >= 2 -- launch with "
              "`torchrun --nproc_per_node=<N>` (N >= 2). Every variant is an inter "
              "variant; there is no single-GPU mode.")
        return

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if rank == 0:
        gpu = torch.cuda.get_device_properties(0).name
        fla_ver = getattr(_fla, "__version__", "?") if HAS_FLA else "not installed"
        print(f"GPU: {gpu} x {world_size}   |   fla: {fla_ver}   |   "
              f"torch: {torch.__version__}   |   chunk_size: {CHUNK_SIZE}")
        print(f"Config: warmup={args.warmup} repeats={args.repeats} "
              f"backend={args.backend} mode={args.mode}")
        print("=" * 120)

    models = MODEL_CONFIGS
    seqlens = SEQLEN_CONFIGS

    def runnable(sl, model):
        return (valid(sl.cu_seqlens, world_size, rank)
                and card_load_ok(sl.cu_seqlens, model.h_v, world_size, rank))

    # One table per direction, each row printed as soon as it is timed. Every rank runs
    # the identical (direction, model, seqlen) sequence so the CP collectives stay in
    # lockstep; only rank 0 prints.
    for direction in (("infer", "train") if is_train else ("infer",)):
        if rank == 0:
            print_header(direction)
        prev = None
        for model in models:
            if prev is not None and prev != model.label and rank == 0:
                print()
            prev = model.label
            for sl in seqlens:
                if not runnable(sl, model):
                    continue
                res = bench_config(
                    sl.cu_seqlens, model.h_qk, model.h_v, rank=rank,
                    world_size=world_size, directions=(direction,),
                    warmup=args.warmup, repeats=args.repeats, backend=args.backend,
                    swa_ratio=args.swa_ratio, seed=args.seed, is_train=is_train,
                )
                if rank == 0:
                    print_row(model, sl, res, direction)
                cleanup_cuda()

    if rank == 0:
        print("\nBenchmark finished.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
