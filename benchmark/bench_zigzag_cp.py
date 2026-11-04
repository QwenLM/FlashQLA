# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Local distributed perf: plain (contiguous) CP vs zigzag CP for the GDN kernels,
# under inter-only and inter+intra, at CP4 / batch=1 over 32k..1M.
#
# One sequence spans every card; each rank holds T/world_size tokens either as one
# contiguous slice (plain) or as its two zigzag chunks (zzcp). We time the public
# autograd path (fwd = infer, fwd+bwd = train) and report the worst-rank ms plus the
# plain/zzcp ratio (>1 means zzcp is slower).
#
# Usage (one command, spawns world_size procs):
#     python benchmark/bench_zigzag_cp.py                 # CP4, all seqlens
#     python benchmark/bench_zigzag_cp.py --world-size 4 --seqlens 32768,65536
#     python benchmark/bench_zigzag_cp.py --h-qk 16 --h-v 16 --iters 20

import argparse
import gc
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flash_qla import chunk_gated_delta_rule as qla
from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE, build_cp_context
from flash_qla.utils import l2norm

HEAD_DIM = 128
SEQLENS = [32768, 65536, 131072, 262144, 524288, 1048576]
# varlen pack: one big doc + these small docs (each a multiple of 2*cp and chunk_size);
# the big doc absorbs the rest so the total stays the swept T.
SMALL_DOCS = [4096, 2048, 2048]


def build_cu_seqlens(T, pack):
    """pack='single' -> [0, T]; pack='varlen' -> [0, big, +small, ...] (big first)."""
    if pack == "single":
        return [0, T]
    big = T - sum(SMALL_DOCS)
    lens = [big, *SMALL_DOCS]
    cu, acc = [0], 0
    for L in lens:
        acc += L
        cu.append(acc)
    return cu
# (label, enable_intra, zigzag)
VARIANTS = [
    ("inter/plain", False, False),
    ("inter/zzcp", False, True),
    ("dual/plain", True, False),
    ("dual/zzcp", True, True),
]


def cleanup_cuda():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _free_port() -> str:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def _make_ctx(label, enable_intra, zigzag, cu_g, h_v, group):
    return build_cp_context(
        cu_g, group=group, enable_inter=True, enable_intra=enable_intra,
        num_v_heads=h_v, is_train=True, zigzag=zigzag, force_intra_cp=enable_intra,
    )


def _local_inputs(n_tok, h_qk, h_v, device, seed):
    """Per-rank local tensors (never materialize the global sequence). The gate mirrors
    the production shape: per-head decay A~U(0,16) so warmup pruning is realistic."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = l2norm(torch.randn(1, n_tok, h_qk, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen))
    k = l2norm(torch.randn(1, n_tok, h_qk, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen))
    v = torch.randn(1, n_tok, h_v, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=gen)
    A = torch.rand(h_v, device=device, dtype=torch.float32, generator=gen) * 16
    gate_in = torch.randn(1, n_tok, h_v, device=device, dtype=torch.float32, generator=gen) * 0.5
    g = -A * torch.nn.functional.softplus(gate_in + 1.0)
    beta = torch.randn(1, n_tok, h_v, device=device, dtype=torch.float32, generator=gen).sigmoid()
    for t in (q, k, v, g, beta):
        t.requires_grad_(True)
    return dict(q=q, k=k, v=v, g=g, beta=beta, scale=HEAD_DIM ** -0.5)


def _worst_rank(ms, device):
    buf = torch.tensor([ms if ms == ms else -1.0], device=device, dtype=torch.float64)
    dist.all_reduce(buf, op=dist.ReduceOp.MAX)
    val = buf.item()
    return float("nan") if val < 0 else val


def _bench(fn, device, warmup, iters):
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            start[i].record()
            fn()
            end[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(start, end))
        ms = times[len(times) // 2]   # median
    except torch.cuda.OutOfMemoryError:
        cleanup_cuda()
        ms = float("nan")
    return _worst_rank(ms, device)


def _infer_call(d, ctx):
    def fn():
        with torch.no_grad():
            qla(d["q"], d["k"], d["v"], d["g"], d["beta"], scale=d["scale"],
                initial_state=None, output_final_state=False, cu_seqlens=None,
                cp_context=ctx)
    return fn


def _train_call(d, ctx):
    def fn():
        for t in (d["q"], d["k"], d["v"], d["g"], d["beta"]):
            t.grad = None
        o, _ = qla(d["q"], d["k"], d["v"], d["g"], d["beta"], scale=d["scale"],
                   initial_state=None, output_final_state=False, cu_seqlens=None,
                   cp_context=ctx)
        o.float().sum().backward()
    return fn


def _run(rank, world_size, cfg, port):
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=port,
                      RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    device = f"cuda:{rank}"
    h_qk, h_v, seqlens, warmup, iters, pack = cfg

    rows = []
    for T in seqlens:
        res = {}
        for label, enable_intra, zigzag in VARIANTS:
            cleanup_cuda()
            try:
                cu_g = torch.tensor(build_cu_seqlens(T, pack), dtype=torch.int32, device=device)
                ctx = _make_ctx(label, enable_intra, zigzag, cu_g, h_v, group)
                if enable_intra and not ctx.is_intra:
                    res[label] = (float("nan"), float("nan"))
                    continue
                n_local = int(ctx.cu_seqlens[-1])   # rank-local token count (both layouts == T/ws)
                d = _local_inputs(n_local, h_qk, h_v, device, seed=1234 + rank)
                infer = _bench(_infer_call(d, ctx), device, warmup, iters)
                train = _bench(_train_call(d, ctx), device, warmup, iters)
                res[label] = (infer, train)
                del d
            except torch.cuda.OutOfMemoryError:
                cleanup_cuda()
                res[label] = (float("nan"), float("nan"))
        rows.append((T, res))

    if rank == 0:
        _report(rows, h_qk, h_v, world_size, pack)
    dist.barrier()
    dist.destroy_process_group()


def _f(ms):
    return "    N/A " if ms != ms else f"{ms:>8.2f}"


def _ratio(plain, zz):
    if plain != plain or zz != zz or zz == 0:
        return "  N/A "
    return f"{zz / plain:>5.2f}x"


def _report(rows, h_qk, h_v, world_size, pack):
    desc = "single doc" if pack == "single" else f"varlen: 1 big + {SMALL_DOCS}"
    for di, direction in enumerate(("INFER (fwd)", "TRAIN (fwd+bwd)")):
        print(f"\n>>> {direction}  h_qk={h_qk} h_v={h_v} CP{world_size} bs=1  [{desc}]  "
              f"(ms = worst rank; ratio = zzcp/plain, >1 = zzcp slower)")
        hdr = (f"{'seqlen':>9} | {'inter plain':>11} {'inter zzcp':>11} {'zz/pl':>6} | "
               f"{'dual plain':>11} {'dual zzcp':>11} {'zz/pl':>6}")
        print(hdr)
        print("-" * len(hdr))
        for T, res in rows:
            ip = res.get("inter/plain", (float("nan"),) * 2)[di]
            iz = res.get("inter/zzcp", (float("nan"),) * 2)[di]
            dp = res.get("dual/plain", (float("nan"),) * 2)[di]
            dz = res.get("dual/zzcp", (float("nan"),) * 2)[di]
            print(f"{T:>9} | {_f(ip)} {_f(iz)} {_ratio(ip, iz):>6} | "
                  f"{_f(dp)} {_f(dz)} {_ratio(dp, dz):>6}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=min(4, torch.cuda.device_count()))
    p.add_argument("--h-qk", type=int, default=16)
    p.add_argument("--h-v", type=int, default=16)
    p.add_argument("--seqlens", type=str, default=",".join(str(s) for s in SEQLENS))
    p.add_argument("--pack", choices=["single", "varlen"], default="single")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    ws = args.world_size
    seqlens = []
    for s in (int(x) for x in args.seqlens.split(",")):
        docs = build_cu_seqlens(s, args.pack)
        doc_lens = [docs[i + 1] - docs[i] for i in range(len(docs) - 1)]
        # zzcp needs every doc divisible by 2*ws; plain needs total % ws and per-card chunk.
        bad = (s % (2 * ws) != 0 or (s // ws) % CHUNK_SIZE != 0
               or any(L <= 0 or L % (2 * ws) != 0 for L in doc_lens))
        if bad:
            print(f"[skip] T={s} pack={args.pack}: need total % 2*ws={2 * ws}=0, "
                  f"per-card T/ws % chunk={CHUNK_SIZE}=0, every doc % 2*ws=0 (docs={doc_lens})")
            continue
        seqlens.append(s)
    assert seqlens, "no valid seqlens"
    assert torch.cuda.device_count() >= ws, f"need {ws} GPUs"

    cfg = (args.h_qk, args.h_v, seqlens, args.warmup, args.iters, args.pack)
    mp.start_processes(_run, args=(ws, cfg, _free_port()), nprocs=ws, join=True,
                       start_method="spawn")


if __name__ == "__main__":
    main()
