"""Single-GPU comparison of SM120 intra-CP boundary correction kernels.

python -m benchmark.demo_intra_cp_scan --heads 16 --segments 16
python -m benchmark.demo_intra_cp_scan --heads 16 --tokens 8192
The serial kernel is the original implementation; no NCCL or CP cache changes.
"""

import argparse
from functools import partial
from statistics import median

import torch

from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.cp_fwd import tilelang_correct_h0
from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120.cp_scan import tilelang_correct_h0_ws


def check(actual, expected, label):
    error = (actual.float() - expected.float()).abs().max().item()
    relative = error / max(expected.float().abs().max().item(), 1e-6)
    assert relative < 0.02, f"{label}: relative max error {relative:.3e} >= 0.02"


def bench_us(jobs, repeats):
    """Interleave preallocated CUDA graphs to limit clock/temperature bias."""
    graphs = []
    for _, fn, _ in jobs:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        graphs.append(graph)
    for _ in range(5):
        for graph in graphs:
            graph.replay()
    torch.cuda.synchronize()
    samples = [[] for _ in jobs]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for round_idx in range(9):
        order = [(i + round_idx) % len(jobs) for i in range(len(jobs))]
        if round_idx % 2:
            order.reverse()
        for i in order:
            start.record()
            graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) * 1000 / repeats)
    return samples


def bench_operator(heads, tokens, repeats):
    """Reuse the repository's single-card fwd/bwd benchmark unchanged."""
    from benchmark import bench_gated_delta_rule as bench
    from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120 import cp_fwd

    print(f"{torch.cuda.get_device_name()}: H={heads}, tokens={tokens}", flush=True)
    bench.HAS_FLA = bench.HAS_FI = False
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    for training in (False, True):
        ctx = bench.build_cp_context(cu, enable_intra=True, num_v_heads=heads,
                                     chunk_size=bench.CHUNK_SIZE, is_train=training)
        parts = ctx.intra_cp_cu_seqlens.numel() - 1 if ctx.is_intra else 1
        print(f"AutoCP training={training}: intra={ctx.is_intra}, parts={parts}", flush=True)
    samples = {name: [[], []] for name in ("serial", "ws_tma")}
    factories = {"serial": tilelang_correct_h0, "ws_tma": tilelang_correct_h0_ws}
    # Both variants use the same inputs and AutoCP decision. Only the SM120
    # correction factory is replaced for this comparison; no production switch.
    for round_idx in range(3):
        order = ("serial", "ws_tma") if round_idx % 2 == 0 else ("ws_tma", "serial")
        for name in order:
            cp_fwd.tilelang_correct_h0_ws = factories[name]
            for i, fn in enumerate((bench.bench_fwd, bench.bench_bwd)):
                torch.manual_seed(0)
                ms = fn([tokens], heads, heads, warmup=5, repeats=repeats,
                        auto_cp=True, backend="cudagraph")[0]
                assert ms > 0, f"{name}: original benchmark failed"
                samples[name][i].append(ms)
    cp_fwd.tilelang_correct_h0_ws = tilelang_correct_h0_ws
    print("Original single-card benchmark: fwd and bwd measured separately (ms).")
    for i, phase in enumerate(("fwd", "bwd")):
        baseline, optimized = (median(samples[name][i]) for name in factories)
        print(f"  {phase}: serial={baseline:.4f}, ws_tma={optimized:.4f}, "
              f"speedup={baseline / optimized:.3f}x", flush=True)
        print(f"    samples: serial={samples['serial'][i]}, ws_tma={samples['ws_tma'][i]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--layout", choices=("kv", "vk"), default="kv")
    parser.add_argument("--m-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--fallback", choices=("all", "none", "mixed"), default="all")
    parser.add_argument("--tokens", type=int, help="Run original operator benchmark instead of the scan microbenchmark")
    args = parser.parse_args()
    assert min(args.heads, args.segments, args.repeats) > 0
    if args.tokens is not None:
        assert args.tokens > 0
        bench_operator(args.heads, args.tokens, args.repeats)
        return
    torch.manual_seed(0)
    heads, segments = args.heads, args.segments
    state = torch.randn((segments, heads, 128, 128), device="cuda") * 0.1
    m = (0.95 * torch.eye(128, device="cuda") + torch.randn_like(state) * 0.005)
    m = m.to(torch.bfloat16 if args.m_dtype == "bf16" else torch.float32)
    seed = torch.randn((1, heads, 128, 128), device="cuda") * 0.1
    seq_map = torch.tensor([0, segments], device="cuda", dtype=torch.int32)
    fallback = torch.ones((segments, heads), device="cuda", dtype=torch.bool)
    print(f"{torch.cuda.get_device_name()}: H={heads}, segments={segments}, "
          f"layout={args.layout}, M={m.dtype}, fallback={args.fallback}", flush=True)

    for reverse in (False, True):
        # Match production: BF16 h/M in forward, FP32 dh with BF16 M in backward.
        h = state if reverse else state.to(m.dtype)
        jobs = []
        for name, factory in (("serial", tilelang_correct_h0), ("ws_tma", tilelang_correct_h0_ws)):
            out = torch.empty_like(state)
            kernel = factory(
                H=heads, DK=128, DV=128, res_dtype=out.dtype, accum_dtype="float32",
                h_buffer_dtype=h.dtype, m_buffer_dtype=m.dtype,
                seqlen_dtype=seq_map.dtype, mask_dtype=fallback.dtype,
                use_raw_h0=True, state_v_first=args.layout == "vk",
                reverse=reverse, transpose_m=reverse,
            )
            jobs.append((name, partial(kernel, seed, h, m, fallback, seq_map, out), out))

        for pattern in ("all", "none", "mixed"):
            fallback.fill_(pattern != "none")
            if pattern == "mixed":
                fallback[::2, ::2] = False
            for _, fn, _ in jobs:
                fn()
            check(jobs[1][2], jobs[0][2], f"reverse={reverse}, {pattern}")
        fallback.fill_(args.fallback != "none")
        if args.fallback == "mixed":
            fallback[::2, ::2] = False
        samples = bench_us(jobs, args.repeats)
        check(jobs[1][2], jobs[0][2], "after graph replay")
        baseline = median(samples[0])
        print("dht correction" if reverse else "h0 correction", flush=True)
        for (name, _, _), times in zip(jobs, samples):
            us = median(times)
            print(f"  {name:8s} {us:9.3f} us  {baseline / us:.3f}x  "
                  f"[{min(times):.3f}, {max(times):.3f}]", flush=True)


if __name__ == "__main__":
    main()
