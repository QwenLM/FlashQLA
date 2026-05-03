# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Apple Silicon benchmark for flash_qla_mlx.
# Measures forward and backward throughput across sequence lengths and
# chunk sizes to find the optimal chunk_size for your hardware.
#
# Usage:
#   python benchmark/bench_mlx.py                  # default sweep
#   python benchmark/bench_mlx.py --fwd-only       # skip backward
#   python benchmark/bench_mlx.py --T 512 1024 2048
#   python benchmark/bench_mlx.py --chunk-sizes 32 64 128

import argparse
import time

import mlx.core as mx

from flash_qla_mlx import chunk_gated_delta_rule


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _warmup_and_time(fn, *args, warmup: int = 3, iters: int = 10) -> float:
    """Return median wall-clock time in ms over `iters` runs."""
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_fwd(B, T, Hk, Hv, K, V, chunk_size, dtype):
    mx.random.seed(0)
    q    = mx.random.normal((B, T, Hk, K)).astype(dtype)
    k    = mx.random.normal((B, T, Hk, K)).astype(dtype)
    v    = mx.random.normal((B, T, Hv, V)).astype(dtype)
    g    = -mx.abs(mx.random.normal((B, T, Hv))).astype(dtype)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(dtype)

    def run():
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size)
        return o

    ms = _warmup_and_time(run)
    toks = B * T
    return ms, toks / ms * 1e3  # tokens/s


def bench_bwd(B, T, Hk, Hv, K, V, chunk_size, dtype):
    mx.random.seed(0)
    q    = mx.random.normal((B, T, Hk, K)).astype(dtype)
    k    = mx.random.normal((B, T, Hk, K)).astype(dtype)
    v    = mx.random.normal((B, T, Hv, V)).astype(dtype)
    g    = -mx.abs(mx.random.normal((B, T, Hv))).astype(dtype)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv))).astype(dtype)

    def run():
        loss, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size)
        grad = mx.grad(
            lambda q, k, v, g, beta: chunk_gated_delta_rule(
                q, k, v, g, beta, chunk_size=chunk_size
            )[0].sum()
        )(q, k, v, g, beta)
        return grad

    ms = _warmup_and_time(run)
    toks = B * T
    return ms, toks / ms * 1e3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="flash_qla_mlx Apple Silicon benchmark")
    parser.add_argument("--B", type=int, default=2, help="Batch size")
    parser.add_argument("--Hk", type=int, default=4, help="Key/query heads")
    parser.add_argument("--Hv", type=int, default=4, help="Value heads")
    parser.add_argument("--K", type=int, default=64, help="Key/query head dim")
    parser.add_argument("--V", type=int, default=64, help="Value head dim")
    parser.add_argument(
        "--T", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--chunk-sizes", type=int, nargs="+", default=[32, 64, 128],
        help="chunk_size values to sweep",
    )
    parser.add_argument("--fwd-only", action="store_true", help="Skip backward pass")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                        default="float32")
    args = parser.parse_args()

    dtype_map = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
    dtype = dtype_map[args.dtype]

    passes = ["fwd"] if args.fwd_only else ["fwd", "bwd"]
    bench_fns = {"fwd": bench_fwd, "bwd": bench_bwd}

    for pass_name in passes:
        bench = bench_fns[pass_name]
        col_w = 12
        header = f"{'T':>6}  " + "  ".join(
            f"cs={cs:>3} ms  Ktok/s" for cs in args.chunk_sizes
        )
        sep = "-" * len(header)

        print(f"\n{pass_name.upper()} pass  "
              f"(B={args.B} Hk={args.Hk} Hv={args.Hv} "
              f"K={args.K} V={args.V} dtype={args.dtype})")
        print(header)
        print(sep)

        for T in args.T:
            if T % max(args.chunk_sizes) != 0:
                # pad check: skip sizes where T < chunk_size
                valid_cs = [cs for cs in args.chunk_sizes if cs <= T]
            else:
                valid_cs = args.chunk_sizes

            row = f"{T:>6}  "
            for cs in args.chunk_sizes:
                if cs > T:
                    row += f"{'--':>8}  {'--':>6}  "
                    continue
                try:
                    ms, tps = bench(
                        args.B, T, args.Hk, args.Hv, args.K, args.V, cs, dtype
                    )
                    row += f"{ms:>8.2f}  {tps/1000:>6.1f}  "
                except Exception as e:
                    row += f"{'ERR':>8}  {'':>6}  "
            print(row)

    print()


if __name__ == "__main__":
    main()
