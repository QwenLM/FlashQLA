# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Apple Silicon benchmark for flash_qla_mlx.
# Measures forward/backward throughput across sequence lengths and chunk sizes,
# and optionally compares against mx.fast.scaled_dot_product_attention.
#
# Usage:
#   # chunk-size sweep (default)
#   python benchmark/bench_mlx.py --B 1 --Hk 8 --Hv 32 --K 128 --V 128
#
#   # Qwen-27B forward-only comparison vs softmax attention
#   python benchmark/bench_mlx.py --B 1 --Hk 8 --Hv 32 --K 128 --V 128 \
#       --fwd-only --compare-sdpa
#
#   # Short options
#   python benchmark/bench_mlx.py --fwd-only --T 512 1024 2048 4096 8192
#   python benchmark/bench_mlx.py --chunk-sizes 32 64 128

import argparse
import math
import time

import mlx.core as mx

from flash_qla_mlx import chunk_gated_delta_rule


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time(fn, warmup: int = 3, iters: int = 10) -> float:
    """Return median wall-clock time in ms."""
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# FlashQLA runners
# ---------------------------------------------------------------------------

def _qla_inputs(B, T, Hk, Hv, K, V, dtype):
    mx.random.seed(0)
    return (
        mx.random.normal((B, T, Hk, K)).astype(dtype),  # q
        mx.random.normal((B, T, Hk, K)).astype(dtype),  # k
        mx.random.normal((B, T, Hv, V)).astype(dtype),  # v
        -mx.abs(mx.random.normal((B, T, Hv))).astype(dtype),              # g
        mx.sigmoid(mx.random.normal((B, T, Hv))).astype(dtype),           # beta
    )


def bench_qla_fwd(B, T, Hk, Hv, K, V, chunk_size, dtype):
    q, k, v, g, beta = _qla_inputs(B, T, Hk, Hv, K, V, dtype)
    ms = _time(lambda: chunk_gated_delta_rule(q, k, v, g, beta,
                                              chunk_size=chunk_size)[0])
    return ms, B * T / ms * 1e3


def bench_qla_bwd(B, T, Hk, Hv, K, V, chunk_size, dtype):
    q, k, v, g, beta = _qla_inputs(B, T, Hk, Hv, K, V, dtype)

    def run():
        return mx.grad(
            lambda q, k, v, g, beta: chunk_gated_delta_rule(
                q, k, v, g, beta, chunk_size=chunk_size
            )[0].sum()
        )(q, k, v, g, beta)

    ms = _time(run)
    return ms, B * T / ms * 1e3


# ---------------------------------------------------------------------------
# SDPA runner
#
# Dimension mapping from FlashQLA to GQA softmax attention (Qwen-27B style):
#   FlashQLA Hk  ->  SDPA N_kv  (KV heads, e.g. 8)
#   FlashQLA Hv  ->  SDPA N_q   (query heads, e.g. 32)
#   FlashQLA K   ->  SDPA D     (head dim, e.g. 128)
#
# mx.fast.scaled_dot_product_attention expects [B, H, T, D] layout.
# Causal mask is used for a fair comparison (FlashQLA is always causal).
# ---------------------------------------------------------------------------

def bench_sdpa_fwd(B, T, Hk, Hv, K, V, dtype):
    """
    Benchmark mx.fast.scaled_dot_product_attention with GQA:
      q : [B, Hv, T, K]   — Hv query heads
      k : [B, Hk, T, K]   — Hk KV heads
      v : [B, Hk, T, V]   — Hk KV heads
    This matches the output dimensionality of FlashQLA (B, T, Hv, V).
    """
    mx.random.seed(0)
    scale = K ** -0.5
    q = mx.random.normal((B, Hv, T, K)).astype(dtype)
    k = mx.random.normal((B, Hk, T, K)).astype(dtype)
    v = mx.random.normal((B, Hk, T, V)).astype(dtype)

    ms = _time(lambda: mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal"
    ))
    return ms, B * T / ms * 1e3


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _print_qla_table(pass_name, args, dtype):
    bench = bench_qla_fwd if pass_name == "fwd" else bench_qla_bwd
    header = f"{'T':>6}  " + "  ".join(
        f"cs={cs:>3}  ms    Ktok/s" for cs in args.chunk_sizes
    )
    print(f"\nFlashQLA {pass_name.upper()}  "
          f"(B={args.B} Hk={args.Hk} Hv={args.Hv} "
          f"K={args.K} V={args.V} {args.dtype})")
    print(header)
    print("-" * len(header))
    for T in args.T:
        row = f"{T:>6}  "
        for cs in args.chunk_sizes:
            if cs > T:
                row += f"{'--':>10}  {'--':>6}  "
                continue
            try:
                ms, tps = bench(args.B, T, args.Hk, args.Hv,
                                args.K, args.V, cs, dtype)
                row += f"{ms:>10.2f}  {tps/1000:>6.1f}  "
            except Exception as e:
                row += f"{'ERR':>10}  {'ERR':>6}  "
        print(row)


def _print_compare_table(args, dtype):
    """
    Side-by-side: best FlashQLA chunk_size vs mx.fast.scaled_dot_product_attention.
    Runs forward pass only (SDPA has no exposed backward in MLX fast path).
    """
    # Pick best chunk_size from a quick pre-sweep
    print(f"\nPre-sweeping chunk sizes to find best for T={args.T[0]}...")
    best_cs, best_ms = args.chunk_sizes[0], float("inf")
    for cs in args.chunk_sizes:
        if cs > args.T[0]:
            continue
        try:
            ms, _ = bench_qla_fwd(args.B, args.T[0], args.Hk, args.Hv,
                                   args.K, args.V, cs, dtype)
            if ms < best_ms:
                best_ms, best_cs = ms, cs
        except Exception:
            pass
    print(f"  Best chunk_size = {best_cs}")

    # GQA layout note
    print(f"\n  SDPA shape: q=[B,{args.Hv},T,{args.K}]  "
          f"k=v=[B,{args.Hk},T,{args.K}]  causal  (GQA {args.Hv//args.Hk}:1)")
    print(f"  FlashQLA:   Hk={args.Hk} Hv={args.Hv} K={args.K} V={args.V}  "
          f"chunk_size={best_cs}")

    col = 14
    header = (f"{'T':>6}  {'FlashQLA':>{col}}  {'SDPA':>{col}}  "
              f"{'QLA ms':>8}  {'SDPA ms':>8}  {'speedup':>8}")
    print()
    print(header)
    print("-" * len(header))

    for T in args.T:
        if best_cs > T:
            print(f"{T:>6}  (T < chunk_size, skipped)")
            continue

        try:
            qla_ms, qla_tps = bench_qla_fwd(args.B, T, args.Hk, args.Hv,
                                             args.K, args.V, best_cs, dtype)
        except Exception as e:
            print(f"{T:>6}  QLA ERR: {e}")
            continue

        try:
            sdpa_ms, sdpa_tps = bench_sdpa_fwd(args.B, T, args.Hk, args.Hv,
                                                args.K, args.V, dtype)
            speedup = sdpa_ms / qla_ms
            sdpa_str    = f"{sdpa_tps/1000:>10.1f} Ktok/s"
            sdpa_ms_str = f"{sdpa_ms:>8.2f}"
            speedup_str = f"{speedup:>7.2f}x"
        except Exception as e:
            sdpa_str    = f"{'OOM/ERR':>{col}}"
            sdpa_ms_str = f"{'--':>8}"
            speedup_str = f"{'--':>8}"

        qla_str = f"{qla_tps/1000:>10.1f} Ktok/s"
        print(f"{T:>6}  {qla_str:>{col}}  {sdpa_str:>{col}}  "
              f"{qla_ms:>8.2f}  {sdpa_ms_str}  {speedup_str}")

    print()
    print("  speedup > 1 means FlashQLA is faster than SDPA")
    print("  Note: FlashQLA is O(T) memory; SDPA is O(T²).  "
          "They are architecturally different primitives.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="flash_qla_mlx Apple Silicon benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Qwen-27B forward comparison:
  python benchmark/bench_mlx.py \\
      --B 1 --Hk 8 --Hv 32 --K 128 --V 128 \\
      --fwd-only --compare-sdpa \\
      --T 512 1024 2048 4096 8192 16384
""",
    )
    parser.add_argument("--B",   type=int, default=1)
    parser.add_argument("--Hk",  type=int, default=8,   help="KV heads (FlashQLA k heads)")
    parser.add_argument("--Hv",  type=int, default=32,  help="Q/output heads (FlashQLA v heads)")
    parser.add_argument("--K",   type=int, default=128, help="Head dim (key/query)")
    parser.add_argument("--V",   type=int, default=128, help="Head dim (value)")
    parser.add_argument(
        "--T", type=int, nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument(
        "--chunk-sizes", type=int, nargs="+", default=[32, 64],
        help="chunk_size values to sweep in the QLA table",
    )
    parser.add_argument("--fwd-only",    action="store_true")
    parser.add_argument("--compare-sdpa", action="store_true",
                        help="Add SDPA vs FlashQLA comparison table")
    parser.add_argument("--dtype",
                        choices=["float32", "float16", "bfloat16"],
                        default="float32")
    args = parser.parse_args()

    dtype = {"float32": mx.float32,
             "float16": mx.float16,
             "bfloat16": mx.bfloat16}[args.dtype]

    passes = ["fwd"] if args.fwd_only else ["fwd", "bwd"]
    for p in passes:
        _print_qla_table(p, args, dtype)

    if args.compare_sdpa:
        _print_compare_table(args, dtype)


if __name__ == "__main__":
    main()
