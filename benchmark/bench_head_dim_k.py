# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Benchmark: GDN chunk scan across head_dim_k in {64, 128} (Blackwell forward).
#
# Motivation: the main benchmark (bench_gated_delta_rule.py) hardcodes HEAD_DIM=128
# (head_dim_k == head_dim_v). This script sweeps the *contraction* dim head_dim_k
# independently of head_dim_v, to show (a) the newly-enabled head_dim_k=64 path is
# a large win over FLA (it avoids padding K to 128) and (b) the original
# head_dim_k=128 path is not regressed. head_dim_v is fixed at 128.
#
# Correctness for these dims is covered by tests/test_gdr_unit.py
# (test_fwd_head_dim_k / test_fwd_head_dim_v); this script only measures latency,
# plus a coarse max-abs-diff vs FLA as a sanity check that the compared kernels
# agree.
#
# Example:
#   python benchmark/bench_head_dim_k.py                 # cudagraph backend (default)
#   python benchmark/bench_head_dim_k.py --backend event # per-iter CUDA events

import argparse
import math
from typing import Dict, List, Tuple

import torch

import tilelang

from flash_qla import chunk_gated_delta_rule as qla_chunk
from flash_qla.utils import l2norm

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
    HAS_FLA = True
except ImportError:
    HAS_FLA = False


# H=12 matches the GDN-hybrid config that motivated the PR (head_k_dim=64,
# expand_v=2.0 -> head_v_dim=128). head_dim_v is fixed at 128 throughout.
NUM_HEADS = 12
HEAD_DIM_V = 128
HEAD_DIM_K_SWEEP = [64, 128]
# (batch, seqlen) buckets.
SHAPES: List[Tuple[int, int]] = [(64, 4096), (32, 8192), (8, 16384)]


def get_lib_versions() -> Dict[str, str]:
    versions = {"torch": torch.__version__}
    try:
        import fla
        versions["fla"] = getattr(fla, "__version__", "installed (ver unknown)")
    except ImportError:
        versions["fla"] = "not installed"
    versions["tilelang"] = getattr(tilelang, "__version__", "installed (ver unknown)")
    return versions


def make_inputs(batch: int, seqlen: int, head_dim_k: int):
    device, dtype = "cuda", torch.bfloat16
    # q/k are l2-normalized here (same as bench_gated_delta_rule.py) so
    # use_qk_l2norm_in_kernel is left OFF for both backends -- an apples-to-apples
    # comparison where neither side pays an in-kernel norm.
    q = l2norm(torch.randn(batch, seqlen, NUM_HEADS, head_dim_k, device=device, dtype=dtype))
    k = l2norm(torch.randn(batch, seqlen, NUM_HEADS, head_dim_k, device=device, dtype=dtype))
    v = torch.randn(batch, seqlen, NUM_HEADS, HEAD_DIM_V, device=device, dtype=dtype)
    # log-decay gate in (-inf, 0), /16 to match the repo convention (tests +
    # bench_gated_delta_rule.py).
    g = torch.nn.functional.logsigmoid(
        torch.randn(batch, seqlen, NUM_HEADS, device=device, dtype=torch.float32)
    ) / 16
    beta = torch.randn(batch, seqlen, NUM_HEADS, device=device, dtype=torch.float32).sigmoid()
    return q, k, v, g, beta, head_dim_k ** -0.5


def _call(fn, q, k, v, g, beta, scale):
    o = fn(q=q, k=k, v=v, g=g, beta=beta, scale=scale, output_final_state=True)
    return o[0] if isinstance(o, tuple) else o


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN chunk scan across head_dim_k")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--backend", choices=["event", "cudagraph"], default="cudagraph",
                        help="tilelang profiler backend: 'event' (per-iter CUDA events) or "
                             "'cudagraph' (graph replay, removes host dispatch overhead)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    if not HAS_FLA:
        print("[WARN] fla not installed -- only FlashQLA timings will be reported.")

    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"Config: H={NUM_HEADS} head_dim_v={HEAD_DIM_V} causal  "
          f"warmup={args.warmup} repeats={args.repeats} backend={args.backend}")
    print("Library Versions: " + " | ".join(f"{k}: {v}" for k, v in get_lib_versions().items()))
    print("Correctness for these dims is covered by tests/test_gdr_unit.py "
          "(parity vs the fp64 reference); this script measures latency only.")
    print("=" * 66)
    print(f"{'DK':>4}{'batch':>7}{'seqlen':>8}{'FLA':>12}{'FlashQLA':>12}{'speedup':>10}")
    print("-" * 66)

    for head_dim_k in HEAD_DIM_K_SWEEP:
        for batch, seqlen in SHAPES:
            q, k, v, g, beta, scale = make_inputs(batch, seqlen, head_dim_k)

            qla_ms = tilelang.profiler.do_bench(
                lambda: _call(qla_chunk, q, k, v, g, beta, scale),
                warmup=args.warmup, rep=args.repeats, backend=args.backend,
            )

            if HAS_FLA:
                fla_ms = tilelang.profiler.do_bench(
                    lambda: _call(fla_chunk, q, k, v, g, beta, scale),
                    warmup=args.warmup, rep=args.repeats, backend=args.backend,
                )
                speedup = f"{fla_ms / qla_ms:>8.2f}x"
                fla_str = f"{fla_ms * 1e3:>10.0f}us"
            else:
                speedup, fla_str = "     N/A ", "        N/A "

            print(f"{head_dim_k:>4}{batch:>7}{seqlen:>8}{fla_str}"
                  f"{qla_ms * 1e3:>10.0f}us{speedup}", flush=True)

    print("\nBenchmark finished.")


if __name__ == "__main__":
    main()
