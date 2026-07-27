# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Benchmark Script for FlashQLA

import argparse
import math
import gc
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F

import tilelang

# Kernel Imports
from flash_qla import (
    chunk_gated_delta_rule_fwd as qla_fwd,
    chunk_gated_delta_rule_bwd as qla_bwd,
)
from flash_qla.utils import l2norm

try:
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_fwd as fla_fwd,
        chunk_gated_delta_rule_bwd as fla_bwd,
    )
    HAS_FLA = True
except ImportError:
    HAS_FLA = False

try:
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_fwd
    HAS_FI = True
except ImportError:
    HAS_FI = False

HEAD_DIM = 128

@dataclass
class ModelConfig:
    label: str
    h_qk: int
    h_v: int


@dataclass
class SeqLenConfig:
    label: str
    seqlens: List[int]


def generate_rand_seqlens(batch_size, num_tokens, seed=42):
    if batch_size <= 1:
        return [num_tokens]
    gen = torch.Generator().manual_seed(seed)
    bars = (
        torch.sort(
            torch.randperm(num_tokens - 1, dtype=torch.int32, generator=gen)[
                : batch_size - 1
            ]
        ).values
        + 1
    )
    cu_seqlens = torch.nn.functional.pad(bars, (1, 1))
    cu_seqlens[-1] = num_tokens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    return seqlens.tolist()


FWD_MODEL_CONFIGS = [
    ModelConfig("397B/122B TP8", 2, 8),
    ModelConfig("397B/122B TP4", 4, 16),
    ModelConfig("397B/122B TP2", 8, 32),
    ModelConfig("397B/122B TP1", 16, 64),
    ModelConfig("35B/9B/4B TP1", 16, 32),
    ModelConfig("27B TP2", 8, 24),
    ModelConfig("27B TP1", 16, 48),
    ModelConfig("2B/0.8B TP1", 16, 16),
    ModelConfig("Sym h32", 32, 32),
]

FWD_SEQLEN_CONFIGS = [
    SeqLenConfig("1x32768", [32768]),
    SeqLenConfig("1x16384", [16384]),
    SeqLenConfig("1x8192", [8192]),
    SeqLenConfig("1x4096", [4096]),
    SeqLenConfig("1x2048", [2048]),
    SeqLenConfig("28672+4096", [28672, 4096]),
    SeqLenConfig("24576+8192", [24576, 8192]),
    SeqLenConfig("16384+16384", [16384, 16384]),
    SeqLenConfig("8192+24576", [8192, 24576]),
    SeqLenConfig("4096+28672", [4096, 28672]),
    SeqLenConfig("12288+4096", [12288, 4096]),
    SeqLenConfig("6144+2048", [6144, 2048]),
    SeqLenConfig("4096+4096", [4096, 4096]),
    SeqLenConfig("2048+6144", [2048, 6144]),
    SeqLenConfig("1024+7168", [1024, 7168]),
    SeqLenConfig("8192x4", [8192] * 4),
    SeqLenConfig("4096x8", [4096] * 8),
    SeqLenConfig("2048x4", [2048] * 4),
    SeqLenConfig("1024x8", [1024] * 8),
]
BWD_MODEL_CONFIGS = [
    ModelConfig("hk16_hv32", h_qk=16, h_v=32),
    ModelConfig("hk8_hv16", h_qk=8, h_v=16),
    ModelConfig("hk4_hv8",  h_qk=4, h_v=8),
    ModelConfig("hk2_hv4",  h_qk=2, h_v=4),
]

BWD_SEQLEN_CONFIGS = [
    SeqLenConfig("8k_1seq", generate_rand_seqlens(1, 8192, seed=42)),
    SeqLenConfig("8k_2seq", generate_rand_seqlens(2, 8192, seed=42)),
    SeqLenConfig("8k_4seq", generate_rand_seqlens(4, 8192, seed=42)),
    SeqLenConfig("8k_8seq", generate_rand_seqlens(8, 8192, seed=42)),
    SeqLenConfig("32k_1seq", generate_rand_seqlens(1, 32768, seed=42)),
    SeqLenConfig("32k_2seq", generate_rand_seqlens(2, 32768, seed=42)),
    SeqLenConfig("32k_4seq", generate_rand_seqlens(4, 32768, seed=42)),
    SeqLenConfig("32k_8seq", generate_rand_seqlens(8, 32768, seed=42)),
]


def cleanup_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_lib_versions() -> Dict[str, str]:
    """Collects version strings for relevant libraries."""
    versions = {}

    # Torch
    try:
        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "N/A"

    # Flash Linear Attention (FLA)
    try:
        import fla

        if hasattr(fla, "__version__"):
            versions["fla"] = fla.__version__
        else:
            versions["fla"] = "Installed (ver unknown)"
    except ImportError:
        versions["fla"] = "Not Installed"

    # FlashInfer
    try:
        import flashinfer

        if hasattr(flashinfer, "__version__"):
            versions["flashinfer"] = flashinfer.__version__
        elif hasattr(flashinfer, "version"):
            versions["flashinfer"] = str(flashinfer.version)
        else:
            versions["flashinfer"] = "Installed (ver unknown)"
    except ImportError:
        versions["flashinfer"] = "Not Installed"

    # TileLang
    try:
        import tilelang

        if hasattr(tilelang, "__version__"):
            versions["tilelang"] = tilelang.__version__
        elif hasattr(tilelang, "version"):
            versions["tilelang"] = str(tilelang.version)
        else:
            versions["tilelang"] = "Installed (ver unknown)"
    except ImportError:
        versions["tilelang"] = "Not Installed"

    return versions


def prepare_tensors(
    seqlens: List[int], h_qk: int, h_v: int, head_dim: int = HEAD_DIM
) -> Optional[Dict[str, Any]]:
    device = "cuda"
    num_seqs = len(seqlens)
    total_tokens = sum(seqlens)
    scale = head_dim ** (-0.5)

    offsets = [0]
    for s in seqlens:
        offsets.append(offsets[-1] + s)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)

    try:
        q = l2norm(
            torch.randn(
                1, total_tokens, h_qk, head_dim, device=device, dtype=torch.bfloat16
            )
        )
        k = l2norm(
            torch.randn(
                1, total_tokens, h_qk, head_dim, device=device, dtype=torch.bfloat16
            )
        )
        v = torch.randn(
            1, total_tokens, h_v, head_dim, device=device, dtype=torch.bfloat16
        )
        g = (
            F.logsigmoid(
                torch.randn(1, total_tokens, h_v, device=device, dtype=torch.float32)
            )
            / 16
        )
        beta = torch.randn(
            1, total_tokens, h_v, device=device, dtype=torch.float32
        ).sigmoid()
        h0 = torch.randn(
            num_seqs, h_v, head_dim, head_dim, device=device, dtype=torch.float32
        )
        do = torch.randn_like(v)
        dht = (
            torch.randn(
                num_seqs, h_v, head_dim, head_dim, device=device, dtype=torch.float32
            )
            / 8
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None
        raise e

    swa_ratio = 0.75
    swa_mask = torch.zeros(h_v, dtype=torch.bool, device=device)
    swa_mask[: math.ceil(swa_ratio * h_v)] = True
    swa_mask = swa_mask[torch.randperm(h_v, device=device)]
    g[:, :, ~swa_mask] = 0.0

    return {
        "device": device,
        "num_seqs": num_seqs,
        "total_tokens": total_tokens,
        "scale": scale,
        "cu_seqlens": cu_seqlens,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "h0": h0,
        "do": do,
        "dht": dht,
    }


def bench_fwd(
    seqlens: List[int],
    h_qk: int,
    h_v: int,
    head_dim: int = HEAD_DIM,
    warmup: int = 10,
    repeats: int = 5,
    auto_cp: bool = True,
    backend: str = "event",
) -> Tuple[float, float, float]:
    """
    Run Forward Pass Benchmark.
    Returns: (qla_mean_ms, fi_mean_ms, fla_mean_ms)
    """
    cleanup_cuda()
    data = prepare_tensors(seqlens, h_qk, h_v, head_dim)
    if data is None:
        return float("nan"), float("nan"), float("nan")

    q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
    h0, scale, cu_seqlens = data["h0"], data["scale"], data["cu_seqlens"]

    results = {}

    def call_qla_fwd():
        qla_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            output_h=False,
            cu_seqlens=cu_seqlens,
            auto_cp=auto_cp,
        )

    try:
        mean = tilelang.profiler.do_bench(call_qla_fwd, warmup=warmup, rep=repeats, backend=backend)
        results["flash_qla"] = mean
    except RuntimeError as e:
        print(f"\n[WARN] FlashQLA Fwd failed: {e}")
        cleanup_cuda()
        results["flash_qla"] = float("nan")

    if HAS_FI:

        def call_fi_fwd():
            fi_fwd(
                q=q.view(-1, h_qk, head_dim),
                k=k.view(-1, h_qk, head_dim),
                v=v.view(-1, h_v, head_dim),
                g=g.view(-1, h_v),
                beta=beta.view(-1, h_v),
                scale=scale,
                initial_state=h0,
                cu_seqlens=cu_seqlens,
                output_final_state=True,
            )

        try:
            mean = tilelang.profiler.do_bench(call_fi_fwd, warmup=warmup, rep=repeats, backend=backend)
            results["fi"] = mean
        except RuntimeError as e:
            print(f"\n[WARN] FI Fwd failed: {e}")
            cleanup_cuda()
            results["fi"] = float("nan")
    else:
        results["fi"] = float("nan")

    if HAS_FLA:
        def call_fla_fwd():
            fla_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )

        try:
            mean = tilelang.profiler.do_bench(call_fla_fwd, warmup=warmup, rep=repeats, backend=backend)
            results["fla"] = mean
        except RuntimeError as e:
            print(f"\n[WARN] FLA Fwd failed: {e}")
            cleanup_cuda()
            results["fla"] = float("nan")
    else:
        results["fla"] = float("nan")

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    return (
        results.get("flash_qla", float("nan")),
        results.get("fi", float("nan")),
        results.get("fla", float("nan")),
    )


def bench_bwd(
    seqlens: List[int],
    h_qk: int,
    h_v: int,
    head_dim: int = HEAD_DIM,
    warmup: int = 10,
    repeats: int = 100,
    auto_cp: bool = True,
    backend: str = "event",
) -> Tuple[float, float]:
    """
    Run Backward Pass Benchmark.
    Returns: (qla_mean_ms, fla_mean_ms)
    """
    cleanup_cuda()

    data = prepare_tensors(seqlens, h_qk, h_v, head_dim)
    if data is None:
        return float("nan"), float("nan")

    q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
    h0, scale, cu_seqlens = data["h0"], data["scale"], data["cu_seqlens"]
    do, dht = data["do"], data["dht"]

    g_cumsum = None
    A = None

    # Pre-run FWD to get intermediates
    try:
        result = qla_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            output_h=False,
            cu_seqlens=cu_seqlens,
            auto_cp=auto_cp,
        )
        if isinstance(result, tuple) and len(result) >= 2:
            g_cumsum, A = result[0], result[1]
        else:
            raise RuntimeError("FlashQLA FWD did not return expected intermediates")
    except RuntimeError as e:
        print(f"[FWD Error] Failed at seqlens={seqlens}, heads={h_qk}. Error: {e}")
        cleanup_cuda()
        return float("nan"), float("nan")

    results = {}

    def call_qla_bwd():
        return qla_bwd(
            q,
            k,
            v,
            g_cumsum,
            beta,
            A,
            do,
            dht,
            scale=scale,
            initial_state=h0,
            cu_seqlens=cu_seqlens,
            auto_cp=auto_cp,
        )

    try:
        mean = tilelang.profiler.do_bench(call_qla_bwd, warmup=warmup, rep=repeats, backend=backend)
        results["flash_qla"] = mean
    except RuntimeError as e:
        print(f"\n[WARN] FlashQLA Bwd failed: {e}")
        cleanup_cuda()
        results["flash_qla"] = float("nan")

    if HAS_FLA:
        def call_fla_bwd():
            return fla_bwd(q, k, v, g_cumsum, beta, A, scale, h0, do, dht, cu_seqlens=cu_seqlens)

        try:
            mean = tilelang.profiler.do_bench(call_fla_bwd, warmup=warmup, rep=repeats, backend=backend)
            results["fla"] = mean
        except RuntimeError as e:
            print(f"\n[WARN] FLA Bwd failed: {e}")
            cleanup_cuda()
            results["fla"] = float("nan")
    else:
        results["fla"] = float("nan")

    return results.get("flash_qla", float("nan")), results.get("fla", float("nan"))


FWD_HDR = (
    f"{'Model Config':<16} {'Seqlens':<17} {'h_qk':>5} {'h_v':>5}    "
    f"{'flash_qla [fwd]':>10}  {'FI [fwd]':>10}  {'FLA [fwd]':>10}   "
    f"{'vs FLA':>7}  {'vs FI':>7}"
)

BWD_HDR = (
    f"{'Model Config':<14} {'SeqLen':<12} {'h_qk':>4} {'h_v':>4} {'#seq':>4}    "
    f"{'flash_qla [bwd]':>10}  {'FLA [bwd]':>10}   {'Speedup':>8}"
)


def fmt_time(ms: float) -> str:
    if math.isnan(ms):
        return "     N/A  "
    return f"{ms:>8.3f}ms"


def fmt_ratio(base: float, other: float) -> str:
    if math.isnan(base) or math.isnan(other) or base == 0:
        return "   N/A  "
    return f"{other / base:>6.2f}x"


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashQLA Gated Delta Rule")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--mode", choices=["fwd", "bwd", "all"], default="all")
    parser.add_argument("--skip-fi", action="store_true")
    parser.add_argument("--skip-fla", action="store_true")
    parser.add_argument("--no-cp", action="store_true", help="Disable chunk parallel")
    parser.add_argument("--backend", choices=["event", "cudagraph"], default="cudagraph",
                        help="Profiler backend: event (per-iter CUDA events) or cudagraph (graph replay, eliminates host dispatch overhead)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    global HAS_FI, HAS_FLA
    if args.skip_fi:
        HAS_FI = False
    if args.skip_fla:
        HAS_FLA = False

    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {gpu_name}")
    print("Models: Qwen3.5 family (397B, 122B, 35B, 27B, 9B, 4B, 2B, 0.8B), d=128")
    print(f"Config: Warmup={args.warmup}, Repeats={args.repeats}, Backend={args.backend}")

    libs = get_lib_versions()
    print("Library Versions:")
    ver_str = " | ".join([f"{k}: {v}" for k, v in libs.items()])
    print(f"  {ver_str}")

    print("=" * 110)

    # Forward
    if args.mode in ("fwd", "all"):
        print("\n>>> FORWARD BENCHMARKS")
        print(FWD_HDR)
        print("-" * len(FWD_HDR))

        prev_model = None
        for cfg in FWD_MODEL_CONFIGS:
            if prev_model is not None and cfg.label != prev_model:
                print()
            prev_model = cfg.label

            for sl_cfg in FWD_SEQLEN_CONFIGS:
                try:
                    qla_ms, fi_ms, fla_ms = bench_fwd(
                        sl_cfg.seqlens,
                        cfg.h_qk,
                        cfg.h_v,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        auto_cp=not args.no_cp,
                        backend=args.backend,
                    )

                    if math.isnan(qla_ms) and math.isnan(fla_ms):
                        cleanup_cuda()

                    ratio_fla = fmt_ratio(qla_ms, fla_ms)
                    ratio_fi = fmt_ratio(qla_ms, fi_ms)

                    print(
                        f"{cfg.label:<16} {sl_cfg.label:<17} {cfg.h_qk:>5} {cfg.h_v:>5}    "
                        f"{fmt_time(qla_ms)}  {fmt_time(fi_ms)}  {fmt_time(fla_ms)}   "
                        f"{ratio_fla}  {ratio_fi}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"\n[ERROR] Forward Case Failed: {cfg.label} / {sl_cfg.label}"
                    )
                    print(f"Exception: {e}")
                    cleanup_cuda()
                    continue

                cleanup_cuda()

    # Backward
    if args.mode in ("bwd", "all"):
        print("\n" + "=" * 110)
        print(f"\n>>> BACKWARD BENCHMARKS")
        print(BWD_HDR)
        print("-" * len(BWD_HDR))

        prev_model = None
        for cfg in BWD_MODEL_CONFIGS:
            if prev_model is not None and cfg.label != prev_model:
                print()
            prev_model = cfg.label

            for sl_cfg in BWD_SEQLEN_CONFIGS:
                try:
                    qla_bwd_ms, fla_bwd_ms = bench_bwd(
                        sl_cfg.seqlens,
                        cfg.h_qk,
                        cfg.h_v,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        auto_cp=not args.no_cp,
                        backend=args.backend,
                    )

                    if math.isnan(qla_bwd_ms) and math.isnan(fla_bwd_ms):
                        speedup = "   Skip "
                    else:
                        speedup = fmt_ratio(qla_bwd_ms, fla_bwd_ms)

                    num_seqs = len(sl_cfg.seqlens)
                    print(
                        f"{cfg.label:<14} {sl_cfg.label:<12} {cfg.h_qk:>4} {cfg.h_v:>4} {num_seqs:>4}    "
                        f"{fmt_time(qla_bwd_ms)}  {fmt_time(fla_bwd_ms)}   {speedup} ",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"\n[CRITICAL ERROR] Case Crashed: Heads={cfg.label}, TotalSeqLen={sl_cfg.label}"
                    )
                    print(f"Exception: {e}")
                    cleanup_cuda()
                    continue

        cleanup_cuda()

    print("\nBenchmark Finished.")


if __name__ == "__main__":
    main()
