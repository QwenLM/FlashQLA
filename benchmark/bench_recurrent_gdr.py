# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Benchmark the FlashQLA GDN verify kernel (memory-bound): wall time + achieved HBM
bandwidth vs the theoretical state-I/O floor, across SGLang-relevant regimes."""
import torch

from flash_qla import fused_recurrent_gdr_verify_fwd
from flash_qla.utils import l2norm

PEAK_TBS = 3.35  # H100 HBM3 ~3.35 TB/s


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def bench(N, H, Hg, D, dtype=torch.bfloat16, store_intermediate=True):
    total = N * D
    q = l2norm(torch.randn(1, total, Hg, 128, device="cuda", dtype=dtype))
    k = l2norm(torch.randn(1, total, Hg, 128, device="cuda", dtype=dtype))
    v = torch.randn(1, total, H, 128, device="cuda", dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(1, total, H, device="cuda")) / 16
    beta = torch.randn(1, total, H, device="cuda").sigmoid()
    pool = torch.randn(N, H, 128, 128, device="cuda", dtype=dtype)
    cu = torch.arange(0, total + 1, D, dtype=torch.int32, device="cuda")
    si = torch.arange(N, dtype=torch.int32, device="cuda")
    ci = torch.arange(N, dtype=torch.int32, device="cuda")
    ibuf = (torch.zeros(N + 1, D, H, 128, 128, device="cuda", dtype=dtype)
            if store_intermediate else None)
    o = torch.empty(1, total, H, 128, device="cuda", dtype=dtype)

    fn = lambda: fused_recurrent_gdr_verify_fwd(
        q, k, v, g, beta, pool, si, cu, ibuf, ci, o, disable_state_update=True)
    ms = _time(fn)

    es = 2  # bf16 state element size
    # dominant state I/O: 1 gather + D intermediate writes per (request, head)
    state_bytes = N * H * (1 + D) * 128 * 128 * es
    io_bytes = (q.numel() + k.numel() + v.numel() + o.numel()) * es  # tiny qkvo
    gbs = (state_bytes + io_bytes) / (ms * 1e-3) / 1e9
    print(f"  N={N:<4d} H={H} Hg={Hg} D={D:<2d}  {ms*1e3:8.1f} us   "
          f"{gbs:7.1f} GB/s  ({100*gbs/1000/PEAK_TBS:4.1f}% peak)  state={state_bytes/1e6:6.1f} MB")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name()}  |  verify kernel (bf16 pool, per-token intermediates)")
    print("server / batched decode (H=32, Hg=16):")
    for N in (8, 64, 256, 512):
        for D in (1, 4, 12):
            bench(N, 32, 16, D)
    print("single-request / TP (N=1, varying H = TP1..TP8):")
    for H in (64, 32, 16, 8):
        bench(1, H, max(1, H // 4), 12)
