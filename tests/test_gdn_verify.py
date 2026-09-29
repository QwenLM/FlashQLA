# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Correctness + speed tests for the fused single-chunk GDN verify forward kernel.

Correctness compares ``gdn_verify_fwd`` against a vectorized recurrent reference
for the Qwen4-120B GDN geometry (HQ=16, HV=48, DK=DV=128) with verify-block
sequence lengths (<= chunk_size), including ragged per-seq lengths and both io
dtypes.

The speed benchmark (``python tests/test_gdn_verify.py --bench``) times the same
verify workloads against FlashQLA's chunked-prefill path and the chunk /
fused_recurrent references (FlashInfer when installed, else FLA). Backends that
cannot be imported are skipped automatically. Add ``--cuda-graph`` to measure
GPU latency without Python launch gaps, using 100 ms of graph replays.
"""

import importlib
import sys
from typing import Optional, Tuple

import torch

try:  # pytest is optional so this file also runs standalone via __main__.
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    class _Mark:
        def __getattr__(self, _name):
            def _decorator(*_a, **_k):
                if len(_a) == 1 and callable(_a[0]) and not _k:
                    return _a[0]
                def _wrap(fn):
                    return fn
                return _wrap
            return _decorator

    class _PytestShim:
        mark = _Mark()

        @staticmethod
        def parametrize(_argnames, _argvalues):
            def _wrap(fn):
                return fn
            return _wrap

    pytest = _PytestShim()

from flash_qla.ops.gated_delta_rule.chunk.blackwell import gdn_verify_fwd

# Geometry shared by the correctness cases and the speed benchmark.
GEOM = (16, 48, 128, 128)  # HQ, HV, DK, DV

# Optional backends for the speed benchmark. ``qla_chunk`` is FlashQLA's own
# chunked-prefill path; the ``chunk`` / ``fused_recurrent`` references resolve
# from FlashInfer when present and otherwise fall back to FLA (same algorithms).
# Imported defensively so the correctness tests never break on a missing library.
try:
    from flash_qla import chunk_gated_delta_rule_fwd as _qla_chunk_fwd
    _HAS_QLA_CHUNK = True
except Exception:  # pragma: no cover
    _qla_chunk_fwd = None
    _HAS_QLA_CHUNK = False


def _resolve_first(candidates):
    """Return (provider, fn) for the first importable candidate, else None."""
    for prov, modpath, fnname in candidates:
        try:
            return prov, getattr(importlib.import_module(modpath), fnname)
        except Exception:
            continue
    return None


_CHUNK_PROVIDER = _resolve_first([
    ("flashinfer", "flashinfer.gdn_prefill", "chunk_gated_delta_rule"),
    ("fla", "fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
])
_RECURRENT_PROVIDER = _resolve_first([
    ("flashinfer", "flashinfer.gdn_decode", "fused_recurrent_gated_delta_rule"),
    ("fla", "fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
])


def gdn_recurrent_ref(
    q: torch.Tensor,          # [total, HQ, DK]
    k: torch.Tensor,          # [total, HQ, DK]
    v: torch.Tensor,          # [total, HV, DV]
    g: torch.Tensor,          # [total, HV]  log gate
    beta: torch.Tensor,       # [total, HV]
    cu_seqlens: torch.Tensor, # [B+1]
    initial_state: Optional[torch.Tensor] = None,  # [B, HV, DK, DV]
    scale: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Post-update-state / q-pre-scaled recurrent reference (FLA convention)."""
    total, hq, dk = q.shape
    hv, dv = v.shape[1], v.shape[2]
    group = hv // hq
    if scale is None:
        scale = dk ** -0.5
    dev = q.device
    cu = cu_seqlens.to(torch.int64).tolist()
    B = len(cu) - 1

    q = q.to(dtype) * scale
    k = k.to(dtype)
    v = v.to(dtype)
    g = g.to(dtype)
    beta = beta.to(dtype)

    qh = q.repeat_interleave(group, dim=1)
    kh = k.repeat_interleave(group, dim=1)

    if initial_state is None:
        state = torch.zeros(B, hv, dk, dv, dtype=dtype, device=dev)
    else:
        state = initial_state.to(dtype).clone()

    o = torch.zeros(total, hv, dv, dtype=dtype, device=dev)

    for b in range(B):
        bos, eos = cu[b], cu[b + 1]
        s = state[b]
        for i in range(bos, eos):
            gate = torch.exp(g[i]).unsqueeze(-1).unsqueeze(-1)
            s = s * gate
            ki = kh[i].unsqueeze(-1)
            pred = torch.matmul(s.transpose(-1, -2), ki).squeeze(-1)
            u = beta[i].unsqueeze(-1) * (v[i] - pred)
            s = s + ki * u.unsqueeze(-2)
            qi = qh[i].unsqueeze(-1)
            o[i] = torch.matmul(s.transpose(-1, -2), qi).squeeze(-1)
        state[b] = s

    return o.to(q.dtype), state


def make_inputs(B, seqlens, HQ, HV, DK, DV, io_dtype, device, seed=0, with_state=True):
    gen = torch.Generator(device=device).manual_seed(seed)
    total = sum(seqlens)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)

    def rnd(*shape, dtype=io_dtype, scale=1.0):
        return (
            torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
            * scale
        ).to(dtype)

    q = rnd(total, HQ, DK)
    k = torch.nn.functional.normalize(
        rnd(total, HQ, DK, dtype=torch.float32), dim=-1
    ).to(io_dtype)
    v = rnd(total, HV, DV)
    g = torch.nn.functional.logsigmoid(
        rnd(total, HV, dtype=torch.float32, scale=2.0)
    ).to(torch.float32)
    beta = torch.sigmoid(
        rnd(total, HV, dtype=torch.float32, scale=2.0)
    ).to(torch.float32)
    state = None
    if with_state:
        state = rnd(B, HV, DK, DV, dtype=torch.float32, scale=0.5)
    return q, k, v, g, beta, cu, state


def rel_err(a, b):
    a = a.float()
    b = b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-6)


CASES = [
    (1, [16], torch.bfloat16, True, 0),
    (1, [1], torch.bfloat16, True, 1),
    (4, [16, 16, 16, 16], torch.bfloat16, True, 2),
    (3, [9, 16, 3], torch.bfloat16, True, 3),
    (8, [1, 3, 16, 2, 5, 7, 4, 6], torch.bfloat16, True, 4),
    (4, [16, 16, 16, 16], torch.bfloat16, False, 5),
    (2, [16, 6], torch.float16, True, 6),
    (3, [0, 16, 7], torch.float16, False, 7),
    (8, [16] * 8, torch.float16, True, 8),
]


@pytest.mark.gpu
@pytest.mark.parametrize("B,seqlens,io_dtype,with_state,seed", CASES)
def test_gdn_verify_fwd(B, seqlens, io_dtype, with_state, seed):
    device = "cuda"
    HQ, HV, DK, DV = 16, 48, 128, 128
    q, k, v, g, beta, cu, state = make_inputs(
        B, seqlens, HQ, HV, DK, DV, io_dtype, device, seed=seed, with_state=with_state
    )
    o_ref, s_ref = gdn_recurrent_ref(
        q, k, v, g, beta, cu, initial_state=state, dtype=torch.float32
    )
    o, s = gdn_verify_fwd(
        q, k, v, g, beta, cu,
        initial_state=(state.clone() if state is not None else None),
    )
    torch.cuda.synchronize()
    eo = rel_err(o, o_ref)
    es = rel_err(s, s_ref)
    thr_o = 3e-2 if io_dtype == torch.bfloat16 else 1e-2
    thr_s = 6e-2 if io_dtype == torch.bfloat16 else 2e-2
    assert eo < thr_o, f"o rel err {eo:.3e} >= {thr_o:.1e}"
    assert es < thr_s, f"state rel err {es:.3e} >= {thr_s:.1e}"


@pytest.mark.gpu
@pytest.mark.parametrize("io_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("B", [3, 8])
def test_gdn_verify_preallocated(B, io_dtype):
    # Weak decay exposes errors that strong forgetting can hide. Also exercise
    # both value-tile sizes, empty sequences, custom scale, and in-place state.
    lengths = [0, 16, 1] if B == 3 else [0, 16, 1, 2, 3, 8, 15, 0]
    q, k, v, g, beta, cu, state = make_inputs(
        B, lengths, *GEOM, io_dtype, "cuda", seed=19,
    )
    g.zero_()
    beta.fill_(1)
    scale = 0.17
    o_ref, s_ref = gdn_recurrent_ref(
        q, k, v, g, beta, cu, initial_state=state, scale=scale,
    )
    out = torch.full_like(v, float("nan"))
    o, s = gdn_verify_fwd(
        q, k, v, g, beta, cu, initial_state=state, scale=scale,
        final_state=state, out=out,
    )
    assert o is out and s is state
    assert rel_err(o, o_ref) < 5e-3
    assert rel_err(s, s_ref) < 3e-3


@pytest.mark.gpu
@pytest.mark.parametrize("with_state", [False, True])
def test_gdn_verify_empty_sequences(with_state):
    q, k, v, g, beta, cu, state = make_inputs(
        2, [0, 0], *GEOM, torch.bfloat16, "cuda", with_state=with_state,
    )
    o, s = gdn_verify_fwd(q, k, v, g, beta, cu, initial_state=state)
    assert o.shape == v.shape
    expected = state if with_state else torch.zeros_like(s)
    torch.testing.assert_close(s, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("io_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("B", [4, 8])
def test_gdn_verify_zero_state_repeat(B, io_dtype):
    data = make_inputs(B, [16] * B, *GEOM, io_dtype, "cuda", with_state=False)
    q, k, v, g, beta, cu, _ = data
    out = torch.empty_like(v)
    state = torch.empty(B, GEOM[1], GEOM[2], GEOM[3], device="cuda")
    o_ref, s_ref = gdn_recurrent_ref(q, k, v, g, beta, cu)
    gdn_verify_fwd(q, k, v, g, beta, cu, out=out, final_state=state)
    assert rel_err(out, o_ref) < 5e-3
    assert rel_err(state, s_ref) < 3e-5
    expected_o, expected_s = out.clone(), state.clone()
    for _ in range(10):
        out.fill_(float("nan"))
        state.fill_(float("nan"))
        gdn_verify_fwd(q, k, v, g, beta, cu, out=out, final_state=state)
        assert torch.equal(out, expected_o)
        assert torch.equal(state, expected_s)


def run_correctness():
    """Run every CASES entry against the recurrent reference; return all-pass."""
    print(f"{'B':>3} {'seqlens':<26} {'dtype':<10} {'state':<6} "
          f"{'o_relerr':>10} {'s_relerr':>10}  verdict")
    ok = True
    for B, seqlens, io_dtype, with_state, seed in CASES:
        device = "cuda"
        HQ, HV, DK, DV = GEOM
        q, k, v, g, beta, cu, state = make_inputs(
            B, seqlens, HQ, HV, DK, DV, io_dtype, device, seed=seed, with_state=with_state
        )
        o_ref, s_ref = gdn_recurrent_ref(
            q, k, v, g, beta, cu, initial_state=state, dtype=torch.float32
        )
        o, s = gdn_verify_fwd(
            q, k, v, g, beta, cu,
            initial_state=(state.clone() if state is not None else None),
        )
        torch.cuda.synchronize()
        eo, es = rel_err(o, o_ref), rel_err(s, s_ref)
        thr_o = 3e-2 if io_dtype == torch.bfloat16 else 1e-2
        thr_s = 6e-2 if io_dtype == torch.bfloat16 else 2e-2
        good = eo < thr_o and es < thr_s
        ok = ok and good
        print(f"{B:>3} {str(seqlens):<26} {str(io_dtype).split('.')[-1]:<10} "
              f"{str(with_state):<6} {eo:>10.2e} {es:>10.2e}  "
              f"{'PASS' if good else 'FAIL'}")
    for B in (3, 8):
        for io_dtype in (torch.bfloat16, torch.float16):
            test_gdn_verify_preallocated(B, io_dtype)
            print(f"preallocated/in-place, zero gates, B={B}, {io_dtype}: PASS")
    for with_state in (False, True):
        test_gdn_verify_empty_sequences(with_state)
        print(f"empty sequences, initial_state={with_state}: PASS")
    for B in (4, 8):
        for io_dtype in (torch.bfloat16, torch.float16):
            test_gdn_verify_zero_state_repeat(B, io_dtype)
            print(f"zero-state repeat, B={B}, {io_dtype}: PASS")
    return ok


# --------------------------------------------------------------------------
# Speed benchmark
#
# Compares the fused single-chunk verify kernel against, on identical verify
# workloads (uniform per-seq length T <= 16):
#   * qla_verify           -- this kernel (gdn_verify_fwd)
#   * qla_chunk            -- FlashQLA's chunked-prefill path (chunk_size=64)
#   * chunk[<prov>]        -- FlashInfer chunked prefill (falls back to FLA)
#   * fused_recurrent[<p>] -- FlashInfer/FLA fused recurrent path
# Every backend is optional: it is probed once and dropped if unavailable.
# --------------------------------------------------------------------------

BENCH_CONFIGS = [
    (16, 8, torch.bfloat16),
    (32, 8, torch.bfloat16),
    (64, 8, torch.bfloat16),
    (16, 16, torch.bfloat16),
    (64, 16, torch.bfloat16),
    (128, 16, torch.bfloat16),
]


def time_call(run, iters, warmup, cuda_graph=False):
    """Return microseconds per call; graph mode excludes Python launch gaps."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    if cuda_graph:
        from triton.testing import do_bench_cudagraph
        return do_bench_cudagraph(run, rep=100) * 1e3
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def build_backends(data, scale, geom, device):
    """Build {key: {label, run}} for every backend that probes successfully.

    Each ``run()`` executes one forward and returns ``(o, s)`` in a canonical
    layout: ``o`` as [total, HV, DV] and ``s`` as [B, HV, DK, DV].
    """
    q, k, v, g, beta, cu, state = data
    HQ, HV, DK, DV = geom
    B = cu.numel() - 1
    total = q.shape[0]
    q4 = q.view(1, total, HQ, DK)
    k4 = k.view(1, total, HQ, DK)
    v4 = v.view(1, total, HV, DV)
    g4 = g.view(1, total, HV)
    b4 = beta.view(1, total, HV)

    backs = {}

    def add(key, label, run):
        try:
            run()
            torch.cuda.synchronize()
            backs[key] = {"label": label, "run": run}
        except Exception as ex:  # pragma: no cover
            print(f"[bench] backend '{label}' unavailable: {repr(ex)[:140]}")

    # 1) This kernel.
    o_v = torch.empty_like(v)
    s_v = torch.empty(B, HV, DK, DV, dtype=torch.float32, device=device)

    def run_verify():
        return gdn_verify_fwd(
            q, k, v, g, beta, cu,
            initial_state=state, scale=scale, final_state=s_v, out=o_v,
        )

    add("verify", "qla_verify", run_verify)

    # 2) FlashQLA's chunked-prefill path.
    if _HAS_QLA_CHUNK:
        def run_qla_chunk():
            out = _qla_chunk_fwd(
                q4, k4, v4, g4, b4, scale=scale, initial_state=state,
                cu_seqlens=cu, output_final_state=True, output_h=False,
                auto_cp=False,
            )
            return out[2].view(total, HV, DV), out[4]
        add("qla_chunk", "qla_chunk", run_qla_chunk)

    # 3) chunk reference (FlashInfer else FLA).
    if _CHUNK_PROVIDER is not None:
        cprov, cfn = _CHUNK_PROVIDER
        if cprov == "flashinfer":
            # FlashInfer's chunked prefill consumes the multiplicative gate
            # (alpha = exp(log_gate)) and a v-major initial state [B,HV,DV,DK].
            fi_alpha = g.exp()
            fi_init = state.transpose(-1, -2).contiguous() if state is not None else None
            fi_out = torch.empty_like(v)
            fi_state = torch.empty(B, HV, DV, DK, dtype=torch.float32, device=device)

            def run_chunk():
                r = cfn(q, k, v, g=fi_alpha, beta=beta, scale=scale,
                        initial_state=fi_init, output_final_state=True,
                        cu_seqlens=cu, use_qk_l2norm_in_kernel=False,
                        output=fi_out, output_state=fi_state)
                o = r[0] if isinstance(r, (tuple, list)) else fi_out
                s = r[1] if isinstance(r, (tuple, list)) and len(r) > 1 else fi_state
                return o.reshape(total, HV, DV), s.transpose(-1, -2)
        else:
            def run_chunk():
                r = cfn(q4, k4, v4, g=g4, beta=b4, scale=scale,
                        initial_state=state, output_final_state=True,
                        cu_seqlens=cu)
                return r[0].view(total, HV, DV), r[1]
        add("chunk", f"chunk[{cprov}]", run_chunk)

    # 4) fused_recurrent reference (FlashInfer else FLA).
    if _RECURRENT_PROVIDER is not None:
        rprov, rfn = _RECURRENT_PROVIDER

        def run_recurrent():
            r = rfn(q4, k4, v4, g=g4, beta=b4, scale=scale,
                    initial_state=state, output_final_state=True,
                    cu_seqlens=cu)
            return r[0].view(total, HV, DV), r[1]
        add("recurrent", f"fused_recurrent[{rprov}]", run_recurrent)

    return backs


def bench_speed_one(B, T, io_dtype, device="cuda", iters=200, warmup=20,
                    geom=GEOM, check=True, cuda_graph=False):
    data = make_inputs(B, [T] * B, *geom, io_dtype, device, seed=0, with_state=True)
    scale = geom[2] ** -0.5
    backs = build_backends(data, scale, geom, device)
    ref = backs["verify"]["run"]() if "verify" in backs else None
    out = {}
    for key, b in backs.items():
        eo = es = float("nan")
        if check and ref is not None and key != "verify":
            try:
                o, s = b["run"]()
                eo, es = rel_err(o, ref[0]), rel_err(s, ref[1])
            except Exception:  # pragma: no cover
                pass
        out[key] = {"label": b["label"], "us": time_call(b["run"], iters, warmup, cuda_graph),
                    "eo": eo, "es": es}
    return out


def run_benchmark(configs=BENCH_CONFIGS, iters=200, warmup=20, cuda_graph=False):
    if not torch.cuda.is_available():
        print("CUDA not available for benchmark.")
        return
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"geom: HQ={GEOM[0]} HV={GEOM[1]} DK={GEOM[2]} DV={GEOM[3]}, "
          f"uniform T<=16 verify blocks, iters={iters} warmup={warmup}")
    print(f"timing: {'CUDA Graph' if cuda_graph else 'CUDA events (includes host launch gaps)'}")
    print(f"backends: qla_chunk={_HAS_QLA_CHUNK} "
          f"chunk={_CHUNK_PROVIDER[0] if _CHUNK_PROVIDER else None} "
          f"recurrent={_RECURRENT_PROVIDER[0] if _RECURRENT_PROVIDER else None}")

    all_res, order = [], []
    for B, T, dt in configs:
        res = bench_speed_one(B, T, dt, device, iters, warmup, cuda_graph=cuda_graph)
        for k in res:
            if k not in order:
                order.append(k)
        all_res.append((B, T, dt, res))

    labels = [all_res[0][3].get(k, {}).get("label", k) for k in order]

    print("\n=== Latency (ms); (xN) = qla_verify speedup over that backend ===")
    hdr = f"{'B':>4} {'T':>3} {'dtype':<9} " + "".join(f"{lab:>22}" for lab in labels)
    print(hdr)
    print("-" * len(hdr))
    for B, T, dt, res in all_res:
        vus = res.get("verify", {}).get("us")
        row = f"{B:>4} {T:>3} {str(dt).split('.')[-1]:<9} "
        for k in order:
            e = res.get(k)
            if e is None:
                row += f"{'N/A':>22} "
            elif k == "verify" or not vus:
                row += f"{e['us'] / 1000:>22.3f} "
            else:
                row += f"{('%0.3f (x%.2f)' % (e['us'] / 1000, e['us'] / vus)):>22} "
        print(row, flush=True)

    print("\n=== Correctness vs qla_verify (o_relerr / s_relerr) ===")
    chdr = f"{'B':>4} {'T':>3} {'dtype':<9} " + "".join(f"{lab:>22}" for lab in labels)
    print(chdr)
    print("-" * len(chdr))
    for B, T, dt, res in all_res:
        row = f"{B:>4} {T:>3} {str(dt).split('.')[-1]:<9} "
        for k in order:
            e = res.get(k)
            if e is None or k == "verify":
                row += f"{'-':>22} "
            else:
                row += f"{('o%.1e/s%.1e' % (e['eo'], e['es'])):>22} "
        print(row, flush=True)


@pytest.mark.slow
@pytest.mark.gpu
def test_gdn_verify_bench_smoke():
    """Smoke-run the benchmark on one small config to keep it from rotting."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    run_benchmark([(4, 16, torch.bfloat16)], iters=20, warmup=5)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--bench" in args:
        iters = int(args[args.index("--iters") + 1]) if "--iters" in args else 200
        warmup = int(args[args.index("--warmup") + 1]) if "--warmup" in args else 20
        run_benchmark(BENCH_CONFIGS, iters=iters, warmup=warmup,
                      cuda_graph="--cuda-graph" in args)
    else:
        ok = run_correctness()
        print("\nALL PASS" if ok else "\nSOME FAILED")
        raise SystemExit(0 if ok else 1)
