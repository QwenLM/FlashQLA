# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Numerical correctness tests for flash_qla_mlx (Apple Silicon / MLX).
# Compares against a PyTorch CPU reference.  No CUDA required.
#
# Run:
#   python tests/test_mlx.py
#
# Design: two sequential subprocesses share a temp directory.
#   --gen-ref  DIR : PyTorch-only process — computes reference, writes .npy files
#   --check-mlx DIR: MLX-only process     — loads .npy files, runs flash_qla_mlx
# Loading PyTorch and MLX in the same process causes macOS memory conflicts
# (OMP dual-runtime + Metal/CPU allocator interference), so they run separately.

import os
import sys
import subprocess
import tempfile
# On macOS with Python 3.14 + PyTorch 2.11, numpy's OMP runtime must not load
# before torch's OMP runtime — the reverse order causes memory corruption on
# cumsum and other ops. Pre-load torch first when we're in gen-ref mode.
if "--gen-ref" in sys.argv:
    import torch as _torch_preload  # noqa: F401 — sets up OMP before numpy
import numpy as np

# ===========================================================================
# Shared test-case definitions  (used in both sub-modes)
# ===========================================================================

CASES = [
    # forward + backward
    dict(B=2, T=256, Hk=4, Hv=4, h0=False, seed=0),
    dict(B=1, T=256, Hk=4, Hv=8, h0=False, seed=1),   # GQA
    dict(B=2, T=128, Hk=4, Hv=4, h0=True,  seed=2),
    dict(B=1, T=192, Hk=2, Hv=2, h0=False, seed=3),   # T not divisible by 64
    dict(B=1, T=128, Hk=2, Hv=2, h0=False, seed=5, bwd=True),
    dict(B=1, T=128, Hk=2, Hv=4, h0=False, seed=6, bwd=True),  # GQA bwd
    dict(B=1, T=128, Hk=2, Hv=2, h0=True,  seed=7, bwd=True),
]
K_DIM = V_DIM = 128


def make_inputs(rng, B, T, Hk, Hv, with_h0, with_bwd=False):
    q    = rng.standard_normal((B, T, Hk, K_DIM)).astype(np.float32)
    k    = rng.standard_normal((B, T, Hk, K_DIM)).astype(np.float32)
    # L2-normalize k so KKT triangular solve stays stable in float32.
    k   /= np.linalg.norm(k, axis=-1, keepdims=True) + 1e-8
    v    = rng.standard_normal((B, T, Hv, V_DIM)).astype(np.float32)
    g    = (-np.log1p(np.exp(-rng.standard_normal((B, T, Hv)).astype(np.float32))) / 16)
    beta = (1 / (1 + np.exp(-rng.standard_normal((B, T, Hv)).astype(np.float32))))
    h0   = (rng.standard_normal((B, Hv, K_DIM, V_DIM)).astype(np.float32)
            if with_h0 else None)
    do   = (rng.standard_normal((B, T, Hv, V_DIM)).astype(np.float32)
            if with_bwd else None)
    dht  = (rng.standard_normal((B, Hv, K_DIM, V_DIM)).astype(np.float32)
            if (with_bwd and with_h0) else None)
    return q, k, v, g, beta, h0, do, dht


# ===========================================================================
# Mode: --gen-ref  (pure PyTorch, no MLX)
# ===========================================================================

def _gen_ref(data_dir):
    import torch

    def pad(x, dim, cs=64):
        amt = (cs - x.shape[dim] % cs) % cs
        if amt > 0:  # F.pad with 6-tuple on 4D tensors segfaults in Py3.14 + torch2.11 when amt==0
            zeros = [0] * (2 * (x.dim() - 1 - dim))
            x = torch.nn.functional.pad(x, (*zeros, 0, amt))
        return x.reshape(list(x.shape[:dim]) + [-1, cs] + list(x.shape[dim + 1:]))

    def fill_g(g, T, cs=64):
        lcs = T % cs
        if lcs:
            g = g.clone()
            g[:, -1, lcs:] = g[:, -1, lcs - 1:lcs]
        return g

    def ref_fwd(q, k, v, g, beta, scale, h0=None, cs=64):
        B, T, Hk, K = q.shape
        _, _, Hv, V = v.shape
        g = pad(g, 1, cs).cumsum(2).reshape(B, -1, Hv)[:, :T]
        if Hk != Hv:
            k = k.repeat_interleave(Hv // Hk, dim=2)
        kc = pad(k, 1, cs); gc = pad(g, 1, cs); bc = pad(beta, 1, cs)
        mask_u = torch.triu(torch.ones(cs, cs, dtype=torch.bool))
        decay = torch.exp(gc[:, :, :, None, :] - gc[:, :, None, :, :])
        decay = decay.masked_fill(mask_u[None, None, :, :, None], 0.0)
        A = (torch.einsum("bnchk,bndhk->bnchd", kc * bc.unsqueeze(-1), kc)
             * decay.swapaxes(-2, -1)).reshape(B, -1, Hv, cs)[:, :T]
        Ac = -pad(A, 1, cs).swapaxes(2, 3)
        for i in range(1, cs):
            row = Ac[..., i, :i].clone()
            sub = Ac[..., :i, :i].clone()
            Ac[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        Ac = Ac + torch.eye(cs, dtype=Ac.dtype)
        A = Ac.swapaxes(2, 3).reshape(B, -1, Hv, cs)[:, :T]
        Ac2 = pad(A, 1)
        kb = pad(k * (beta * g.exp()).unsqueeze(-1), 1)
        vb = pad(v * beta.unsqueeze(-1), 1)
        w = torch.einsum("bnchd,bndhk->bnchk", Ac2, kb).reshape(B, -1, Hv, K)[:, :T]
        u = torch.einsum("bnchd,bndhv->bnchv", Ac2, vb).reshape(B, -1, Hv, V)[:, :T]
        kc2 = pad(k, 1, cs); wc = pad(w, 1, cs); uc = pad(u, 1, cs)
        gc2 = fill_g(pad(g, 1, cs), T, cs)
        hs = (torch.zeros(B, Hv, K, V, dtype=g.dtype)
              if h0 is None else h0.to(g.dtype))
        hl, vl = [], []
        for i in range(kc2.shape[1]):
            hl.append(hs.clone())
            vn = uc[:, i] - torch.einsum("bchk,bhkv->bchv", wc[:, i], hs)
            vl.append(vn)
            hs = hs * gc2[:, i, -1, :].exp()[:, :, None, None]
            hs = hs + torch.einsum(
                "bchk,bchv->bhkv",
                kc2[:, i] * (gc2[:, i, -1:, :, None] - gc2[:, i, :, :, None]).exp(),
                vn,
            )
        hc = torch.stack(hl, 1)
        vn = torch.stack(vl, 1).reshape(B, -1, Hv, V)[:, :T]
        qr = q.repeat_interleave(Hv // Hk, dim=2) if Hk != Hv else q
        kr = k
        qc = pad(qr, 1, cs) * scale; kc3 = pad(kr, 1, cs)
        vnc = pad(vn, 1, cs); gc3 = pad(g, 1, cs)
        mask_o = torch.triu(torch.ones(cs, cs, dtype=torch.bool), diagonal=1)
        dec = torch.exp(gc3[:, :, :, None, :] - gc3[:, :, None, :, :])
        dec = dec.masked_fill(mask_o[None, None, :, :, None], 0.0)
        at = torch.einsum("bnchk,bndhk->bncdh", qc, kc3) * dec
        ai = torch.einsum("bnchk,bnhkv->bnchv", qc * gc3.exp().unsqueeze(-1), hc)
        o = (ai + torch.einsum("bncdh,bndhv->bnchv", at, vnc)).reshape(B, -1, Hv, V)[:, :T]
        return g, o, A, hs

    for idx, case in enumerate(CASES):
        B, T, Hk, Hv = case["B"], case["T"], case["Hk"], case["Hv"]
        rng = np.random.default_rng(case["seed"])
        scale = K_DIM ** -0.5
        q, k, v, g, beta, h0, do, dht = make_inputs(
            rng, B, T, Hk, Hv, case["h0"], case.get("bwd", False)
        )
        tq = torch.from_numpy(q.copy()); tk = torch.from_numpy(k.copy())
        tv = torch.from_numpy(v.copy()); tg = torch.from_numpy(g.copy())
        tb = torch.from_numpy(beta.copy())
        th0 = torch.from_numpy(h0.copy()) if h0 is not None else None
        g_ref, o_ref, A_ref, s_ref = ref_fwd(tq, tk, tv, tg, tb, scale, th0)
        d = {
            "case": case,
            "inputs": dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0),
            "fwd": dict(
                g_cumsum=g_ref.numpy(), o=o_ref.numpy(),
                A=A_ref.numpy(), s=s_ref.numpy(),
            ),
        }
        if case.get("bwd"):
            tq2 = torch.from_numpy(q.copy()).requires_grad_(True)
            tk2 = torch.from_numpy(k.copy()).requires_grad_(True)
            tv2 = torch.from_numpy(v.copy()).requires_grad_(True)
            tg2 = torch.from_numpy(g.copy()).requires_grad_(True)
            tb2 = torch.from_numpy(beta.copy()).requires_grad_(True)
            th02 = (torch.from_numpy(h0.copy()).requires_grad_(True)
                    if h0 is not None else None)
            _, o2, _, s2 = ref_fwd(tq2, tk2, tv2, tg2, tb2, scale, th02)
            loss = (o2 * torch.from_numpy(do.copy())).sum()
            if dht is not None:
                loss = loss + (s2 * torch.from_numpy(dht.copy())).sum()
            loss.backward()
            d["bwd"] = dict(
                do=do, dht=dht,
                dq=tq2.grad.numpy(), dk=tk2.grad.numpy(),
                dv=tv2.grad.numpy(), db=tb2.grad.numpy(), dg=tg2.grad.numpy(),
                dh0=th02.grad.numpy() if th02 is not None else None,
            )
        np.save(os.path.join(data_dir, f"case_{idx}.npy"), d, allow_pickle=True)
    print(f"[gen-ref] wrote {len(CASES)} cases to {data_dir}")


# ===========================================================================
# Mode: --check-mlx  (pure MLX, no PyTorch)
# ===========================================================================

def _check_mlx(data_dir, tol=0.02):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import mlx.core as mx
    from flash_qla_mlx import (
        chunk_gated_delta_rule_fwd as mlx_fwd,
        chunk_gated_delta_rule_bwd as mlx_bwd,
        chunk_gated_delta_rule,
    )

    def _mx(a): return mx.array(a)
    def _np(a): mx.eval(a); return np.array(a)
    def rel(got, ref):
        g = np.asarray(got, dtype=np.float64)
        r = np.asarray(ref, dtype=np.float64)
        e = float(np.abs(g - r).max() / (np.abs(r).max() + 1e-8))
        return e if np.isfinite(e) else float("inf")

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))
    all_passed = True

    for fname in files:
        d    = np.load(os.path.join(data_dir, fname), allow_pickle=True).item()
        case = d["case"]
        inp  = d["inputs"]
        fwd  = d["fwd"]
        q, k, v, g, beta = inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"]
        h0 = inp["h0"]
        B, T, Hk, Hv = case["B"], case["T"], case["Hk"], case["Hv"]
        scale = K_DIM ** -0.5

        g_out, A_out, o_out, _, s_out = mlx_fwd(
            q=_mx(q), k=_mx(k), v=_mx(v), g=_mx(g), beta=_mx(beta),
            scale=scale,
            initial_state=_mx(h0) if h0 is not None else None,
            output_final_state=True,
        )
        mx.eval(g_out, A_out, o_out)
        errs_fwd = {
            "o": rel(_np(o_out), fwd["o"]),
            "g": rel(_np(g_out), fwd["g_cumsum"]),
            "A": rel(_np(A_out), fwd["A"]),
        }
        if h0 is not None and s_out is not None:
            mx.eval(s_out)
            errs_fwd["s"] = rel(_np(s_out), fwd["s"])

        tag = f"B={B} T={T} Hk={Hk} Hv={Hv} h0={case['h0']}"
        status = "PASS"
        for n, e in errs_fwd.items():
            if e > tol:
                status = f"FAIL({n}={e:.4f})"
                all_passed = False
        print(f"  [fwd] {tag}  " + "  ".join(f"{n}={e:.4f}" for n, e in errs_fwd.items())
              + f"  {status}")

        if "bwd" in d:
            bwd = d["bwd"]
            dq_a, dk_a, dv_a, db_a, dg_a, dh0_a = mlx_bwd(
                q=_mx(q), k=_mx(k), v=_mx(v), g=g_out, beta=_mx(beta), A=A_out,
                do=_mx(bwd["do"]),
                dht=_mx(bwd["dht"]) if bwd["dht"] is not None else None,
                scale=scale,
                initial_state=_mx(h0) if h0 is not None else None,
            )
            mx.eval(dq_a, dk_a, dv_a, db_a, dg_a)
            errs_bwd = {
                "dq": rel(_np(dq_a), bwd["dq"]),
                "dk": rel(_np(dk_a), bwd["dk"]),
                "dv": rel(_np(dv_a), bwd["dv"]),
                "db": rel(_np(db_a), bwd["db"]),
                "dg": rel(_np(dg_a), bwd["dg"]),
            }
            if case["h0"] and dh0_a is not None:
                mx.eval(dh0_a)
                errs_bwd["dh0"] = rel(_np(dh0_a), bwd["dh0"])
            status = "PASS"
            for n, e in errs_bwd.items():
                if e > tol:
                    status = f"FAIL({n}={e:.4f})"
                    all_passed = False
            print(f"  [bwd] {tag}  " + "  ".join(f"{n}={e:.4f}" for n, e in errs_bwd.items())
                  + f"  {status}")

    # Autograd smoke: mx.grad(chunk_gated_delta_rule) == mlx_bwd
    rng = np.random.default_rng(42)
    q2, k2, v2, g2, beta2, _, do2, _ = make_inputs(rng, 1, 128, 2, 2, False, True)
    scale2 = K_DIM ** -0.5
    g_o2, A_o2, _, _, _ = mlx_fwd(
        q=_mx(q2), k=_mx(k2), v=_mx(v2), g=_mx(g2), beta=_mx(beta2),
        scale=scale2, output_final_state=False,
    )
    mx.eval(g_o2, A_o2)
    dq_e, dk_e, dv_e, db_e, dg_e, _ = mlx_bwd(
        q=_mx(q2), k=_mx(k2), v=_mx(v2), g=g_o2, beta=_mx(beta2), A=A_o2,
        do=_mx(do2), scale=scale2,
    )
    mx.eval(dq_e, dk_e, dv_e, db_e, dg_e)
    mdo = _mx(do2)
    def loss_fn(q_, k_, v_, g_, b_):
        o, _ = chunk_gated_delta_rule(q_, k_, v_, g_, b_, scale=scale2)
        return (o * mdo).sum()
    dq_ag, dk_ag, dv_ag, dg_ag, db_ag = mx.grad(loss_fn, argnums=(0,1,2,3,4))(
        _mx(q2), _mx(k2), _mx(v2), _mx(g2), _mx(beta2)
    )
    mx.eval(dq_ag, dk_ag, dv_ag, db_ag, dg_ag)
    ag_errs = {
        "dq": rel(_np(dq_e), _np(dq_ag)),
        "dk": rel(_np(dk_e), _np(dk_ag)),
        "dv": rel(_np(dv_e), _np(dv_ag)),
    }
    status = "PASS"
    for n, e in ag_errs.items():
        if e > 0.02:
            status = f"FAIL({n}={e:.4f})"
            all_passed = False
    print(f"  [autograd] " + "  ".join(f"{n}={e:.4f}" for n, e in ag_errs.items())
          + f"  {status}")

    return all_passed


# ===========================================================================
# Orchestrator
# ===========================================================================

def main():
    if "--gen-ref" in sys.argv:
        _gen_ref(sys.argv[sys.argv.index("--gen-ref") + 1])
        return
    if "--check-mlx" in sys.argv:
        ok = _check_mlx(sys.argv[sys.argv.index("--check-mlx") + 1])
        sys.exit(0 if ok else 1)

    print("=" * 64)
    print("flash_qla_mlx  numerical correctness  (PyTorch CPU reference)")
    print("=" * 64)

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    with tempfile.TemporaryDirectory() as tmp:
        print("\n[1/2] PyTorch CPU reference...")
        r = subprocess.run(
            [sys.executable, __file__, "--gen-ref", tmp],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            print("STDOUT:", r.stdout)
            print("STDERR:", r.stderr)
            sys.exit(1)
        print(r.stdout.strip())

        print("\n[2/2] flash_qla_mlx (MLX)...")
        r = subprocess.run(
            [sys.executable, __file__, "--check-mlx", tmp],
            capture_output=True, text=True, env=env,
        )
        print(r.stdout.strip())
        if r.returncode != 0:
            print("STDERR:", r.stderr)
            print("\nSome tests FAILED.")
            sys.exit(1)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
