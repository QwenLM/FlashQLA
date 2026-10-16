# SM120 Dual-CP Forward validation

Base: `5538c35`, branch: `feat/sm120-dual-cp-fwd`.

The SM120 scan shares the existing `tilelang_correct_h0` state update. Its new
`store_inter_h` mode consumes **all** segments and writes the exiting state;
the default mode still writes segment entrances and does not consume the last
segment. A separate column-tiled `tilelang_aggregate_m` computes the ordered
transition product. This keeps fp32 shared-memory allocations within SM120's
budget without adopting Hopper's warp-specialized/TMA pipeline. `compute_m=False`
omits that launch. There are no changes to AutoCP, communication, or the main
GDN Forward kernel, nor to the SM90/SM100/SM103 kernel implementations.

For each raw sequence, in K/V layout, start with `h = raw_h0` (or zero), `M = I`.
For each segment, including the last:

- `fallback=True`: `h = ht + mt @ h`, `M = mt @ M`.
- `fallback=False`: `h = ht`, `M = 0`. An undefined `mt` must not be read.
- In V/K layout the state update is `h = ht + h @ mt.T`; M's layout is unchanged.
- Once a segment resets M, subsequent finite transitions leave it zero. The
  product order matters because segment matrices generally do not commute.

As in the SM90 wrapper, the seed and both card outputs use `ht_buffer.dtype`.
The h accumulator is fp32; its GEMM operand and the running M are rounded to
`mt_buffer.dtype`. `correct_initial_states` retains its existing fp32 output
when no seed is supplied, or the supplied seed's dtype otherwise.

The inter-card boundary override forces full warmup and fallback on the first
and last local raw sequences. This includes the single-sequence case and each
boundary sequence's terminal segment. Interior terminal masks still follow the
existing warmup semantics: their summaries describe the effective masked affine
map, not an unconditionally exact terminal state with arbitrary incoming history.
The main forward uses the corrected segment entrance states for those sequences.

Dual-CP capabilities are checked separately for forward and backward. SM120's
ordinary backward support is preserved; Dual-CP backward is rejected before
prepare kernels or collectives. Reverse card aggregation is also rejected.

## Local results (2026-10-16)

Actual device: one NVIDIA GeForce RTX 5070 Ti **Laptop GPU**, compute capability
12.0, 12227 MiB reported by `nvidia-smi`. Environment: Python 3.10.20,
PyTorch 2.13.0+cu130, TileLang 0.1.12, pytest 9.1.1.

Activate the existing environment before running the commands:

```bash
conda activate FlashQLA
python -m pytest tests/test_sm120_dual_cp_fwd.py -x -q
python -m pytest tests/test_cp_kernels.py -k 'not aggregate or fwd' -q
python -m pytest tests/test_gdr_unit.py -k test_fwd -q
python -m pytest tests/test_cp_e2e.py -k dual_cp_forward --collect-only -q
```

| Command target | Actual result | Time |
| --- | --- | --- |
| New SM120 tests | 89 passed, no skips | 82.16 s |
| Existing CP support tests | 304 passed, 48 deselected | 33.39 s |
| Existing Forward tests | 39 passed, 43 deselected | 51.56 s |
| Forward-only distributed E2E collection | 20 collected, 52 deselected; **not executed** | 2.32 s |

The 432 passing local tests include:

- 72 independent float64 homogeneous-matrix oracle cases for h_card/M_card:
  both layouts; absent, explicitly zero, and nonzero seeds; all/none/mixed
  fallback; fp32, bf16, fp16, and mixed buffers; singleton and uneven segment
  groups; undefined M filled with NaN; exact reset-to-zero assertions; identical
  h with `compute_m` enabled/disabled.
- 8 real `prepare_h` integration cases, including a 7-token singleton and a
  13-token last segment, checked against an independent tokenwise GDN recurrence
  for both card boundary sequences, plus a masked affine oracle for every group.
- 8 forced pure-intra Forward cases, cached/uncached, both layouts and seed modes,
  checking output and final states against the existing float64 GDN reference.
- Early rejection of Dual-CP backward; existing initial/terminal state correction
  and warmup regressions; 36 existing Forward correctness cases and 3 existing
  determinism cases with 1000 iterations each.

The excluded CP cases are reverse **card aggregation**, which is outside this
Forward-only change; existing terminal-state correction tests were executed.
No failing case was skipped or tolerance relaxed. All runs emitted only the
existing 14 `torch.jit.script_method` deprecation warnings. These durations are
test-suite timings, not performance measurements.

## PR review follow-up

The follow-up keeps the runtime change in three files: the SM120 aggregation
kernel, the chunk-level direction capabilities, and the CP entry guard. Kernel
naming, dtype arguments, and wrapping follow the surrounding code. The existing
E2E process-group timeout is preserved; the new final-state assertion applies
only to the Forward-only cases. No whole-file formatting was applied.

After these edits, all 89 new tests and 304 existing CP support tests passed
again. The 39 existing Forward tests passed in a fresh process (26.69 s).
Syntax compilation and `git diff --check` passed. Full pre-commit/Ruff was not
run because Ruff is not installed in the local environment.

An additional, broader run accidentally selected ordinary Backward tests too:
437 passed and 38 failed. Its first failure was a CUDA unknown error in the
float64 backward reference for `test_bwd[no_h0-kv-dense-long]`; subsequent tests
failed in the same CUDA process. No failure was suppressed or marked to skip.
The same isolated test passed on the modified tree (13.73 s). On an isolated
copy of base `5538c35`, it failed with a different error (`dv_qla` relative
error 0.301107, tolerance 0.02). These results do not establish the root cause
of the CUDA error. The broader backward run is **not** reported as passing;
that instability remains outside this Forward-only PR's verified scope.

## Two-GPU verification

Use two SM120 GPUs with NCCL and the same supported software environment. Check
that both devices are visible, then run the full Forward-only matrix:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -c 'import torch; assert torch.cuda.device_count() == 2; assert all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(2)); print([torch.cuda.get_device_name(i) for i in range(2)])'
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/test_cp_e2e.py -k dual_cp_forward -v -s
```

Pytest spawns one process per GPU; do not wrap this command in `torchrun`.
Expected: **20 passed with zero skips**. Every case forces and asserts actual
intra partitioning, runs the public forward API under `torch.no_grad()`, and
compares each rank's output and all sequence-final states owned by that rank
against a global float64 reference. Cases cover both layouts, zero/nonzero
seeds, full fallback, decay, card-aligned/ragged boundaries, multiple sequences,
and GVA. No backward kernel or backward oracle is called in this suite.

Unverified locally: real multi-GPU communication/E2E correctness and performance,
SM121 hardware, and execution on SM90/SM100/SM103. Dual-CP Backward remains
unsupported on SM120. Multi-GPU performance optimization is deferred until the
above correctness matrix passes on two actual GPUs.
