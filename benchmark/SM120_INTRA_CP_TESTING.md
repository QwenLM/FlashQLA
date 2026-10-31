# SM120 intra-CP correction pipeline

This change adapts the Hopper load-warp state scan to SM120. It keeps the
existing recurrence, output layout, numerical precision and CP interfaces.
It does not change NCCL collectives, forward caches or AutoCP decisions.

The new kernel uses 128 math threads and a 32-thread TMA producer in a
256-thread CTA. M has two stages for BF16/FP16 inputs; h/dh has one buffer,
released after the math threads copy it into registers. Boundaries are stored
directly from registers. Shared data uses 80 KiB for BF16 h/M and 88 KiB for
FP32 dh with BF16 M, plus barrier storage. FP32 M uses one stage (96 KiB).

Production changes are confined to `blackwell_sm120/cp_scan.py` and the two
correction factory calls in `blackwell_sm120/cp_fwd.py`. Card-level aggregation
keeps its original implementation. These correction functions serve intra-CP
and the intra stage of Dual-CP; the original inter-card correction path also
calls them. Its communication protocol is unchanged. SM90/SM100 are unchanged.

## Single RTX 5090 validation

Run in the installed project environment, with only one GPU visible. No
distributed process group is needed. The pytest files and assertions are the
existing repository tests.

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

python -m pytest tests/test_cp_kernels.py \
  -k 'correct_initial or correct_terminal' -q -x
python -m pytest tests/test_gdr_unit.py \
  -k 'not deterministic and not long' -q -x
```

The first command covers 192 correction cases; the second covers 60 existing
operator cases, including forced intra-CP with the forward cache.

One demo provides both kernel and original operator-benchmark comparisons:

```bash
python -m benchmark.demo_intra_cp_scan --heads 16 --segments 16
python -m benchmark.demo_intra_cp_scan --heads 4 --segments 40
python -m benchmark.demo_intra_cp_scan --heads 16 --segments 16 --fallback mixed
python -m benchmark.demo_intra_cp_scan --heads 16 --segments 16 --layout vk
python -m benchmark.demo_intra_cp_scan --heads 16 --tokens 8192
python -m benchmark.demo_intra_cp_scan --heads 4 --tokens 32768
```

`serial` uses the original correction kernel; `ws_tma` uses the new kernel.
The kernel comparison checks all/none/mixed fallback and then measures the
requested pattern, interleaving nine CUDA-graph samples with preallocated
outputs. Forward uses BF16 h/M; backward uses FP32 dh and BF16 M. `--m-dtype
fp32` also checks the single-stage resource configuration.

With `--tokens`, the demo calls `bench_fwd` and `bench_bwd` from the unchanged
`benchmark/bench_gated_delta_rule.py`, alternates implementations over three
rounds and reports medians. It temporarily substitutes only the correction
factory within the demo process. This adds no production environment switch.
Those original benchmarks measure forward and backward separately; forward
setup for the backward benchmark is outside its timed region. Check the printed
AutoCP decision: a case with `intra=False` does not exercise the changed kernel.

## Synchronization checks

```bash
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" \
    --kernel-name kns=correct_h0_ws_kernel --error-exitcode 1 \
    python -m pytest tests/test_cp_kernels.py \
    -k 'correct and deep-split' -q -x
done
```

Sanitizer runs validate accesses and synchronization, not speed. A scan-only
speedup is not the full operator speedup. Local measurements on another SM120
GPU do not establish RTX 5090 performance; validate both benchmark levels there.

## Local validation, 2026-11-01

Hardware: RTX 5070 Ti Laptop, SM120, 46 SMs, 101376-byte shared-memory opt-in
limit. Python 3.11.15, PyTorch 2.11.0+cu128, TileLang 0.1.15. These are NOT
RTX 5090 measurements. No other test process was running during timing.

- Existing correction pytest: 192 passed.
- Existing operator pytest (excluding long/determinism cases): 60 passed.
- Clean-branch regression against base `a14dbf3`: 412 passed (56 CP context,
  352 CP kernel and 4 forced intra-CP cache integration cases).
- Each sanitizer: the existing 48 deep-split correction cases passed.
  memcheck/synccheck: 0 errors; racecheck: 0 hazards, 0 errors, 0 warnings.
- Generated CUDA contains TMA loads, synchronous MMA and separate h/M buffer
  release barriers. No changes to the recurrence or communication protocol.

Preallocated scan timings, median of nine interleaved CUDA-graph samples,
100 calls per replay. BF16 h/M in forward; FP32 dh with BF16 M in backward.
Times are microseconds. All cases use KxV unless marked VxK.

| H / partitions / fallback | h0 serial -> WS | speedup | dht serial -> WS | speedup |
|---|---:|---:|---:|---:|
| 8 / 1 / all | 1.592 -> 1.700 | 0.937x | 1.586 -> 1.700 | 0.933x |
| 8 / 4 / all | 7.731 -> 6.122 | 1.263x | 7.829 -> 6.702 | 1.168x |
| 16 / 16 / all | 62.961 -> 43.591 | 1.444x | 74.639 -> 57.401 | 1.300x |
| 4 / 40 / all | 75.640 -> 55.251 | 1.369x | 77.722 -> 57.146 | 1.360x |
| 64 / 16 / all | 282.171 -> 274.845 | 1.027x | 329.972 -> 334.771 | 0.986x |
| 16 / 16 / none | 23.915 -> 21.820 | 1.096x | 24.751 -> 22.872 | 1.082x |
| 16 / 16 / mixed | 53.596 -> 39.278 | 1.365x | 65.510 -> 50.402 | 1.300x |
| 8 / 16 / all, VxK | 33.257 -> 24.772 | 1.343x | 33.025 -> 24.935 | 1.324x |

Repeating H16/16/all gave h0 65.536 -> 45.595 us (1.437x) and dht
80.242 -> 57.480 us (1.396x). A one-partition sequence has no recurrence to
overlap and pays roughly 0.1 us extra setup. The H64 case has no clear gain.
No shape-specific production gate was introduced to hide these cases.

Original operator benchmark, two runs, medians over three alternating rounds
per run (first `--repeats 50`, repeat `--repeats 100`). Times are milliseconds:

| H / tokens / phase | first serial -> WS | repeat serial -> WS |
|---|---:|---:|
| 16 / 8192 / fwd, AutoCP off | 0.6239 -> 0.6214 | 0.6276 -> 0.6342 |
| 16 / 8192 / bwd, 9 parts | 2.5686 -> 2.5426 | 2.6585 -> 2.4823 |
| 4 / 32768 / fwd, 12 parts | 0.9322 -> 0.9261 | 0.9368 -> 0.9325 |
| 4 / 32768 / bwd, 12 parts | 2.5257 -> 2.3186 | 2.4961 -> 2.4754 |

The approximately 9% first-run backward gain did not reproduce consistently.
Whole-operator measurements remain noisy; the evidence currently supports
scan-kernel improvements for several shapes, not a stable whole-operator
speedup or any specific RTX 5090 gain. The demo prints the individual operator
samples to make this limitation visible. Validate on RTX 5090 before merging.
