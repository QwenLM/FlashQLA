# SM100 fused GDN backward: mask-spill reduction evidence

## Decision

**ACCEPT for the validated scope.** Relative to the 448-column TMEM baseline,
the 480-column candidate passed the selected FP64-reference correctness matrix,
three independent exact-workload runs, all three dynamic spill gates, and all
three paired CUDA-event timing gates. The official paired median ratio is
`0.9052657216` (candidate/baseline), approximately 9.47% lower latency in this
experiment. This is not a claim for other workloads, dtypes, GPUs, or clocks.

## Exact validation scope

- GPU architecture: NVIDIA GB200, compute capability 10.0 / SM100.
- Data: BF16; logical `B=2`, packed `B=1`, packed `T=16384`; varlen sequence
  lengths `[8192,8192]`, `cu_seqlens=[0,8192,16384]`.
- Shape: `H=64`, `Dk=Dv=128`; chunk size 64; `auto_cp=True`.
- Profiled kernel: `tilelang_fused_chunk_gdr_bwd_kernel_kernel`, grid 128,
  block 512.
- Timing seed: 42. Each of six independent trial processes used 10 warmups and
  50 measured CUDA-event samples.

## Design and lifetime proof

The starting point was the previously verified `dq_tmem = u_tmem` alias. For
chunk 64 and head dimension 128, `u_tmem` and `dq_tmem` have disjoint logical
lifetimes, reducing the original physical allocation from 512 to 448 columns.

The spill candidate deliberately spends 32 of those 64 freed columns on an
explicit mask scratch:

```text
mask_tmem_layout = Layout([64, 64],
    lambda i, j: [i + (j // 32) * 64, j % 32])
mask_tmem: mask_tmem_layout
```

This Layout-E mapping is a bijection from the 4,096 logical mask elements to a
`[128,32]` physical layout, so `mask_tmem` consumes 32 columns. Generated CUDA
contains one 32-column `tcgen05` store and three 32-column loads, each with its
matching wait; relevant pipeline barrier waits are also present.

Stage 08 snapshots the complete Q tile from `q_shared` through fragments into
`tmp_shared_2_1`. Stage 09 first scales and persists dQ, then the destructive
`dg += sum(Q*dQ)` operation reads the shared-Q snapshot. That snapshot remains
read-only through the stage 12/14 GEMM consumers. This replaces the generated
register arrays `mask_fragment[32]` and `odot_fragment_2[64]` (96 FP32 values,
384 source-level bytes) without changing the `dq_tmem`/`u_tmem` alias.

## Physical TMEM proof

The fresh generated CUDA has ten balanced allocations/deallocations:

| Width | Allocations | Names | Columns |
|---|---:|---|---:|
| 64 | 5 | `dk`, `dq`, `dv`, `dh_L`, `dh_R` | 320 |
| 32 | 5 | `a`, `dp`, `p`, `mask`, `da` | 160 |
| **Total** | **10** | `5*64 + 5*32` | **480** |

The proof is recorded in `analysis/tmem_allocation_check_v4.txt`,
`analysis/generated_mask_copy_check_v4.txt`, and
`analysis/generated_local_arrays_v4.txt`.

## Provenance, tools, and hashes

| Item | Value |
|---|---|
| Candidate source SHA-256 | `6d0285f5a8bf7ec5092a11b65811b5f4142651bfc4f45979da94b9c9e1d30928` |
| Structural-test SHA-256 | `0f01841c0236ef578ab239e0b1cc59afd26868481c3d6f45680110eef7818fbe` |
| Shared-Q design snapshot SHA-256 | `df84903f354d7460ddba1959f043355016473f57e327d5c4a9154482c8190919` |
| Generated CUDA SHA-256 | `bf823e5016ee29c5d71045d74048cbfcec15e640e0bc4cf3c2036483fe392d61` |
| Generated host-wrapper SO SHA-256 | `5bd4294f99fdf4d2559cc0c9f54e55b9b1b1d25db79025b4778266e9aa5764e7` |
| TMEM448 NCU report SHA-256 | `5a6e16a320e7c4f04d4a2e208f396d6d2ed6ba2ce5bfe70195971d645c7093b8` |
| TMEM480 NCU report SHA-256 | `32a7790ba05fdca3833e5d9f877653bd14ec5f36ceede94452d811b7c08bf278` |
| NCU analyzer SHA-256 | `10b389b9d0da4b9898dfc30166f1c255e872612ce69b424369172b18caca8aa2` |
| Timing runner SHA-256 | `a0afba5b1e9d02f97c9543cc1eec3c38b204e2eb14dbe07cb93d5879f7bc73ce` |
| Timing summarizer SHA-256 | `0244f67206d11cd448399df32c002da4ab33d4dc14a644b25c91d186ba06b0d7` |

Compile, correctness, and formal NCU collection used SLURM job `1483359`, node
`gb200-nvl4-ts2-69`, container `codex-flashqla-tmemspill-1483359`, and GPU UUID
`GPU-035bd673-d588-8449-6557-c80854fa6bd5`. Paired timing used job `1485241`,
node `gb200-nvl4-ts2-105`, container `codex-flashqla-tmemspill-1485241`, GPU
UUID `GPU-9401ba74-5b12-baf4-3196-0e3d789c22b9`, and driver `595.84.01`.

The compile/correctness stack was TileLang 0.1.9, PyTorch
`2.10.0a0+a36e1d39eb.nv26.01.42222806`, Python 3.12.3, and pytest 8.1.1.
Profiling used Nsight Compute `2025.4.1.0` build `37053803` and its Python
report-import API.

## Correctness evidence

Four selected FP64-reference pytest nodes passed:

1. `test_bwd[no_h0-kv-B1-T4096-H4]`
2. `test_bwd[h0-kv-B1-T4096-H4]`
3. `test_bwd[h0-vk-B1-T4096-H4]`
4. `test_bwd[h0-kv-B1-T4096-H64G16-padding]`

They cover absent/present initial state, KV/VK state orientation, and fixed plus
padded/varlen inputs. Result: `4 passed, 14 warnings in 348.97s`, wall 352 s,
RC 0. The warnings were existing `torch.jit.script_method` deprecations.

After that gate passed, three independent exact-workload processes (PIDs
11486, 11716, 11946) completed in 10/9/9 seconds, each with RC 0 and identical
output shapes: three `(1,16384,64,128)` tensors, two `(1,16384,64)` tensors,
and one `(2,64,128,128)` tensor. Full evidence is in
`analysis/task5_correctness_v4.md` and its referenced logs/JSON/RC files.

## Static compile resources

The same nvcc/ptxas flags were used for both embedded-cubin-equivalent rebuilds;
the complete normalized SASS stream of each rebuild matched its embedded cubin.

| Metric | TMEM448 | TMEM480 | Delta |
|---|---:|---:|---:|
| registers/thread | 128 | 128 | 0 |
| static stack frame | 248 B | 144 B | -104 B (-41.9%) |
| ptxas spill stores | 356 B | 176 B | -180 B (-50.6%) |
| ptxas spill loads | 708 B | 456 B | -252 B (-35.6%) |
| static SASS LDL occurrences | 175 | 112 | -63 |
| static SASS STL occurrences | 86 | 40 | -46 |
| cuobjdump LOCAL | 0 B | 0 B | 0 |
| cuobjdump SHARED | 4,096 B | 4,096 B | 0 |
| normalized SASS instructions | 11,744 | 11,816 | +72 |

These are compiler/disassembly facts, not dynamic traffic. The 144 B static
ptxas stack frame is distinct from the 1,024 B NCU launch-stack metric below.

## Formal NCU collection and correction record

There is exactly one valid candidate `.ncu-rep` in this run:
`reports/full_source_import_source_clock_none_varlen_b2x8192_h64_tmem480.ncu-rep`
(32,091,003 bytes, candidate-report SHA-256 above). It was collected with
`--set full`, `SourceCounters`, `PmSampling`, `PmSampling_WarpStates`,
`--import-source yes`, `--clock-control none`, the exact kernel regex, and
`-c 1`. It contains one range, one action, the expected kernel/grid/block, and
2,594 exported metrics; collection completed 48 passes.

Attempt 1 is preserved as a correction record. It ran NCU around a shell
wrapper with `--target-processes application-only`; the wrapper launched the
Python child, so NCU returned RC 0 but warned `No kernels were profiled` and
created no report. Attempt 2 targeted Python directly and produced the sole
formal report. No third NCU collection occurred. The analysis phase only
imported the two existing reports on CPU; it did not launch another kernel.

## Dynamic NCU evidence

All three required gates independently require `delta_pct <= -10%`:

| Gate | TMEM448 | TMEM480 | Delta | Result |
|---|---:|---:|---:|---|
| local LD instructions | 7,316,992 | 3,188,224 | -56.4271% | PASS |
| local ST instructions | 5,012,352 | 2,062,720 | -58.8473% | PASS |
| local spilling requests | 12,307,072 | 5,228,672 | -57.5149% | PASS |

All 13 report spill metrics are retained, including zero/unchanged values:

| Metric | TMEM448 | TMEM480 | Delta |
|---|---:|---:|---:|
| `derived__local_spilling_requests` | 12,307,072 | 5,228,672 | -57.5149% |
| `derived__local_spilling_requests_pct` | 100 | 100 | 0% |
| `derived__shared_spilling_requests` | 2,944 | 2,944 | 0% |
| `derived__shared_spilling_requests_pct` | 100 | 100 | 0% |
| `sass__inst_executed_register_spilling` | 12,329,344 | 5,250,944 | -57.4110% |
| `sass__inst_executed_register_spilling_mem_local` | 12,307,072 | 5,228,672 | -57.5149% |
| `sass__inst_executed_register_spilling_mem_local_op_read` | 7,294,720 | 3,165,952 | -56.5994% |
| `sass__inst_executed_register_spilling_mem_local_op_write` | 5,012,352 | 2,062,720 | -58.8473% |
| `sass__inst_executed_register_spilling_mem_shared` | 2,944 | 2,944 | 0% |
| `sass__inst_executed_register_spilling_mem_shared_op_read` | 2,944 | 2,944 | 0% |
| `sass__inst_executed_register_spilling_mem_shared_op_write` | 0 | 0 | unavailable (zero baseline) |
| `sass__inst_executed_register_spilling_op_read` | 7,297,664 | 3,168,896 | -56.5766% |
| `sass__inst_executed_register_spilling_op_write` | 5,012,352 | 2,062,720 | -58.8473% |

The reduction is in local-memory spill traffic; shared spilling stayed at
2,944 requests. Supporting measurements show the tradeoff:

| Metric | TMEM448 | TMEM480 | Delta |
|---|---:|---:|---:|
| registers/thread | 128 | 128 | 0% |
| NCU launch stack | 1,024 | 1,024 | 0% |
| waves/SM | 0.842105 | 0.842105 | 0% |
| NCU kernel duration | 1,465,920 ns | 1,354,432 ns | -7.6053% |
| SM throughput | 21.3951% | 23.6078% | +10.3421% |
| memory throughput | 31.7714% | 34.2322% | +7.7453% |
| long-scoreboard ratio | 12.3394 | 10.1255 | -17.9413% |
| short-scoreboard ratio | 1.01994 | 1.33142 | +30.5393% |
| shared-load instructions | 19,841,536 | 20,628,480 | +3.9661% |
| shared-store instructions | 9,895,040 | 9,895,168 | +0.0013% |
| all shared bank conflicts | 23,084,641 | 26,063,357 | +12.9035% |
| shared-load bank conflicts | 21,243,353 | 23,193,702 | +9.1810% |
| shared-store bank conflicts | 1,801,670 | 2,883,101 | +60.0238% |
| derived shared conflict N-way | 1,916 | 1,961 | +2.3486% |

Per-PC report import found 11,743 -> 11,815 executed PCs overall, 175 -> 112
LDL PCs, 86 -> 40 STL PCs, and 261 -> 152 source-mapped local PCs. Top
candidate examples all executed 65,536 times: an LDL at `tvm_kernels.cu:556`
had 27 samples (26 no-instructions, 1 long scoreboard); an LDL at line 634 had
22 (19 short scoreboard, 3 wait); an STL at `cuda_bf16.hpp:637` had 55 barrier
samples; an STL at `tvm_kernels.cu:801` had 19 (18 no-instructions, 1 selected).
Addresses are cubin-specific; exact SASS, paths, top-25 lists, and all nonzero
stall reasons are in `analysis/local_spill_hotspots.txt`.

## Paired CUDA-event timing

All warm/trial/overall return codes were zero. The prescribed order was B/C,
C/B, B/C, using distinct baseline/candidate HOME, TileLang cache, and extension
directories. Six unique trial PIDs produced 300 finite positive samples.

| Pair (order) | B p10 / median / p90 ms | C p10 / median / p90 ms | C/B median ratio | Gate `<=1.01` |
|---|---|---|---:|---|
| 1 (B,C) | 1.807533 / 1.822624 / 1.839216 | 1.623446 / 1.655504 / 1.701942 | 0.9083080313 | PASS |
| 2 (C,B) | 1.816982 / 1.831312 / 1.844694 | 1.633139 / 1.657824 / 1.704496 | 0.9052657216 | PASS |
| 3 (B,C) | 1.805341 / 1.826640 / 1.854384 | 1.604874 / 1.637216 / 1.683229 | 0.8962991958 | PASS |
| **Median ratio** | | | **0.9052657216** | **PASS** |

There is a measurable first-sample ramp: all 6/6 first samples are each
process's maximum, and the project linear-percentile Tukey check flags 5/6.
First-sample overhead versus each process median is B1 +3.482%, C1 +6.041%, B2
+4.560%, C2 +8.198%, B3 +3.005%, C3 +5.647%. Dropping the first sample gives
pair ratios `[0.9071879138, 0.9051746582, 0.8958270861]`, median
`0.9051746582`: still PASS and only -0.009106 percentage point from the
official ratio. Candidate per-process CV was about 1.86-2.07%, versus baseline
0.89-1.08%; this variability did not create a threshold-sensitive decision.

The paired CUDA-event wrapper is the performance acceptance measurement. The
single profiled NCU kernel duration is a diagnostic under profiler replay and
is not treated as the same timing population.

## Final verification

The completed verification was split at the host/container boundary. These
checks predate this documentation-only amendment; no GPU workload or NCU
collection was rerun while adding this section.

### Successful boundaries

- Container attempt 2 returned RC 0. It passed all 9 structural tests, all 4
  analyzer/provenance unit tests, and all 11 paired-timing tool tests. The same
  chained command also completed `py_compile` and `bash -n`; those checks are
  silent on success. Primary evidence:
  `analysis/final_verification_container_job1485241_attempt2.{log,rc}`.
- Host attempt 3 returned RC 0. It completed `git diff --check`, verified that
  the candidate run contains exactly one `.ncu-rep`, found no hidden staging or
  temporary verification file, and recorded SHA/status evidence. Primary
  evidence: `analysis/final_verification_host_job1485241_attempt3.{log,rc}`.
  Its point-in-time hashes were source `6d0285f5...d30928`, structural test
  `0f01841c...fbe`, candidate report `32a7790b...f278`, timing summary
  `810e0d7f...c74`, pre-amendment REPORT `14196039...478`, and memory
  `a75b8e1e...478`. The REPORT hash is expected to change when this verification
  record is appended; the immutable source/test/report hashes remain unchanged.
- The host status snapshot contained only unstaged/untracked worktree entries;
  nothing was staged or committed. No push, merge, or cleanup was performed.

### Preserved correction ledger

- The first combined container command returned RC 128 only after the 9, 4,
  and 11 test groups had passed: the container could not resolve the host-only
  `.git/worktrees/sm100-dq-tmem-reuse` path. This is a Git-visibility failure,
  not a test failure. Evidence:
  `analysis/final_verification_job1485241.{log,rc}`.
- Host attempt 1 returned RC 4 while importing `tests/conftest.py` because the
  host Python lacked `torch`; collection never began. The container test run
  above provides the dependency-complete test result. Evidence:
  `analysis/final_verification_host_job1485241.{log,rc}`.
- Host attempt 2 returned RC 1 because its report-path input misspelled the
  actual filename, using hyphens before `h64`/`tmem480` instead of underscores.
  It had already printed matching source/test hashes; host attempt 3 used the
  real path and passed. Evidence:
  `analysis/final_verification_host_job1485241_attempt2.{log,rc}`.

These correction facts do not overturn the successful structural, analyzer,
timing-tool, hash, report-count, diff, or status checks. They also do not add
new numerical/kernel evidence beyond the successful runs documented above;
all original logs and return-code files remain preserved.

## Facts, inference, and acceptance rationale

**Measured facts:** selected correctness passed; generated physical TMEM is
480 columns; static spill footprint and LDL/STL sites decreased; all three
dynamic local-spill gates decreased by 56-59%; every timing pair and the median
ratio passed; shared bank conflicts and short-scoreboard ratio increased.

**Inference:** eliminating the two generated register arrays and shortening
their live ranges is the most direct explanation for the lower local spill
traffic. The stage-08 shared-Q snapshot is also the likely source of the added
shared traffic/conflicts. These causal statements are supported by source,
generated code, static resources, and NCU correlation, but are not isolated
single-variable hardware proofs.

**Acceptance:** correctness plus the explicit spill and timing gates outweigh
the observed shared-memory cost for this validated workload. The candidate is
accepted within the scope stated above; the 448-column result remains the
historical baseline rather than being erased.

## Remaining risks

- Shared bank conflicts rose 12.90% overall (shared-store conflicts +60.02%),
  and the short-scoreboard ratio rose 30.54%. Shared-Q layout/banking remains
  the clearest follow-up optimization target.
- `--clock-control none` means clocks were not fixed. Same-node alternating
  pairs and order reversal reduce, but do not eliminate, drift risk.
- Validation covers one BF16 varlen workload on GB200/SM100. Other shapes,
  dtypes, fixed-length paths, GPU SKUs, and architectures are unproven.
- First-sample ramp and higher candidate CV are real, although drop-first
  sensitivity does not change the gate.
- CUDA-event wrapper timing and NCU replay duration answer different questions;
  they must not be compared as interchangeable latency estimates.

## Evidence index

- Design/TDD: `analysis/task3_tdd_evidence.md`.
- Compile/resources: `analysis/task4_compile_resources_v4.md` and
  `analysis/resources_v4/`.
- Correctness: `analysis/task5_correctness_v4.md`.
- NCU collection/correction: `analysis/task6_collection_v4.md`.
- Dynamic analysis: `analysis/task6_analysis_v4.md`,
  `analysis/compare_tmem448_vs_tmem480.json`, and
  `analysis/local_spill_hotspots.txt`.
- Timing protocol/results: `analysis/timing_preflight_job1485241.txt`, six
  `analysis/timing_{baseline,candidate}_pair{1,2,3}_v4.{json,log}`, and
  `analysis/timing_summary.json`.
