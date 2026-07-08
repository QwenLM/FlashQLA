# SM100 TMEM-Assisted Spill Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the gate mask in a 32-column TMEM allocation and use the existing shared Q snapshot for the stage-09 Q·dQ reduction, removing two large Consumer-A fragments while preserving the spill and latency gates.

**Architecture:** Keep `dq_tmem = u_tmem` and `mask_tmem[64,64]` (480 physical TMEM columns). Bind the non-MMA mask to explicit TCGEN05 Layout-E `[i + (j // 32) * 64, j % 32]`, whose output shape is `[128,32]`. `mask_tmem` is used only in stages 00–07. Stage 08 writes both Q halves immediately through `a_fragment`/`p_fragment` into `tmp_shared_2_1`; stage 09 persists scaled dQ, performs one full `dq_fragment *= tmp_shared_2_1` loop, and reduces it without changing barriers, stage-10 output, or GEMM signatures.

**Tech Stack:** Python, TileLang 0.1.9, PyTorch 2.10, pytest, CUDA SM100/GB200, Nsight Compute 2025.4.1.

---

### Task 1: Add the failing shared-Q structural regression

**Files:**
- Modify: `tests/test_sm100_tmem_reuse.py`
- Read: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py`

- [x] **Step 1: Encode the shared-Q invariants**

The AST tests require:

- each Q half to pass through `a_fragment` or `p_fragment` and be written
  immediately to its `tmp_shared_2_1` half before `bar_09`;
- exactly one `a_fragment -> mask_tmem` store and one
  `mask_tmem -> a_fragment` load, for stages 00 and 07 respectively;
- no dQ-half-to-`dp_fragment` adapters or half-fragment reductions;
- `dq_fragment -> dq_tmem` before one full `[64,128]` shared-Q multiply,
  followed by one full reduction before `bar_10`;
- `tmp_shared_2_1` to be RHS-only in the dot interval; and
- unchanged stage-12 and both stage-14 Q-consuming GEMM signatures.

- [x] **Step 2: Verify RED before touching the kernel**

```bash
python3 -m pytest tests/test_sm100_tmem_reuse.py -q --noconftest
```

Actual RED on old-adapter SHA `4f652d29245f7eda77323822e16224d10e6db84ab550a9de9a8678b389ff4891`:
five tests collected, three passed, and two failed because stage 08 still
stored Q into `mask_tmem` and stage 09 lacked the full shared-Q multiply.

### Task 2: Freeze the 448-column baseline and prepare the new run

**Files:**
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/reports/`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/`
- Reuse: `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/harness/run_fused_bwd.py`

- [ ] **Step 1: Create a fresh, self-contained profile directory**

```bash
mkdir -p profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/{harness,reports,analysis}
cp profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/harness/run_fused_bwd.py \
  profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/
cp profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/analysis/{ncu_utils.py,validate_report.py} \
  profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/
```

- [ ] **Step 2: Start one persistent GB200 session**

Copy the validated session launcher, change `RUN_NAME` to the new run name,
change the container prefix to `codex-flashqla-tmemspill`, and retain:

```text
cluster=dlcluster
partition=gb200nvl4
account/qos=dlfw-inference-priority
gres=gpu:1
time=04:00:00
image=nvcr.io/nvidia/pytorch:26.01-py3
```

Install TileLang 0.1.9 and its declared runtime dependencies into a fresh
`/tmp/flashqla_site_spill`. Verify `python3 -c 'import tilelang, torch'`
before running any GPU test.

- [ ] **Step 3: Snapshot the exact baseline Python package inside the container**

Before editing production source:

```bash
mkdir -p /tmp/flashqla_tmem448_baseline
cp -a flash_qla /tmp/flashqla_tmem448_baseline/
```

Record the source SHA-256 of both the live and snapshot
`fused_bwd.py`; they must match at this point.

- [ ] **Step 4: Add CUDA-event timing to the run-local harness**

Extend `run_fused_bwd.py` with `--warmup`, `--repeats`, and
`--timing-json`. After the forward preparation, call the same backward
wrapper for each warmup, then record each timed call using CUDA events:

```python
times_ms = []
for _ in range(args.warmup):
    outputs = chunk_gated_delta_rule_bwd(
        q=q, k=k, v=v, g=g_cumsum, beta=beta, A=A, do=do, dht=dht,
        scale=scale, initial_state=h0, cu_seqlens=cu_seqlens,
        auto_cp=auto_cp, cp_cache=cp_cache,
    )
torch.cuda.synchronize()

for _ in range(args.repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = chunk_gated_delta_rule_bwd(
        q=q, k=k, v=v, g=g_cumsum, beta=beta, A=A, do=do, dht=dht,
        scale=scale, initial_state=h0, cu_seqlens=cu_seqlens,
        auto_cp=auto_cp, cp_cache=cp_cache,
    )
    end.record()
    end.synchronize()
    times_ms.append(start.elapsed_time(end))

if args.timing_json is not None:
    args.timing_json.write_text(
        json.dumps({"times_ms": times_ms, "metadata": metadata}, indent=2) + "\n"
    )
```

Run one untimed exact-workload smoke using the baseline snapshot to prove the
snapshot imports and compiles before production edits.

### Task 3: Implement the 480-column Consumer-A data flow

**Files:**
- Modify: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py:155-199`
- Modify: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py:584-699`
- Test: `tests/test_sm100_tmem_reuse.py`

- [x] **Step 1: Remove the two dedicated fragments and add TMEM**

Use:

```python
# CONSUMER_A
p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
dp_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
da_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
u_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
dq_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
db_fragment = T.alloc_fragment((block_S), dtype=accum_dtype)
dg_fragment_2 = T.alloc_fragment((block_S), dtype=accum_dtype)
```

After `dp_tmem`, allocate:

```python
mask_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
```

Keep `dq_tmem = u_tmem` unchanged.

- [x] **Step 2: Persist and reload the stage-00–07 mask**

Stage 00:

```python
for j_s, j_t in T.Parallel(block_S, block_S):
    a_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
for j_s, j_t in T.Parallel(block_S, block_S):
    if j_s >= j_t:
        a_fragment[j_s, j_t] = T.exp2(a_fragment[j_s, j_t] * 1.442695)
    else:
        a_fragment[j_s, j_t] = 0
T.copy(a_fragment, mask_tmem)
T.barrier_wait(tcbar_00, (i_s + 0) % 2)
T.copy(p_tmem, p_fragment)
for j_s, j_t in T.Parallel(block_S, block_S):
    p_fragment[j_s, j_t] *= a_fragment[j_s, j_t]
```

Before each stage-02 and stage-06 mask multiplication, load
`T.copy(mask_tmem, p_fragment)` and multiply by `p_fragment`. At stage 07,
load `T.copy(mask_tmem, a_fragment)`, multiply `dp_fragment` by
`a_fragment`, then load P into `p_fragment` as before.

- [x] **Step 3: Snapshot Q in shared memory and consume it in one full dot**

Stage 08 uses only transient half fragments:

```python
T.copy(q_shared[:, : DK // 2], a_fragment)
T.copy(a_fragment, tmp_shared_2_1[:, : DK // 2])
T.copy(q_shared[:, DK // 2 :], p_fragment)
T.copy(p_fragment, tmp_shared_2_1[:, DK // 2 :])
```

After scaling dQ and storing it back to `dq_tmem`, stage 09 uses:

```python
for j_s, j_k in T.Parallel(block_S, DK):
    dq_fragment[j_s, j_k] *= tmp_shared_2_1[j_s, j_k]
T.reduce_sum(dq_fragment, dg_fragment_2, dim=1, clear=False)
```

Keep `bar_08`, `bar_09`, `tcbar_08`, `bar_10`, stage 10, layouts,
and all GEMM signatures unchanged. The snapshot remains read-only for stage 12
and stage 14.

v1 failed because 64-column `a_fragment/p_fragment` ownership conflicted with
the 128-column `dq_fragment` layout. v2 failed because a slice-to-`dp_fragment`
copy retained the underlying 128-column source ownership. The full
fragment-times-shared loop avoids both fragment-to-fragment constraints.

The extra `64 * 128 * 2 = 16 KiB` shared read per iteration is a performance
risk inferred from source traffic; it remains unmeasured until GPU profiling.

- [x] **Step 4: Verify host and container GREEN**

```bash
python3 -m pytest tests/test_sm100_tmem_reuse.py -q --noconftest
```

Actual: host Python 3.6.8 and container Python 3.12.3 each passed all five
tests; both files also passed container `py_compile`.

- [x] **Step 5: Add the explicit non-MMA mask TMEM Layout-E via TDD**

The v3 exact-workload compile reached `LowerTmemCopy` and failed because
`mask_tmem` had no layout. Add a host-only AST test first that requires one
factory-scope ordinary `tilelang.layout.Layout`, exact mapping
`[i + (j // 32) * 64, j % 32]`, and a `mask_tmem: mask_tmem_layout` entry in
the existing sole annotation map. Also enumerate all 4,096 logical points and
prove a bijection onto 128 rows by 32 columns.

Actual RED on v3 SHA
`df84903f354d7460ddba1959f043355016473f57e327d5c4a9154482c8190919`:
`1 failed, 8 passed`; the first failure was the missing factory assignment,
and source inspection also confirmed the annotation entry was absent.

Minimal production change after `block_S`:

```python
mask_tmem_layout = tilelang.layout.Layout(
    [block_S, block_S],
    lambda i, j: [i + (j // 32) * 64, j % 32],
)
```

Bind it in the existing annotation map. Do not add the Consumer-A warp-group
absolute `+256` offset: `LowerTmemCopy` uses the current `[256,384)` thread
bounds separately. Do not change data flow, barriers, GEMMs, nreg values, or
allocations.

Actual GREEN on v4 SHA
`6d0285f5a8bf7ec5092a11b65811b5f4142651bfc4f45979da94b9c9e1d30928`:
host Python 3.6.8 and container Python 3.12.3 each passed all nine structural
tests; both passed `py_compile`, and host `git diff --check` passed. AST/hash
comparison reconstructed the exact v3 SHA and found identical data-flow,
barrier, GEMM, and allocation call sequences. The sole layout call remains;
only its map grows from 13 to 14 entries.

### Task 4: Compile and inspect physical resources

**Files:**
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/generated_fused_bwd_sm100_tmem480_v4.cu`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/generated_fused_bwd_sm100_tmem480_v4.so`

- [x] **Step 1: Compile with a fresh TileLang cache on GB200**

Run the exact workload once with:

```text
BF16, B=2 logical sequences, packed B=1, total T=16384,
Q/K/V heads=64, D=128, cu_seqlens=[0,8192,16384],
chunk_size=64, auto_cp=True
```

Copy the generated CUDA and shared object into the new run directory.

Completed with rc=0 and the expected six output shapes. The complete
timestamped pre-run directory listing was not saved, so preflight absence is
not retroactively independently auditable.

- [x] **Step 2: Verify generated declarations and TMEM capacity**

```bash
grep -n 'tmem_allocate\|tmem_deallocate\|mask_fragment\|odot_fragment_2' \
  profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/generated_fused_bwd_sm100_tmem480_v4.cu
```

Expected:

- ten allocations and ten deallocations;
- six 64-column plus four 32-column allocations would be 512 and is wrong;
- the required total is five 64-column plus five 32-column allocations:
  `5*64 + 5*32 = 480`;
- no C++ arrays named `mask_fragment` or `odot_fragment_2`.

TileLang 0.1.9 source shows `LowerTileOp::makeBufferWithLayout` remapping the
mask to `Layout::OutputShape() == [128,32]` before `LowerSharedTmem` rounds the
second dimension to 32 columns. Treat 480 as expected, not verified, until the
generated v4 CUDA is inspected.

Actual v4 generated CUDA verifies ten balanced allocations/deallocations,
five 64-column plus five 32-column allocations (480 columns), and no
mask_fragment or odot_fragment_2 arrays.

- [x] **Step 3: Inspect compiler resource usage**

Capture `cuobjdump --dump-resource-usage` and generated array declarations.
Record registers, stack frame, spill load/store bytes if exposed, and compare
them with the 448-column generated module. Do not infer NCU instruction-count
improvement from stack size alone.

Completed static audit: REG 128 -> 128, STACK 248 B -> 144 B, ptxas static
spill stores 356 B -> 176 B, and static spill loads 708 B -> 456 B. Dynamic
traffic/performance remains pending NCU. Evidence: analysis/resources_v4/.

### Task 5: Verify numerical correctness and synchronization

**Files:**
- Test: `tests/test_gdr_unit.py`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/task5_correctness_v4.md`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/correctness_v4_*`

- [x] **Step 1: Run selected FP64-reference cases**

```bash
python3 -m pytest \
  'tests/test_gdr_unit.py::test_bwd[no_h0-kv-B1-T4096-H4]' \
  'tests/test_gdr_unit.py::test_bwd[h0-kv-B1-T4096-H4]' \
  'tests/test_gdr_unit.py::test_bwd[h0-vk-B1-T4096-H4]' \
  'tests/test_gdr_unit.py::test_bwd[h0-kv-B1-T4096-H64G16-padding]' \
  -q --maxfail=1
```

Actual: one pytest process with an overall 30-minute timeout collected exactly
the four explicit nodes and reported `4 passed, 14 warnings in 348.97s`;
process wall duration was 352 seconds and return code was zero. This is
equivalent to the planned coverage: no-h0/h0, both KV/VK state orientations,
and fixed plus padded/varlen paths.

- [x] **Step 2: Repeat the exact workload three times**

Only after Step 1 passed, three independent Python processes ran:

```bash
bash profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/run_variant.sh \
  candidate --varlen --num-seqs 2 --seqlen 16384 --heads 64 \
  --cu-seqlens 0,8192,16384 \
  --metadata-out profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/correctness_v4_exact_N.json
```

Actual PIDs were 11486, 11716, and 11946; wall durations were 10, 9, and 9
seconds. Every process returned zero and produced exactly one identical DONE
record:

```text
(1,16384,64,128) x3, (1,16384,64) x2, (2,64,128,128) x1
```

Complete commands, logs, return codes, and workload metadata are in the
`correctness_v4_*` artifacts and summarized in `task5_correctness_v4.md`.

### Task 6: Collect and attribute the candidate NCU report

**Files:**
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/reports/full_source_import_source_clock_none_varlen_b2x8192_h64_tmem480.ncu-rep`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/compare_tmem448_vs_tmem480.json`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/local_spill_hotspots.txt`

- [x] **Step 1: Collect one full/source/import-source report**

```bash
ncu --set full \
  --section SourceCounters \
  --section PmSampling \
  --section PmSampling_WarpStates \
  --import-source yes \
  --source-folders /tmp/flashqla_home_spill/.tilelang/cache \
  --clock-control none \
  --target-processes application-only \
  -k regex:tilelang_fused_chunk_gdr_bwd_kernel_kernel \
  -c 1 --force-overwrite \
  -o reports/full_source_import_source_clock_none_varlen_b2x8192_h64_tmem480 \
  python3 harness/run_fused_bwd.py --varlen --num-seqs 2 \
    --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384
```

Actual collection was staged. Attempt1 preserved its `No kernels were profiled`
result and no report. The corrected direct-Python attempt2 command in
`analysis/ncu_collection_tmem480_v4_attempt2_command.txt` returned zero after
48 passes and produced exactly one validated report with SHA-256
`32a7790ba05fdca3833e5d9f877653bd14ec5f36ceede94452d811b7c08bf278`.

- [x] **Step 2: Parse both reports with the NCU Python API**

Compare the existing 448-column baseline report with the new report for:

```text
smsp__sass_inst_executed_op_local_ld.sum
smsp__sass_inst_executed_op_local_st.sum
derived__local_spilling_requests
sass__inst_executed_register_spilling_mem_local_op_read
sass__inst_executed_register_spilling_mem_local_op_write
launch__registers_per_thread
launch__stack_size
launch__waves_per_multiprocessor
gpu__time_duration.sum
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
```

Enumerate per-PC metrics with correlation IDs and use `action.sass_by_pc()`
plus `action.source_info(pc)` to rank executed `LDL` and `STL` locations.
Write the exact SASS, source file/line, execution count, and warp-stall samples
for the top locations.

Actual formal CPU-only parse used Nsight Compute 2025.4.1's
`extras/python/ncu_report.py`; analyzer SHA-256 was
`10b389b9d0da4b9898dfc30166f1c255e872612ce69b424369172b18caca8aa2`.
Both reports yielded 2,594 metrics. Executed local PCs changed from LDL/STL
`175/86` to `112/40`, with full SASS/source/stall attribution written to
`analysis/local_spill_hotspots.txt`.

- [x] **Step 3: Apply the spill gate**

Candidate must reduce each of local LD instructions, local ST instructions,
and local spilling requests by at least 10%. If not, mark it rejected without
claiming the source-level fragment reduction fixed spilling.

Actual gate result: local LD `-56.4271%`, local ST `-58.8473%`, and
local spilling requests `-57.5149%`; all three independently passed the
`delta_pct <= -10.0%` requirement. Shared bank-conflict and short-scoreboard
regressions remain Task7 follow-up signals, not Task6 spill-gate failures.

### Task 7: Run paired same-node performance trials

**Files:**
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/timing_baseline_*.json`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/timing_candidate_*.json`
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/timing_summary.json`

- [x] **Step 1: Warm both compiled variants**

Use distinct `HOME` and TileLang cache directories for:

```text
baseline PYTHONPATH=/tmp/flashqla_tmem448_baseline
candidate PYTHONPATH=/workspace/AI_tests/.codex-worktrees/FlashQLA/sm100-dq-tmem-reuse
```

Run one warmup process for each before collecting paired trials.

Actual result: the prior job `1483359` reached its time limit before formal
Task7 timing. Recovery used new job `1485241` on `gb200-nvl4-ts2-105`, container
`codex-flashqla-tmemspill-1485241`, and GPU
`GPU-9401ba74-5b12-baf4-3196-0e3d789c22b9`. Baseline and candidate warm
processes both returned zero with isolated HOME/cache/torch-extension paths and
validated source hashes.

- [x] **Step 2: Collect at least three alternating A/B pairs**

Use order `baseline,candidate`, then `candidate,baseline`, then
`baseline,candidate`. Every process runs ten warmups and fifty timed
backward calls on the exact requested workload. Record GPU UUID, package hash,
trial order, all 50 CUDA-event samples, median, p10, and p90.

Actual result: order was exactly `B,C / C,B / B,C`; six unique processes each
ran ten warmups and retained fifty samples, for 300 samples total. All six
trial RCs were zero and every JSON recorded the same session/container/GPU plus
variant-specific import/cache provenance. Per-pair distributions and raw paths
are recorded in `analysis/task7_paired_timing_v4.md`.

- [x] **Step 3: Apply the performance gate**

Compute the median of the three paired candidate/baseline median ratios.
Accept only when the ratio is at most 1.01. Treat the one-sample NCU duration
as diagnostic only; the paired CUDA-event result decides the performance gate.

Actual result: paired ratios were `0.9083080313`, `0.9052657216`, and
`0.8962991958`; their median `0.9052657216` passed the `<= 1.01` gate and the
runner returned zero. A read-only sensitivity check removed each process's
first timed sample and produced median ratio `0.9051747`, so the observed
first-sample ramp Minor did not change the decision. Single-run NCU duration
remains diagnostic only; this result is not generalized to arbitrary GPUs,
nodes, or clock-lock regimes.

### Task 8: Decide, document, and verify

**Files:**
- Create: `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/REPORT.md`
- Update: `AGENT_MEMORY.md`
- Review: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py`
- Review: `tests/test_sm100_tmem_reuse.py`

- [x] **Step 1: Accept or reject from measured gates**

Accept only if correctness passes, all three spill metrics improve by at least
10%, and paired median latency is within 1%. If rejected, use `apply_patch`
to restore only the Task-3 kernel changes while preserving the accepted
`dq_tmem = u_tmem` alias, tests/evidence, and the rejected-experiment report.

Actual result: **ACCEPT** for the stated GB200/SM100 BF16 varlen workload. The
selected correctness matrix and three exact-workload runs passed; local LD,
local ST, and local spilling requests improved 56.43%, 58.85%, and 57.51%; the
three paired timing ratios all passed and their median was `0.9052657216`.

- [x] **Step 2: Write the report**

Record workload, GPU/container/tool versions, source hashes, 448/480 TMEM
proof, correctness commands, raw spill metrics, top LDL/STL sites, paired
timing distributions, gate decision, and remaining risks. Clearly separate
facts from interpretations.

Actual result: the self-contained run-root `REPORT.md` was created and root
`AGENT_MEMORY.md` was updated. The report records the decision, exact scope,
raw/static/dynamic/timing evidence, facts versus inference, correction ledger,
and remaining risks; memory preserves the historical 448-column result while
adding the verified 480-column follow-up.

- [x] **Step 3: Run fresh final verification**

```bash
python3 -m pytest tests/test_sm100_tmem_reuse.py -q
git diff --check
git status --short
```

Re-read the design requirements line by line. Preserve all pre-existing
untracked profile directories; do not stage, commit, push, merge, or clean
without explicit user authorization. Release the GB200 allocation after all
artifacts are written.

Actual result: final verification completed across the dependency-complete
container and Git-visible host. Container attempt 2 returned RC 0 after 9
structural, 4 analyzer, and 11 timing-tool tests plus silent-success
`py_compile`/`bash -n`; host attempt 3 returned RC 0 for diff, unique-report,
no-hidden-staging/tmp, SHA, and status checks. Earlier RC 128/4/1 attempts were
respectively host-`.git` invisibility in the container, missing host `torch`,
and a misspelled report-path input; the original logs/RCs are preserved and do
not contradict the successful checks. No file was staged or committed, and no
push, merge, or cleanup was performed.
