# SM100 `dq_tmem` Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse `u_tmem` for `dq_tmem` in the SM100 fused GDN backward kernel, reducing compiled TMEM allocation from 512 to 448 columns without changing results.

**Architecture:** Keep the current 16-stage pipeline and all barriers unchanged. Bind the logical `dq_tmem` handle to the existing `[64, 128]` FP32 `u_tmem` allocation after `u_tmem` becomes dead; stage 08 clears the aliased allocation before producing `dQ`.

**Tech Stack:** Python, TileLang 0.1.9, PyTorch, pytest, CUDA SM100/GB200, Nsight Compute 2025.4.1.

---

### Task 1: Add the structural TMEM reuse regression test

**Files:**
- Create: `tests/test_sm100_tmem_reuse.py`
- Read: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py`

- [ ] **Step 1: Write the failing test**

```python
import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py"
)


def test_dq_tmem_reuses_u_tmem_allocation():
    tree = ast.parse(SOURCE.read_text())
    assignments = {}
    tmem_allocations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assignments[target.id] = node.value
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "T"
            and node.value.func.attr == "alloc_tmem"
        ):
            tmem_allocations.append(target.id)

    assert ast.unparse(assignments["dq_tmem"]) == "u_tmem"
    assert len(tmem_allocations) == 9
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_sm100_tmem_reuse.py -q
```

Expected: one assertion failure showing that `dq_tmem` is still a call to
`T.alloc_tmem`, proving the test detects the current 512-column design.

### Task 2: Implement the minimal buffer alias

**Files:**
- Modify: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py:188-190`
- Test: `tests/test_sm100_tmem_reuse.py`

- [ ] **Step 1: Replace the independent allocation**

```python
u_tmem = T.alloc_tmem((block_S, DK), dtype=accum_dtype)
dq_tmem = u_tmem
dk_tmem = T.alloc_tmem((block_S, DK), dtype=accum_dtype)
```

Do not change the existing stage barriers, GEMM `clear_accum` flags, or
fragment layouts.

- [ ] **Step 2: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_sm100_tmem_reuse.py -q
```

Expected: `1 passed`.

- [ ] **Step 3: Inspect the focused diff**

Run:

```bash
git diff -- tests/test_sm100_tmem_reuse.py flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py
```

Expected: one test file plus the one-line allocation-to-alias change. Do not
commit; the project instructions require explicit user authorization.

### Task 3: Compile and verify generated TMEM allocation

**Files:**
- Create: `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/harness/`
- Create: `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/analysis/`
- Reuse: `profile/sm100-fused-gdn-bwd-varlen-b2x8192-h64-import-source-20260707/harness/run_fused_bwd.py`

- [ ] **Step 1: Create a fresh run directory and reuse the validated harness**

```bash
mkdir -p profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/{harness,reports,analysis}
cp profile/sm100-fused-gdn-bwd-varlen-b2x8192-h64-import-source-20260707/harness/run_fused_bwd.py profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/harness/
```

- [ ] **Step 2: Compile once on GB200 and capture generated CUDA**

Run the harness inside the GB200 container with a fresh TileLang cache, then
copy the generated `device_kernel.cu` into the new run's `harness/` directory
as `generated_fused_bwd_sm100_tmem448.cu`.

- [ ] **Step 3: Verify the physical allocation**

Run:

```bash
rg -n "tmem_allocate|tmem_deallocate|dq_tmem|u_tmem" profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/harness/generated_fused_bwd_sm100_tmem448.cu
```

Expected: nine allocation calls totaling `4*32 + 5*64 = 448` columns; no
separate `dq_tmem` shared address; stage-08 `dQ` operations target the same
address used earlier by `u_tmem`.

### Task 4: Verify numerical correctness and synchronization

**Files:**
- Test: `tests/test_gdr_unit.py`
- Reuse: `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/harness/run_fused_bwd.py`

- [ ] **Step 1: Run targeted reference comparisons**

Inside the GB200 environment, run:

```bash
python3 -m pytest tests/test_gdr_unit.py -q -k "test_bwd and (B1-T4096-H4 or B1-T4096-H64G16-padding)" --maxfail=1
```

Expected: all selected `state_v_first`, `use_h0`, fixed, varlen, and padded
backward cases pass their existing FP64-reference tolerance.

- [ ] **Step 2: Run the exact requested workload**

```bash
python3 profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/harness/run_fused_bwd.py --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --metadata-out profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/analysis/workload.json
```

Expected: kernel completes without hang or CUDA error and reports the six
backward output shapes for the packed BF16 workload.

- [ ] **Step 3: Repeat the exact workload**

Run the same exact-workload command at least three times with a warm cache.
Expected: all repetitions finish without a race, illegal access, or divergent
output-shape metadata.

### Task 5: Collect and compare one NCU report

**Files:**
- Create: `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/reports/full_source_import_source_clock_none_varlen_b2x8192_h64_tmem448.ncu-rep`
- Create: `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/REPORT.md`
- Compare: `profile/sm100-fused-gdn-bwd-varlen-b2x8192-h64-import-source-20260707/reports/full_source_import_source_clock_none_varlen_b2x8192_h64.ncu-rep`

- [ ] **Step 1: Collect the candidate report**

Use exactly:

```bash
ncu --set full --section SourceCounters --section PmSampling --section PmSampling_WarpStates --import-source yes --source-folders /tmp/flashqla_home/.tilelang/cache --clock-control none --target-processes application-only -k regex:tilelang_fused_chunk_gdr_bwd_kernel_kernel -c 1 --force-overwrite -o full_source_import_source_clock_none_varlen_b2x8192_h64_tmem448 python3 run_fused_bwd.py --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384
```

- [ ] **Step 2: Extract evidence from both reports**

Record kernel duration, local/shared spilling requests and overhead, register
count, TMEM-related source/SASS evidence, SM throughput, and replay-pass count.
Use the NCU Python API or `ncu --import ... --page details`; do not infer a
spill reduction from source allocation alone.

- [ ] **Step 3: Write the comparison report**

Document the environment, exact shape, commands, correctness results, 512 vs
448 TMEM columns, baseline/candidate metrics, and whether the optimization is
accepted or rejected. Keep the report in the new run directory.

### Task 6: Final verification and handoff

**Files:**
- Review: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py`
- Review: `tests/test_sm100_tmem_reuse.py`
- Review: `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/REPORT.md`
- Update if warranted: `AGENT_MEMORY.md`

- [ ] **Step 1: Run fresh verification**

```bash
python3 -m pytest tests/test_sm100_tmem_reuse.py -q
git diff --check
git status --short
```

- [ ] **Step 2: Check requirements line by line**

Confirm: 448 columns generated, numerical tests passed, exact workload ran,
the single requested NCU report exists with source imported and clock control
disabled, and spill/performance claims match freshly extracted metrics.

- [ ] **Step 3: Preserve user changes**

Leave all pre-existing untracked profile directories untouched. Do not stage,
commit, push, or clean files unless the user explicitly asks.
