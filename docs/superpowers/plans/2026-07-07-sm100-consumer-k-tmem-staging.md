# SM100 Consumer-K TMEM Staging Implementation Plan

> **For agentic workers:** Execute inline in this worktree. Do not create a subagent unless the user explicitly requests delegation. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Reduce the remaining Consumer-K local-memory spill traffic by staging dead values in existing TMEM, without increasing the physical 480-column TMEM allocation.

**Architecture:** Reuse the full `[64,128]` `dk_fragment` before its original stage-08 dK lifetime. Stage 04 loads U into it; stage 05 forms U times dVg in it; stage 06 reduces it before dK overwrites it after bar_08. Stage 06--08 stage dVg in `u_tmem` and K in `dv_tmem` within their already-proven dead windows. Existing barriers and producer GEMMs remain in their original order.

**Tech Stack:** Python, TileLang 0.1.9, PyTorch 2.10, pytest, CUDA SM100/GB200, Nsight Compute 2026.1.

## Execution revision (authoritative)

The half-fragment proposal below was attempted and rejected by TileLang layout
inference when `[64,128]` and `[64,64]` fragments coexist in the Consumer-K
parallel scope. It was not used as a fallback.

Completed implementation and source checks use the following invariant:

- `dk_fragment` holds U and then U*dVg in stages 04--06, and is reduced before bar_07.
- dVg is staged through `u_tmem`; K is staged through `dv_tmem`.
- stage 08 reloads K first, then overwrites `dk_fragment` with dK after bar_08.

The unchecked half-fragment task text is retained as rejected-history only;
the source, structural test, and profile artifacts are the implementation record.

---

### Task 1: Add failing Consumer-K structural tests

**Files:**

- Modify: tests/test_sm100_tmem_reuse.py
- Read: flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py:154-159
- Read: flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py:507-555

- [ ] **Step 1: Require the new allocation shape and no physical TMEM growth**

Append a test that uses allocation_targets() and dataflow_events(). It must assert:

    assignments, alloc_tmem_targets, alloc_fragment_targets = allocation_targets()
    assert len(alloc_tmem_targets) == 10
    assert "odot_fragment_1" not in alloc_fragment_targets
    assert "u_half_fragment" in alloc_fragment_targets
    assert isinstance(assignments["dq_tmem"], ast.Name)
    assert assignments["dq_tmem"].id == "u_tmem"

- [ ] **Step 2: Require stage-04 U-half staging before bar_05**

Use event_lines() to locate these events:

    T.copy(u_tmem, dv_fragment)
    for j_s, j_v in T.Parallel(block_S, DK // 2):
        u_half_fragment[j_s, j_v] = dv_fragment[j_s, j_v]
    T.copy(u_half_fragment, dp_tmem)
    for j_s, j_v in T.Parallel(block_S, DK // 2):
        u_half_fragment[j_s, j_v] = dv_fragment[j_s, j_v + DK // 2]
    T.copy(u_half_fragment, da_tmem)

Require:

    bar_04_wait < u_left < u_left_store < u_right < u_right_store < bar_05_arrive

where bar_04_wait is the Consumer-K bar_04 wait and bar_05_arrive is its following arrival.

- [ ] **Step 3: Require the dead-buffer windows**

Add assertions for:

    T.copy(dv_fragment, u_tmem)
    T.copy(k_shared, dv_fragment)
    T.copy(dv_fragment, dv_tmem)
    T.copy(u_tmem, dv_fragment)
    T.copy(dv_tmem, dv_fragment)

Require:

    bar_06_wait < dvg_stage < k_stage < dvg_restore < bar_08_arrive < k_restore

Also use buffer_mutations_between() to require no Consumer-K source access to dp_tmem or da_tmem after bar_06_wait and before the stage-08 K restore.

- [ ] **Step 4: Require two ordered half reductions**

Require exactly one each of:

    T.reduce_sum(u_half_fragment, dg_fragment_1, dim=1, clear=True)
    T.reduce_sum(u_half_fragment, dg_fragment_1, dim=1, clear=False)

Both must precede the Consumer-K bar_06 arrival. Preserve the existing stage-08 dK reduction, but require its source to be dv_fragment rather than odot_fragment_1.

- [ ] **Step 5: Verify RED**

Run:

    python3 -m pytest --noconftest tests/test_sm100_tmem_reuse.py -q

Expected: the new Consumer-K test fails because the current source still declares odot_fragment_1 and lacks all staging copies. The existing nine checks pass.

### Task 2: Implement the source-only Consumer-K lifetime change

**Files:**

- Modify: flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py:154-159
- Modify: flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py:507-555

- [ ] **Step 1: Replace the full odot array**

Change the Consumer-K declarations to:

    dk_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
    dv_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
    u_half_fragment = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
    dg_fragment_1 = T.alloc_fragment((block_S), dtype=accum_dtype)
    dg_last_local_1 = T.alloc_fragment((1), dtype=accum_dtype)

Do not add T.alloc_tmem, change dq_tmem = u_tmem, or touch Consumer-A/Consumer-S declarations.

- [ ] **Step 2: Persist U halves in stage 04**

Replace the single U-to-odot load with:

    T.barrier_wait(bar_04, (i_s + 0) % 2)
    T.copy(u_tmem, dv_fragment)
    for j_s, j_v in T.Parallel(block_S, DK // 2):
        u_half_fragment[j_s, j_v] = dv_fragment[j_s, j_v]
    T.copy(u_half_fragment, dp_tmem)
    for j_s, j_v in T.Parallel(block_S, DK // 2):
        u_half_fragment[j_s, j_v] = dv_fragment[j_s, j_v + DK // 2]
    T.copy(u_half_fragment, da_tmem)
    T.barrier_wait(tcbar_04, (i_s + 0) % 2)
    T.barrier_arrive(bar_05)

- [ ] **Step 3: Form two U times dVg reductions in stage 05**

Keep final-dV loading, the dqkv_shared publication, and the full dVg scaling. Replace the old full odot multiply with:

    T.copy(dp_tmem, u_half_fragment)
    for j_s, j_v in T.Parallel(block_S, DK // 2):
        u_half_fragment[j_s, j_v] *= dv_fragment[j_s, j_v]
    T.reduce_sum(u_half_fragment, dg_fragment_1, dim=1, clear=True)
    T.copy(da_tmem, u_half_fragment)
    for j_s, j_v in T.Parallel(block_S, DK // 2):
        u_half_fragment[j_s, j_v] *= dv_fragment[j_s, j_v + DK // 2]
    T.reduce_sum(u_half_fragment, dg_fragment_1, dim=1, clear=False)
    T.barrier_arrive(bar_06)

Do not read dp_tmem or da_tmem after that arrival.

- [ ] **Step 4: Use u_tmem and dv_tmem only in their dead windows**

Replace stages 06--08 dataflow with:

    # 06
    T.barrier_wait(bar_06, (i_s + 0) % 2)
    T.copy(dv_fragment, u_tmem)
    T.copy(dg_fragment_1, dg_shared)
    T.copy(k_shared, dv_fragment)
    T.copy(dv_fragment, tmp_shared_2_2)
    T.copy(dv_fragment, dv_tmem)
    T.barrier_arrive(bar_07)

    # 07
    T.barrier_wait(bar_07, (i_s + 0) % 2)
    T.copy(u_tmem, dv_fragment)
    T.copy(dv_fragment, tmp_shared_2_3)
    T.barrier_wait(tcbar_07, (i_s + 0) % 2)
    T.barrier_arrive(bar_08)

    # 08
    T.barrier_wait(bar_08, (i_s + 0) % 2)
    T.copy(dv_tmem, dv_fragment)
    T.copy(dk_tmem, dk_fragment)
    for j_s, j_k in T.Parallel(block_S, DK):
        dk_fragment[j_s, j_k] *= g_rev_exp_shared[j_s]
    T.copy(dk_fragment, dk_tmem)
    for j_s, j_k in T.Parallel(block_S, DK):
        dv_fragment[j_s, j_k] *= -dk_fragment[j_s, j_k]
    T.reduce_sum(dv_fragment, dg_fragment_1, dim=1, clear=True)

Do not move tcbar_07, bar_08, tcbar_08, bar_09, or any producer GEMM.

- [ ] **Step 5: Verify GREEN locally**

Run:

    python3 -m pytest --noconftest tests/test_sm100_tmem_reuse.py -q
    python3 -m py_compile flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py
    git diff --check

Expected: all structural checks pass, Python compilation succeeds, and diff check is silent.

### Task 3: Build an isolated GB200 profile run

**Files:**

- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/harness/
- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/analysis/
- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/reports/
- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/REPORT.md

- [ ] **Step 1: Make a new run directory and copy reusable harnesses**

Run:

    RUN=profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707
    mkdir -p "$RUN"/{harness,analysis,reports,cache,home}
    cp profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/run_fused_bwd.py "$RUN"/harness/
    cp profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/harness/start_gb200_session.sh "$RUN"/harness/
    cp profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/analyze_ncu_v4.py "$RUN"/analysis/
    cp profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/analysis/ncu_utils.py "$RUN"/analysis/

Change only the copied launcher's RUN_NAME. Create a new run-local run_variant.sh that imports baseline from a pre-edit /tmp/flashqla_tmem480_baseline snapshot and candidate from this worktree.

- [ ] **Step 2: Use the established GB200 session contract**

Before allocation, inspect existing allocations and session state. The repository has no .session/config.sh, so reuse the prior run-local launcher rather than inventing cluster parameters. It already records the approved GB200-NVL partition, account, QoS, image, worktree, and four-hour limit. Record allocation ID, node, GPU UUID, image, branch, and source SHA in the new analysis/allocation.txt.

- [ ] **Step 3: Snapshot baseline and compile candidate**

Inside the container before the candidate run:

    cp -a /workspace/AI_tests/.codex-worktrees/FlashQLA/sm100-dq-tmem-reuse/flash_qla /tmp/flashqla_tmem480_baseline/

Run the candidate with a fresh extension cache:

    bash "$RUN/harness/run_variant.sh" candidate --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --metadata-out "$RUN/analysis/candidate_metadata.json"

Expected: a fused backward launch, six output shapes, and generated CUDA captured into the new harness directory.

### Task 4: Validate generated code and numerical behavior

**Files:**

- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/harness/generated_fused_bwd_sm100_consumer_k.cu
- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/analysis/resources_consumer_k.txt
- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/analysis/correctness.md

- [ ] **Step 1: Audit generated TMEM and static resources**

Require ten balanced allocations/deallocations totaling 5*64 + 5*32 = 480 columns, no odot_fragment_1 C++ array, a 64-wide half fragment, and LDTM/STTM staging operations. Rebuild captured generated CUDA with ptxas verbose output and record registers, stack frame, spill load/store bytes, and static LDL/STL against the committed 480-column values: 128 registers/thread, 144-byte stack, 112 LDL, and 40 STL.

- [ ] **Step 2: Run the four selected FP64-reference tests**

Run:

    python3 -m pytest 'tests/test_gdr_unit.py::test_bwd[no_h0-kv-B1-T4096-H4]' 'tests/test_gdr_unit.py::test_bwd[h0-kv-B1-T4096-H4]' 'tests/test_gdr_unit.py::test_bwd[h0-vk-B1-T4096-H4]' 'tests/test_gdr_unit.py::test_bwd[h0-kv-B1-T4096-H64G16-padding]' -q --maxfail=1

Expected: all four pass. On the first mismatch, preserve generated CUDA/logs and stop before performance collection.

- [ ] **Step 3: Run the exact BF16 varlen case three times**

Run three independent candidate processes:

    for i in 1 2 3; do
      bash "$RUN/harness/run_variant.sh" candidate --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --metadata-out "$RUN/analysis/exact_$i.json"
    done

Expected: three zero return codes and the same output shapes.

### Task 5: Measure spill traffic and timing

**Files:**

- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/reports/full_source_import_source_clock_none_varlen_b2x8192_h64_consumer_k.ncu-rep
- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/analysis/compare_tmem480_vs_consumer_k.json
- Create: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/analysis/timing_summary.json
- Modify: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/REPORT.md

- [ ] **Step 1: Collect one direct-Python Full+Source NCU report**

Invoke Python directly, not the wrapper:

    export PYTHONPATH=/tmp/flashqla_site_spill:/workspace/AI_tests/.codex-worktrees/FlashQLA/sm100-dq-tmem-reuse
    ncu --set full --section SourceCounters --section PmSampling --section PmSampling_WarpStates --import-source yes --clock-control none -k 'regex:tilelang_fused_chunk_gdr_bwd_kernel_kernel' -c 1 -o "$RUN/reports/full_source_import_source_clock_none_varlen_b2x8192_h64_consumer_k" python3 "$RUN/harness/run_fused_bwd.py" --variant candidate --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384

Expected: one range, one target action, imported source, and no No kernels were profiled warning.

- [ ] **Step 2: Compare NCU metrics and source hotspots**

Use the run-local analyzer and NCU Python API to compare the committed 480-column report with the new report. Archive local LD, local ST, local spill requests, shared spill, registers, launch stack, TMEM LDT/STT instructions, long/short-scoreboard ratios, and source-PC hotspots. Specifically verify whether the former Consumer-K generated-CUDA 620/634 cluster declines.

- [ ] **Step 3: Run alternating CUDA-event timing pairs**

For baseline/candidate, candidate/baseline, baseline/candidate, use 50 warmups and 100 repeats with fresh HOME, TileLang cache, and extension directories per process:

    bash "$RUN/harness/run_variant.sh" baseline --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --warmup 50 --repeats 100 --timing-json "$RUN/analysis/baseline_1.json"
    bash "$RUN/harness/run_variant.sh" candidate --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --warmup 50 --repeats 100 --timing-json "$RUN/analysis/candidate_1.json"
    bash "$RUN/harness/run_variant.sh" candidate --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --warmup 50 --repeats 100 --timing-json "$RUN/analysis/candidate_2.json"
    bash "$RUN/harness/run_variant.sh" baseline --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --warmup 50 --repeats 100 --timing-json "$RUN/analysis/baseline_2.json"
    bash "$RUN/harness/run_variant.sh" baseline --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --warmup 50 --repeats 100 --timing-json "$RUN/analysis/baseline_3.json"
    bash "$RUN/harness/run_variant.sh" candidate --varlen --num-seqs 2 --seqlen 16384 --heads 64 --cu-seqlens 0,8192,16384 --warmup 50 --repeats 100 --timing-json "$RUN/analysis/candidate_3.json"

Compute candidate/baseline medians per pair, the median of three ratios, and drop-first-sample sensitivity. Treat NCU replay duration only as diagnostic.

- [ ] **Step 4: Apply gates and write REPORT.md**

Accept only if all are true relative to committed TMEM480:

    local LDL delta <= -10%
    local STL delta <= -10%
    local spill-request delta <= -10%
    paired median candidate/baseline <= 1.01
    all selected correctness tests pass
    physical TMEM allocation == 480 columns

Record source/generated/report hashes, exact commands, allocation metadata, static/dynamic comparisons, timing samples, and any shared-memory or TMEM-traffic regression. Keep raw reports, binaries, caches, and snapshots untracked.

### Task 6: Final review and handoff

**Files:**

- Modify: AGENT_MEMORY.md
- Modify: profile/sm100-fused-gdn-bwd-consumer-k-tmem-stage-varlen-b2x8192-h64-20260707/REPORT.md

- [ ] **Step 1: Run final host checks**

Run:

    python3 -m pytest --noconftest tests/test_sm100_tmem_reuse.py -q
    git diff --check
    git status --short

Expected: structural test passes, diff check is silent, and raw profile artifacts are not staged.

- [ ] **Step 2: Update durable memory and request commit direction**

Record the measured result, source/report hashes, accepted or rejected decision, and residual risks in AGENT_MEMORY.md. Do not commit or push the source change automatically; ask the user after all validation evidence is available.
