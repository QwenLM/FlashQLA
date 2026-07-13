# Agent memory

## SM100 fused GDN backward

### Historical baseline: 448-column dQ/u reuse

- The Blackwell fused backward kernel is in
  `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py`.
- For block/chunk size 64 and head dim 128, `u_tmem` and `dq_tmem` have
  non-overlapping logical lifetimes. Binding `dq_tmem = u_tmem` compiles and
  reduces physical TMEM from 512 to 448 columns without changing barriers.
- Generated CUDA for the reuse has nine physical TMEM allocations:
  `5 * 64 + 4 * 32 = 448` columns. TileLang canonicalizes the shared address
  under the generated symbol `dq_tmem`.
- GB200 correctness checks passed for fixed, initial-state, state-v-first, and
  padded/varlen selected cases. The exact BF16 B=2, packed T=16384,
  H=64, D=128, `cu_seqlens=[0,8192,16384]` workload completed three times.
- This reuse does not fix register spilling. A full/source/import-source NCU
  comparison measured local spilling requests at 12,307,072 and shared
  spilling requests at 2,944 for both 512- and 448-column kernels; both use
  128 registers/thread.
- The candidate report and full evidence live under
  `profile/sm100-fused-gdn-bwd-tmem448-varlen-b2x8192-h64-20260707/`.
- The historical next experiment was to tune warp-group register allocation or
  use the freed 64 TMEM columns to shorten a high-pressure register lifetime.

### Verified follow-up: 480-column mask scratch and shared-Q

- Final candidate source SHA-256 is
  `6d0285f5a8bf7ec5092a11b65811b5f4142651bfc4f45979da94b9c9e1d30928`.
  It preserves `dq_tmem = u_tmem`, adds an explicit 32-column mask scratch
  with bijective Layout-E, and snapshots Q to shared memory at stage 08 for the
  stage 09 destructive reduction and later stage 12/14 readers.
- Generated CUDA proves ten balanced TMEM allocations:
  `5 * 64 + 5 * 32 = 480` columns. The 32-column mask has one store and three
  loads with the required waits; generated `mask_fragment[32]` and
  `odot_fragment_2[64]` arrays are absent.
- On GB200/SM100, the selected four-case FP64-reference matrix passed, followed
  by three independent exact BF16 B=2, packed T=16384, H=64, D=128,
  `cu_seqlens=[0,8192,16384]`, varlen/chunk-64 runs with RC 0.
- Versus TMEM448, static stack fell 248 B -> 144 B; ptxas spill stores/loads
  fell 356/708 B -> 176/456 B; static LDL/STL sites fell 175/86 -> 112/40;
  registers/thread remained 128.
- The sole valid candidate Full+Source/import-source/clock-none NCU report has
  SHA-256 `32a7790ba05fdca3833e5d9f877653bd14ec5f36ceede94452d811b7c08bf278`.
  A first wrapper attempt returned RC 0 but reported `No kernels were profiled`
  and created no report; direct-Python attempt 2 produced the valid report.
- Dynamic local LD, local ST, and local spilling requests decreased 56.43%,
  58.85%, and 57.51%, all passing the <= -10% gates. Shared spilling stayed
  2,944; shared bank conflicts rose 12.90% and short-scoreboard ratio rose
  30.54%, so shared-Q banking remains a follow-up risk.
- Three paired CUDA-event median ratios were 0.908308, 0.905266, and 0.896299;
  median 0.905266 passed the <=1.01 gate. All six processes showed a first-
  sample ramp, but drop-first median 0.905175 left the decision unchanged.
- Decision: accept only for the validated BF16 varlen B2/T16384/H64/D128
  workload on GB200/SM100. Clocks were not fixed; other workloads/GPUs remain
  unproven, and NCU replay duration is diagnostic rather than wrapper timing.
- Overall evidence is in
  `profile/sm100-fused-gdn-bwd-tmem-mask-spill-varlen-b2x8192-h64-20260707/REPORT.md`;
  raw and intermediate evidence remains under that run's `analysis/` and
  `reports/` directories.

### Verified follow-up: Consumer-K dk-fragment reuse

- Candidate source SHA-256: `5a3b56f28bd14ec480f62033d96d46e4b54b2ede20a8f8ccbafc073f11b8cd54`.
- The workable TileLang design reuses full-width `dk_fragment` for U and U*dVg
  in stages 04--06; half-fragment mixing failed layout inference and is rejected.
- Generated CUDA retains ten balanced TMEM allocations (480 columns); static
  LDL/STL is 112/40 -> 89/25 and ptxas stack is 144 B -> 96 B at 128 registers.
- Full+Source/import-source/clock-none NCU measured local LD/ST 3,188,224/
  2,062,720 -> 1,680,896/1,079,680 and local spill requests 5,228,672 -> 2,738,304.
- Four selected FP64-reference tests and three exact BF16 varlen runs passed.
- Paired GB200 CUDA-event ratios were 0.939390, 0.924556, 0.944366; median
  0.939390 passed the <=1.01 gate.

### Rejected candidate: Consumer-A dQ TMEM half slices

- Candidate source SHA-256: `e5f3bf1b538215518a2ec679302d4b3a99817312f6bf3ae57937b9b1a8acc55f`.
- Structural regression tests passed 11/11 in the GB200 container, but the
  exact BF16 varlen target JIT failed before any backward launch at
  `CopyNode::LowerTmemCopy`: `Failed to find a suitable instruction for
  tcgen05.ld. Check your layout.`
- Root cause: TileLang only selects tcgen05.ld when the full TMEM-to-fragment
  physical layout matches a supported instruction; a `dq_tmem[:, half]` view
  does not. No correctness, NCU, or timing result exists for this candidate.
- Follow-up must use two separately addressed full 64x64 TMEM buffers and
  independent mbarriers, not nonzero or partial views of a 64x128 buffer.

### Verified benchmark: Consumer-K versus pre-reuse commit

- On GB200/SM100, the BWD matrix from `benchmark_results_GB200.txt` was run
  for exact snapshots `4c1109e6269b16910364767d52947e0ae1006174` and
  `67cadab3d1e3ce98bcd6554e8f86986661e01e27`: seven head configurations by
  six 32k varlen sequence partitions, 42 cases total.
- Three alternating CUDAGraph pairs used warmup=10 and repeats=100 per case;
  all six logs completed 42/42 cases with no benchmark errors. The geometric
  mean of the 42 per-case median `67cadab / 4c1109e` ratios was 0.948966
  (5.10% lower time; 1.054x faster). Pair-level GM ratios were 0.949034,
  0.948190, and 0.949516.
- The detailed report and raw logs are under
  `profile/sm100-fused-gdn-bwd-producer-split-dq-varlen-b2x8192-h64-20260708/analysis/`.

### Verified clean reproduction: uploaded benchmark commit

- A clean GB200/SM100 worktree at `5db6772` reran the 42-case BWD matrix
  (`B=1`, total `T=32768`, BF16 Q/K/V, varlen seed 42) with CUDAGraph,
  warmup=10, and repeats=100 per case. Both fresh runs completed 42/42.
- The two-run geometric-mean times were 0.935931954 ms and 0.937575107 ms;
  their ratio was 1.001755632. Against the uploaded three-run median, the
  fresh two-run median ratio was 0.994035720 (0.596% lower time).
- This is a small node/clock variation, not a kernel regression: neither
  experiment fixed clocks, and the clean reproduction used a different GB200
  node. Evidence is in
  `profile/gb200-bwd-matrix-repro-20260709/analysis/REPORT.md`.

### Verified follow-up: full dQ TMEM and full B operand

- A full `dq_tmem = u_tmem` (64x128) layout compiles when every TMEM load is
  full-width. The producer can use the full `h_shared` B operand and one
  mbarrier each for stages 08 and 10; no B half-slice scratch copy or its
  added wait is required.
- TileLang still rejects a 64x64 view of that 64x128 TMEM allocation at
  `CopyNode::LowerTmemCopy`. The valid consumer path stages the complete dQ
  buffer through the dead full-width `u_fragment`, aliases it as
  `dq_fragment`, and performs the scalar/dot work there before restaging it.
  This does not allocate another full fragment.
- Structural checks passed 17/17. On GB200/SM100, the exact BF16 packed
  varlen target (`B=2`, total T=16384, H=64, D=128,
  `cu_seqlens=[0,8192,16384]`, chunk 64) JIT-compiled and launched with RC 0;
  the FP64-reference case `test_bwd[h0-kv-B3-T4096-H4-varlen]` passed.
- This validates compilation and correctness, not performance. Re-profile
  before making a performance claim.
- The post-redesign diagnostic NCU run is
  `profile/sm100-fused-gdn-bwd-full-dq-full-b-varlen-b2x8192-h64-full-source-import-clock-none-20260710-213343/`.
  Its single Full+SourceCounters+PM-sampling, source-import, clock-none report
  has SHA-256 `89c0676b51485e35072b507ff0bae0c0c4ccc80c95dfe4db1e71260ba26ff338`.
