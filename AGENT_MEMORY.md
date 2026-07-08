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
