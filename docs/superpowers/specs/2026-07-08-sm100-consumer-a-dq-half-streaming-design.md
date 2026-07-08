# SM100 Consumer-A dQ Half-Streaming Spill Reduction Design

## Objective

Further reduce local-memory spilling in the SM100 fused GDN backward kernel
without adding physical Tensor Memory (TMEM), changing the producer schedule,
or accepting a timing regression.  The target workload is BF16 packed-varlen
with `B=2`, two 8192-token sequences, 64 Q/K and V heads, head dimension 128,
chunk size 64, and `cu_seqlens=[0,8192,16384]` on GB200/SM100.

The committed Consumer-K reuse baseline has 480 physical TMEM columns,
2,738,304 dynamic local-memory spilling requests, and 89/25 static LDL/STL
PCs.  In Consumer-A, the generated kernel still materializes a separate
`dq_fragment[64]` during stages 09--10.  This is an evidence-backed target,
but the exact spill reduction remains a hypothesis until compilation and NCU
measurement.

## Selected design

Remove `dq_fragment` and stream dQ through the existing 32-element
`a_fragment` twice.  The two 64-by-64 halves of dQ map to the same logical
shape as `a_fragment`; all four Consumer-A warps participate in each half.
No new `T.alloc_tmem`, buffer alias, or barrier is introduced.

### Lifetime contract

At stage 08, Consumer-A copies both Q halves from `q_shared` into
`tmp_shared_2_1`.  After those two copies complete, the prior contents of
`a_fragment` are dead.  Stage 11 overwrites `a_fragment` from `a_shared`, so
stages 09--10 may reuse it without affecting any later Consumer-A operation.

`dq_tmem` (the physical alias of `u_tmem`) becomes valid for the initial dQ
after Consumer-A waits on `tcbar_08`.  Consumer-A scales and persists that
initial dQ before arriving at `bar_10`; the producer then accumulates
`dP @ K` and signals `tcbar_10`.  Consumer-A's unchanged stage-10 waits
observe that completed accumulation before publication to `dqkv_shared`, and
its `bar_11` arrival remains the release to the producer's next stage.

### Stage 09: scale, persist, and reduce dQ in two halves

For the left half and then the right half, in that order:

1. Copy the corresponding `[64,64]` slice of `dq_tmem` to `a_fragment`.
2. Apply the existing `g_exp_shared` and `scale` factors.
3. Copy the scaled half back to the same `dq_tmem` slice so the dQ result is
   preserved before the destructive Q dot product.
4. Multiply it by the corresponding Q half in `tmp_shared_2_1` and reduce it
   into `dg_fragment_2` with `clear=False`.

Both reductions must use `clear=False`: `dg_fragment_2` already contains the
stage-07 dP contribution, and the right-half reduction must accumulate the
left-half contribution rather than replace it.

### Stage 10: publish dQ in two halves

After the unchanged `bar_10` and `tcbar_10` waits, load each dQ half from its
`dq_tmem` slice into `a_fragment` and copy it to the matching half of
`dqkv_shared`.  This replaces the single full-width dQ load/copy while
preserving the values and the producer/Consumer-K handoff.  `p_fragment` is
not repurposed; retaining one reusable 32-element fragment minimizes live
Consumer-A state.

## Constraints and non-goals

- Keep exactly ten balanced TMEM allocations totaling 480 columns
  (`5*64 + 5*32`).  `mask_tmem` is not reused in this change.
- Do not change Consumer-K, Consumer-S, producer GEMMs, chunk size, output
  interfaces, or the SM90 implementation.
- Do not alter `set_max_nreg` / warpgroup register budgets in this experiment.
- Do not add CTA, named, or tcgen05 barriers.

## Main implementation risk

The right-half expression `dq_tmem[:, DK//2:]` uses a nonzero TMEM slice
offset.  A prior Consumer-K half-fragment attempt encountered TileLang layout
inference limitations for such views.  Compilation is therefore a hard gate:
the implementation must either lower both half transfers with the intended
full-warp TMEM mapping or be rejected; no claim of spill reduction is made
from the source-level lifetime proof alone.

## Validation and acceptance gates

1. Add structural tests that prove `dq_fragment` is absent; stage 08 finishes
   the two Q snapshots before the first dQ-half overwrite; stage 09 has two
   ordered TMEM load/scale/store/dot/reduce sequences; and every new reduction
   uses `clear=False`.
2. Compile on GB200 and inspect generated CUDA/SASS.  Require no new TMEM
   allocation, a 480-column allocation total, no generated `dq_fragment[64]`,
   and no new barriers.  Record stack size, ptxas spill bytes, and static
   LDL/STL counts against commit `67cadab`.
3. Run the selected FP64-reference matrix and three independent exact BF16
   varlen launches for the target workload.
4. Collect one fresh NCU report with Full + Source + imported source and
   `clock=None` in a new `profile/` run directory.  Require dynamic local
   spilling requests to be strictly below 2,738,304; report local LD/ST,
   shared spilling, stalls, and resources alongside it.
5. Run at least three alternating CUDA-event baseline/candidate timing pairs
   on the same GB200 allocation.  Accept only if the paired median
   `candidate / baseline` ratio is `<= 1.000`; otherwise reject the candidate
   even if spilling improves.

The source edit is deliberately limited to Consumer-A stage 09/10 and the
now-unused `dq_fragment` allocation.  Correctness, lowering, local-spill, and
strict no-regression timing gates all must pass before it is considered an
optimization.
