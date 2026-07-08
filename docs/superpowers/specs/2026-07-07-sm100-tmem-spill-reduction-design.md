# SM100 TMEM-Assisted Spill Reduction Design

## Objective

Use part of the 64 TMEM columns freed by the `dq_tmem = u_tmem` lifetime
reuse to reduce compiler-generated local-memory loads/stores in the SM100
fused GDN backward kernel. Keep the candidate only if selected numerical tests
pass, local spilling materially decreases, and paired same-node median kernel
latency does not regress by more than 1%.

## Baseline

The accepted 448-column kernel uses 128 registers per thread and has a
1,024-byte stack frame. On the requested BF16 packed-varlen workload, its full
NCU report records 7,316,992 local loads, 5,012,352 local stores, and
12,307,072 local spilling requests. NCU attributes 99.70% of LDL and 100% of
STL instructions to register spilling; local traffic accounts for 62.55% of
L1TEX sectors.

The 448-column baseline declares eight large Consumer-A FP32 fragments. Two
are especially avoidable:

- `mask_fragment[32]` remains live from stage 00 through stage 07.
- `odot_fragment_2[64]` captures Q at stage 08 and remains live through stage
  09 because the producer starts overwriting `q_shared` after `bar_09`.

## Selected design

Add one `(block_S, block_S)` FP32 `mask_tmem` allocation. This is 32 physical
TMEM columns, increasing the compiled total from 448 to 480 columns.
`mask_tmem` holds only the persistent gate mask in stages 00-07; it is not
reused for Q.

The non-MMA-produced mask needs an explicit TCGEN05 Layout-E mapping in
TileLang 0.1.9:

```python
tilelang.layout.Layout(
    [block_S, block_S],
    lambda i, j: [i + (j // 32) * 64, j % 32],
)
```

This is a relative Consumer-A thread mapping; it must not include the warp
group's absolute `+256` thread offset. The layout output shape is `[128, 32]`.
`LowerTileOp` remaps the TMEM buffer to that output shape before
`LowerSharedTmem` selects its 32-column physical allocation.

At stage 08, use the existing `a_fragment` and `p_fragment` only as transient
64-column layout adapters and immediately store Q's left and right halves into
the existing BF16 `tmp_shared_2_1` snapshot. At stage 09, after scaled dQ has
been persisted to `dq_tmem`, multiply the complete FP32 `dq_fragment[64,128]`
by that shared Q snapshot and reduce it directly. The same snapshot remains
unchanged for the stage-12 `dP^T @ Q` and stage-14 `Q^T @ dOg` GEMMs.

## Consumer-A data flow

### Stages 00-07: mask in TMEM

- Stage 00 computes the mask in `a_fragment`, stores it to `mask_tmem`, loads
  P into `p_fragment`, and multiplies P by the still-live `a_fragment` mask.
- Stage 02 loads the mask into `p_fragment` while `a_fragment` holds A.
- Stage 06 loads the mask into `p_fragment` while `da_fragment` holds dA.
- Stage 07 loads the mask into `a_fragment`, applies it to `dp_fragment`, then
  reuses `p_fragment` for P after the mask is no longer needed.
- `mask_fragment` is removed. No barrier, MMA, or shared-memory handoff moves.

### Stages 08-09: shared-Q dot without a dedicated fragment

- Stage 08 copies Q's left half through `a_fragment` and right half through
  `p_fragment`, with each half immediately written to `tmp_shared_2_1`
  before `bar_09`. Neither fragment carries Q across the barrier.
- Stage 09 scales `dq_fragment` and stores it to `dq_tmem` before the
  destructive dot product.
- One complete `T.Parallel(block_S, DK)` loop multiplies `dq_fragment` by
  `tmp_shared_2_1`, followed by one `T.reduce_sum(..., clear=False)`.
- Stage 10 reloads dQ from TMEM after the unchanged `bar_10/tcbar_10` handoff.
- `odot_fragment_2` is removed entirely.

The BF16 Q input is exactly representable after BF16-to-FP32-to-BF16 snapshot
round-tripping for normal values. The new stage-09 read adds about 16 KiB of
shared-memory traffic per iteration (`64 * 128 * 2` bytes). Any latency or
bank-pressure cost is an inference until measured. The source still removes 96
FP32 fragment elements per thread, but actual spill reduction must be
established from generated CUDA/SASS and NCU.

## Rejected layout attempts and alternatives

- v1 directly multiplied 64-column `a_fragment/p_fragment` by slices of the
  128-column `dq_fragment`; TileLang rejected their incompatible fragment
  ownership maps.
- v2 copied each dQ slice into `dp_fragment`; slicing changed the range but
  retained the underlying 128-column source layout, so layout inference failed
  inside the fragment-to-fragment copy itself.
- v3 reached `LowerTmemCopy` after layout inference but failed with
  `Tmem buffer mask_tmem does not have a layout specified`: unlike an MMA
  output, the non-MMA mask has no producer from which TileLang can infer its
  TMEM layout.
- A full TMEM Q scratch adds traffic and depends on unproven nonzero TMEM-slice
  offset lowering in TileLang 0.1.9.
- Reading `q_shared` directly at stage 09 is invalid because producer TMA can
  overwrite it after `bar_09`.
- Register-budget changes remain a measured follow-up if this candidate is
  insufficient.

## Verification and acceptance

1. Add structural tests before production changes. They must fail on the old
   adapter source and lock the two immediate stage-08 shared writes, the full
   stage-09 shared-Q dot/reduction ordering, absence of both half adapters,
   mask-TMEM copy counts, and unchanged stage-12/14 Q GEMM signatures.
2. Compile on GB200 and inspect generated CUDA: ten TMEM allocations totaling
   480 columns, no dedicated mask/odot arrays, and balanced deallocations. The
   source pass chain predicts a `[128,32]` mask allocation, but the 480-column
   total remains pending generated-v4 verification.
3. Run selected fixed, state-orientation, initial-state, padded/varlen FP64
   reference tests plus three exact-workload smoke runs.
4. Collect a new single NCU report for the exact requested workload using
   Full + Source + PM sampling, imported source, and clock control `none`.
5. Compare LDL, STL, spilling requests, stack size, registers/thread,
   short/long-scoreboard stalls, and kernel duration against the 448-column
   baseline.
6. Benchmark baseline and candidate on the same GB200 allocation in alternating
   process order. Each trial performs warmup followed by repeated CUDA-event
   timing; report at least three A/B trial pairs and aggregate medians.

Accept only if all correctness checks pass, each of LDL, STL, and local
spilling requests decreases by at least 10%, and candidate median latency is
no more than 1% slower. Otherwise retain the 448-column kernel and record the
rejected experiment.

## Scope

Only the SM100 fused backward kernel, its focused structural test, and a new
self-contained profiling run are in scope. The SM90 kernel, public API, tensor
layouts other than the explicit mask-TMEM Layout-E, 16-stage barrier graph,
GEMM order, and existing profile runs remain unchanged. No files are staged or
committed without explicit user direction.
