# SM100 `dq_tmem` Lifetime Reuse Design

## Context

The fused SM100 GDN backward kernel currently creates ten independent TMEM
allocations. They reserve all 512 columns for the lifetime of the CTA even
though several logical values have disjoint lifetimes. In particular,
`u_tmem`, `dv_tmem`, and `dq_tmem` all have logical shape `[64, 128]`, use
FP32 accumulators, and each reserve 64 TMEM columns.

## Goal

Make `dq_tmem` reuse an existing 64-column allocation, reducing the compiled
TMEM reservation from 512 to 448 columns without changing equations, barrier
ordering, output accuracy, register allocation, or the public API.

## Considered Approaches

1. **Alias `dq_tmem` to `u_tmem` (selected).** `u_tmem` is last loaded at
   stage 05, while stage 08 creates `dq_tmem` with `clear_accum=True`. The
   intervening stage barriers make the two logical lifetimes disjoint. This is
   the smallest source change and keeps identical shape and dtype.
2. **Alias `dq_tmem` to `dv_tmem`.** The lifetimes are also disjoint, but the
   final `dv_tmem` consumer belongs to a different consumer warpgroup. This is
   a valid fallback if TileLang rejects the first alias, but it has a slightly
   broader synchronization argument.
3. **Introduce explicit TMEM offset/lifetime allocation.** This would make
   reuse more general, but requires compiler-facing allocation machinery and
   is outside the scope of this isolated experiment.

## Dataflow and Synchronization

- Stage 03 writes `U` into `u_tmem`.
- Stage 04 consumes `U`.
- Stage 05 overwrites the same allocation with `V'`, then all required
  consumers load `V'` before arriving at later stage barriers.
- Stage 08 waits on `bar_08` and initializes `dQ` with
  `clear_accum=True`; therefore no prior `u_tmem` value must survive.
- Existing barriers remain unchanged. The implementation only aliases the
  TileLang buffer handle; it does not add an unsynchronized overwrite.

## Implementation

In `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py`, retain the
`u_tmem` allocation and bind `dq_tmem` to that buffer instead of calling
`T.alloc_tmem` a second time. Do not modify any consumer or producer stage.

## Validation

1. Add a structural regression test before the implementation and confirm it
   fails against the current 512-column kernel.
2. Compile the SM100 kernel and inspect generated CUDA: nine TMEM allocations,
   448 total columns, and the `dQ` GEMMs targeting the aliased address.
3. Run GPU correctness tests, including both state layouts and varlen behavior.
4. Run the requested workload: BF16, `B=2`, two 8192-token sequences,
   64 Q/K/V heads, dimensions 128, and `cu_seqlens=[0,8192,16384]`.
5. Collect a fresh single NCU report with Full + Source, imported source, and
   `--clock-control none`; compare local spilling, duration, and TMEM source.

## Failure Handling

If compilation, synchronization, or correctness validation fails, remove only
the alias change and test the `dq_tmem`-to-`dv_tmem` fallback under the same
checks. A smoke-test success alone is not sufficient evidence because a TMEM
lifetime race can be input- or scheduling-dependent.
