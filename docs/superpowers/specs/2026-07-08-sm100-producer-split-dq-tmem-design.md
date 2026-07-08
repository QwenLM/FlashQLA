# SM100 Producer-Split dQ TMEM Spill Reduction Design

## Objective and scope

This design supersedes the rejected `dq_tmem[:, half]` candidate. It removes
Consumer-A's live `dq_fragment[64,128]` without adding physical Tensor Memory
(TMEM), changing the dQ equation, or changing the SM90 kernel.

The target is GB200/SM100 BF16 packed-varlen: `B=2`, two 8192-token sequences,
64 Q/K heads, 64 V heads, head dimension 128, chunk size 64, and
`cu_seqlens=[0,8192,16384]`. The accepted Consumer-K-reuse baseline is
`67cadab`: 480 physical TMEM columns, 89/25 static LDL/STL sites, 96 B stack,
and 2,738,304 dynamic local spill requests.

## Rejected predecessor and constraint

The predecessor loaded `dq_tmem[:, :64]` and `dq_tmem[:, 64:]`. Its exact
target JIT failed in `CopyNode::LowerTmemCopy` with `Failed to find a suitable
instruction for tcgen05.ld. Check your layout.` It did not launch a backward
kernel, so it has no correctness, NCU, or timing result.

TileLang must match a complete physical TMEM layout to a supported TCGEN05
load. Existing generated CUDA proves that whole `64x64` `p_tmem` and
`dp_tmem` buffers load into a Consumer-A fragment with
`tcgen05_ld_32dp32bNx<32, false>`. This design therefore uses only whole
64x64 TMEM copies; it never takes a TMEM slice or offset view.

## Allocation and lifetime

Keep ten TMEM allocations totaling `5 * 64 + 5 * 32 = 480` columns. Replace
the `dq_tmem = u_tmem` alias with Python-level aliases only:

```python
dq_tmem_L = p_tmem
dq_tmem_R = dp_tmem
```

No `T.alloc_tmem` is added. `u_tmem` remains the U/V' workspace.

| Buffer | Last original Consumer-A read | New writer | Ordering proof |
| --- | --- | --- | --- |
| `p_tmem` | stage 07 | stage-08 dQ-left | Consumer-A arrives, Producer waits at `bar_08` |
| `dp_tmem` | stage 07 | stage-08 dQ-right | The same `bar_08` handoff |

Producer has no later use of the original P/dP values. Its stage-08 wait on
`bar_08` therefore proves both buffers are dead before either overwrite.

## Producer and barrier dataflow

Replace `tcbar_08` by `tcbar_08_L` and `tcbar_08_R`; replace `tcbar_10` by
`tcbar_10_L` and `tcbar_10_R`. All have `arrive_count=1`. The net cost is two
additional mbarriers. A single mbarrier must not signal two independent GEMMs,
because a consumer could observe only one completion.

At stage 08, split `dQ = dO @ S0^T` along output K:

- `state_v_first=True`: use `h_shared[:, :DK // 2]` and
  `h_shared[:, DK // 2:]` without
  B transpose.
- `state_v_first=False`: use `h_shared[:DK // 2, :]` and
  `h_shared[DK // 2:, :]` with
  `transpose_B=True`.

Each GEMM writes a full `64x64` `dq_tmem_L/R`, uses `clear_accum=True`, and
signals its own stage-08 barrier. Consumer-A waits for both before `bar_09`.

At stage 10, split `dQ += dP @ K` with `tmp_shared_2_2[:, :DK // 2]` and
`tmp_shared_2_2[:, DK // 2:]`. Each GEMM accumulates into the matching buffer with
`clear_accum=False` and signals its own stage-10 barrier. Consumer-A waits for
both before publishing dQ.

## Consumer-A dataflow

Delete `dq_fragment`. At stage 09, process left then right, each as:

1. Whole-load `dq_tmem_L/R` into the existing `a_fragment`.
2. Multiply by `g_exp_shared` and `scale`.
3. Whole-store it back to preserve scaled dQ.
4. Multiply by the matching 64-column Q snapshot and reduce to
   `dg_fragment_2` with `clear=False`.

Both reductions retain `clear=False`: the stage-07 dP term and first dQ-half
term must survive. At stage 10, after both stage-10 waits, whole-load each
buffer through `a_fragment` and copy it to the corresponding half of
`dqkv_shared`. Stage 11 onward is unchanged.

## Non-goals and expected effect

- Preserve both `state_v_first` paths, public API, chunk size, numerical
  formulae, Consumer-K, Consumer-S, and warpgroup register budgets.
- Do not repurpose `mask_tmem`, add a producer warpgroup, or use the rejected
  TMEM-slice path.
- The hypothesis is lower Consumer-A register pressure: a 64-element
  per-thread dQ fragment becomes serial use of existing 32-element
  `a_fragment`. TMEM capacity intentionally remains 480 columns.

## Test and acceptance gates

1. Add structural tests first. They must fail on the current half-slice source
   and require: no `dq_fragment`; L/R aliases to `p_tmem/dp_tmem`; no dQ TMEM
   subscripts; two producer mbarriers and two Consumer-A waits per stage; and
   correct branch-specific stage-08 inputs plus `clear_accum` modes.
2. Compile both `state_v_first` branches on GB200 before any correctness or
   profiling. Inspect generated CUDA for full 64x64 dQ loads, no
   `dq_fragment`, 480 TMEM columns, and four dQ mbarriers.
3. Run the selected FP64-reference matrix and three independent exact BF16
   target-varlen launches.
4. Collect one Full + Source + imported-source NCU report with `clock=None`.
   Dynamic local spill requests must be strictly below 2,738,304; record local
   LD/ST, shared spills, stack, registers, and static LDL/STL versus `67cadab`.
5. Run at least three alternating CUDA-event baseline/candidate pairs on the
   same GB200 allocation. Report all samples and accept only a paired median
   candidate/baseline ratio of `<= 1.01`.

Any compile-layout failure, correctness mismatch, spill non-improvement, or
performance regression rejects this candidate. There is no fallback to the
unsupported half-slice design.
