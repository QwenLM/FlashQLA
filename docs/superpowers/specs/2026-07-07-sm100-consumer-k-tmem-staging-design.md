# SM100 Consumer-K TMEM Staging Spill Reduction Design

## Objective

Reduce the remaining local-memory spilling in the SM100 fused GDN backward
kernel without allocating additional physical Tensor Memory (TMEM) columns.
The selected workload remains BF16 packed-varlen with two 8192-token logical
sequences, 64 Q/K/V heads, head dimension 128, chunk size 64, and
`cu_seqlens=[0,8192,16384]` on GB200/SM100.

The accepted 480-column candidate still records 3,188,224 dynamic local loads,
2,062,720 dynamic local stores, and 5,228,672 local spilling requests. Static
SASS contains 112 LDL and 40 STL instructions. The strongest remaining source
cluster is Consumer-K: `dv_fragment` and `odot_fragment_1` each represent a
`[64,128]` FP32 logical tile and are simultaneously live around stages 04--08.

## Non-goals

- Do not increase the physical TMEM allocation above 480 columns.
- Do not change the 16-stage barrier graph, producer GEMM order, output API,
  chunk size, or SM90 implementation.
- Do not tune `set_max_nreg` in this experiment; register budgeting is a
  separate measured follow-up.

## Selected design

Use already allocated TMEM buffers only during windows in which their logical
values are dead. The implementation introduces no new `T.alloc_tmem` call.

| Scratch payload | Physical buffer(s) | Window | Why it is safe |
| --- | --- | --- | --- |
| U left/right halves, each `[64,64]` | `dp_tmem` / `da_tmem` | Consumer-K stage 04 through stage 05, before `bar_06` arrival | Producer first overwrites `dp_tmem` after `bar_06`; Consumer-A first overwrites `da_tmem` in stage 06 after `bar_06`. |
| dVg `[64,128]` | `u_tmem` (the physical buffer also aliased as `dq_tmem`) | after Consumer-K observes `bar_06`, through stage 07, before Consumer-K arrives at `bar_08` | Consumer-A has consumed U/V' before its `bar_06` arrival; producer first overwrites this physical storage with dQ after `bar_08`. |
| K `[64,128]` | `dv_tmem` | after Consumer-K's final stage-05 dV load, through Consumer-K stage 08 | Producer has finished every dV write by stage 04; no consumer reads dV TMEM after Consumer-K stage 05. |

### Stage changes

1. In Consumer-K stage 04, load U in two 64-wide halves. Immediately store
   the left half to `dp_tmem` and the right half to `da_tmem`. The temporary
   per-thread fragment is `[64,64]`, so only one 32-element half is live at a
   time instead of a full `odot_fragment_1[64]`.
2. In stage 05, load final dV once, publish unchanged dV to `dqkv_shared`, and
   scale it into dVg. Reload each U half from `dp_tmem` / `da_tmem`, multiply
   it by the matching dVg half, and perform two ordered reductions into the
   existing `dg_fragment_1` (`clear=True`, then `clear=False`). Both scratch
   reads complete before Consumer-K arrives at `bar_06`.
3. Immediately after Consumer-K waits on `bar_06`, store dVg to `u_tmem`.
   Load K into the normal 64-wide temporary, publish it to
   `tmp_shared_2_2`, then store K to `dv_tmem`. The original `dv_fragment`
   and K temporary are therefore dead across the stage-06/07 handoff.
4. In stage 07, reload dVg from `u_tmem`, publish it to `tmp_shared_2_3`, and
   finish that read before Consumer-K arrives at `bar_08`.
5. In stage 08, reload K from `dv_tmem`, combine it with dK exactly as before,
   and retain the existing `dg_fragment_1` / `dg_last_local_1` reductions.

## Synchronization contract

`bar_06` is the release point for the U/V' lifetime: Consumer-A reads V' from
`u_tmem` before it arrives, and Consumer-K reaches the same barrier only after
forming dVg. Therefore Consumer-K may write dVg to `u_tmem` only after its
`bar_06` wait. Consumer-K must reload dVg and complete the TMEM read before
its `bar_08` arrival, because the producer begins the dQ GEMM into the aliased
`dq_tmem` buffer after waiting for `bar_08`.

`dp_tmem` is scratch only before Consumer-K arrives at `bar_06`; it is never
read after that point because the producer may begin the dP GEMM immediately.
`da_tmem` scratch is likewise consumed before `bar_06`, before Consumer-A
stores dAb there in its stage 06. `dv_tmem` is overwritten only after
Consumer-K's final stage-05 dV load and is not accessed by the producer after
its stage-04 dV GEMM.

No additional CTA, warpgroup, named, or tcgen05 barrier is introduced. Every
new TMEM transfer follows the same TileLang `T.copy` lowering path already
used by this kernel; generated CUDA must show the expected `LDTM`/`STTM`
operations and no additional TMEM allocation/deallocation.

## Validation and acceptance

1. Add AST/source structural tests before production edits. They must reject a
   new TMEM allocation, an early `u_tmem` write, a late `u_tmem` read, or
   `dp_tmem` scratch traffic after `bar_06`.
2. Compile on GB200 and inspect generated CUDA/SASS for exactly ten balanced
   allocations totaling `5*64 + 5*32 = 480` columns. Compare stack frame,
   ptxas spill bytes, and static LDL/STL occurrences against the committed
   480-column candidate.
3. Run the existing selected FP64-reference matrix and three independent exact
   BF16 varlen runs before interpreting performance data.
4. Collect a fresh Full + Source/import-source NCU report with clock control
   `none`, in a new profile directory. Compare dynamic local LD, local ST,
   local spilling requests, shared spill, TMEM traffic, stall ratios, and
   resource metrics.
5. Use at least three alternating baseline/candidate CUDA-event timing pairs
   on one GB200 allocation. Accept the change only if each local LD, local ST,
   and local spilling-request count falls by at least 10% versus the committed
   480-column baseline and the paired median ratio is no slower than 1.01.

The design is an inference from source-level lifetimes and existing NCU/SASS
hotspots. It is not a correctness or performance claim until the generated
code, GPU tests, NCU report, and paired timing satisfy the listed gates.
