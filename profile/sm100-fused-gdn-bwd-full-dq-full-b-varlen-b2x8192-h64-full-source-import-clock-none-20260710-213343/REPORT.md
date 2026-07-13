# SM100 fused GDN backward — full dQ/full B

## Status

Collection completed successfully on NVIDIA GB200.

## Workload

- BF16, packed varlen: B=2, total T=16384, Hq/Hk/Hv=64, D=128, chunk=64
- `cu_seqlens=[0,8192,16384]`

## Collection

- one NCU report only
- `--set full --section SourceCounters --section PmSampling --section PmSampling_WarpStates`
- `--import-source yes`, `--clock-control none`
- kernel: `tilelang_fused_chunk_gdr_bwd_kernel_kernel`

## Artifact verification

- report: `reports/full_source_import_source_clock_none_varlen_b2x8192_h64_full_dq_full_b_20260710_213343.ncu-rep`
- SHA-256: `89c0676b51485e35072b507ff0bae0c0c4ccc80c95dfe4db1e71260ba26ff338`
- source import marker found once; source page and details page exported.
- collection ran 48 replay passes and reported one matching kernel.

## Context only

The report records 1,234,944 ns kernel duration, 128 CTA grid, 512 threads/CTA,
and 2,594 metrics. Because clock control is explicitly disabled, do not use this
single diagnostic collection as a benchmark result.
