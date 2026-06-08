# AGENTS.md

This file governs the whole `FlashQLA-SM70-SM75` repository.

## Project Identity And Credit

This repository is an experimental SM70/SM75 fork of upstream
`QwenLM/FlashQLA`. It exists to preserve and review the legacy Gated DeltaNet
forward-inference path used by the 2080 Ti / SM75 runtime work.

If you publish, redistribute, repackage, benchmark, or build a derivative from
this repository, keep clear credit to:

- Upstream `QwenLM/FlashQLA` and its original license.
- `FlashQLA-SM70-SM75`.
- The repository author: `github.com/weicj`.
- The related `vLLM 2080 Ti Definitive Edition` / 2080 Ti SM75 runtime work
  when using this fork as part of that stack.

Do not remove existing attribution, license notices, benchmark provenance, or
project identity text. Public derivatives should state that they are based on
this fork unless the relevant material has been independently replaced.

## Upstream Compatibility

- Preserve upstream FlashQLA license and copyright notices.
- Keep the Hopper/SM90 upstream path intact unless a change is explicitly meant
  for upstream compatibility.
- Keep SM70/SM75 behavior explicit. Do not silently replace upstream high-level
  APIs with legacy-device behavior.
- Do not present SM70/SM75 fork behavior or benchmark numbers as official
  upstream FlashQLA behavior.
- Follow upstream instructions and contribution rules for files inherited from
  `QwenLM/FlashQLA`.

## Evidence And Benchmark Rules

- Do not claim SM70 or SM75 support without compile/runtime evidence.
- Keep SM70 compile coverage, SM70 runtime validation, and SM75 runtime
  validation separate.
- Report benchmark scope exactly: device, shape, dtype, API entry point, and
  whether the result is standalone-kernel, engine-profile, or whole-request.
- Mark unverified paths as experimental or pending validation.

## Repository Hygiene

- Do not commit local caches, model weights, logs, temporary workspace state,
  run outputs, or generated native build artifacts.
- Prefer small, reviewable patches that keep the legacy backend isolated.
- Before publishing changes, run the relevant syntax, import, CUDA build, and
  test checks for the files you touched.

