# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Architecture detection and the per-arch capability table.

Deliberately dependency-light: it imports ``tilelang.contrib.nvcc`` (guarded) and
nothing from ``flash_qla``, so ``conftest.py`` can use it during collection even on
a machine where ``import flash_qla`` would fail.

The capability table is the single place that records what each arch can actually
run, so tests and profiling stop hard-coding arch names:

* ``chunk_size``   -- must match ``flash_qla...chunk.CHUNK_SIZE``
* ``intra``        -- intra-card CP
* ``inter``        -- inter-card CP
* ``inter_intra_fwd`` / ``inter_intra_bwd`` -- independent Dual-CP directions.
  SM120/SM121 provide forward aggregation only; ordinary backward is separate.
"""
from __future__ import annotations

import torch

GPU_AVAILABLE = torch.cuda.is_available()

_COMPUTE_VERSION_TO_ARCH = {
    "9.0": "SM90",
    "10.0": "SM100",
    "10.3": "SM103",
    "12.0": "SM120",
    "12.1": "SM121",
}


def detect_arch() -> str | None:
    """Return the arch tag (``"SM100"`` ...) or ``None`` when it cannot be determined."""
    if not GPU_AVAILABLE:
        return None
    try:
        import tilelang.contrib.nvcc

        cv = tilelang.contrib.nvcc.get_target_compute_version()
    except Exception:
        return None
    return _COMPUTE_VERSION_TO_ARCH.get(cv)


ARCH = detect_arch()

ARCH_CAPS: dict[str, dict] = {
    "SM90": dict(chunk_size=64, bwd=True, intra=True, inter=True, inter_intra_fwd=True, inter_intra_bwd=True),
    "SM100": dict(chunk_size=64, bwd=True, intra=True, inter=True, inter_intra_fwd=True, inter_intra_bwd=True),
    "SM103": dict(chunk_size=64, bwd=True, intra=True, inter=True, inter_intra_fwd=True, inter_intra_bwd=True),
    "SM120": dict(chunk_size=32, bwd=True, intra=True, inter=True, inter_intra_fwd=True, inter_intra_bwd=False),
    "SM121": dict(chunk_size=32, bwd=True, intra=True, inter=True, inter_intra_fwd=True, inter_intra_bwd=False),
}

# Used when the arch is unknown (no GPU / detection failed). Everything is off, so
# capability-gated tests skip rather than fail with a confusing kernel error.
_UNKNOWN_CAPS = dict(chunk_size=None, bwd=False, intra=False, inter=False, inter_intra_fwd=False, inter_intra_bwd=False)


def caps(arch: str | None = None) -> dict:
    arch = ARCH if arch is None else arch
    return ARCH_CAPS.get(arch, _UNKNOWN_CAPS)


def supports(capability: str, arch: str | None = None) -> bool:
    return bool(caps(arch).get(capability, False))
