# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from dataclasses import dataclass

DV = 128
FWD_TARGET_FRAC = 0.7  # blackwell/fused_fwd.py:12

COMPUTE_VERSION_ARCH = {"9.0": "sm90", "10.0": "sm100",
                        "10.3": "sm103", "12.0": "sm120"}

BACKEND_OF = {"sm90": "hopper", "sm100": "blackwell",
              "sm103": "blackwell", "sm120": "blackwell_sm120"}


@dataclass(frozen=True)
class LaunchShape:
    H: int
    n_part: int
    P: int
    n_raw: int = 1

    @property
    def grid(self) -> int:
        return self.H * self.n_part  # == real_batch_size * H in the kernels


@dataclass(frozen=True)
class LaunchConfig:
    kernel: str
    block_DV: int
    precision: str | None = None

    @property
    def ways(self) -> int:
        # the kernel grid is ways * H * batch, e.g. hopper/cp_fwd.py:392
        return -(-DV // self.block_DV)

    @property
    def row(self) -> str:
        return coef_row(self.kernel, self.block_DV, self.precision)


def coef_row(kernel: str, block_DV: int, precision: str | None = None) -> str:
    if precision is not None:
        return f"{kernel}_{precision}_dv{int(block_DV)}"
    return f"{kernel}_dv{int(block_DV)}" if kernel in DV_TILED else kernel


@dataclass(frozen=True)
class Rule:
    widths: tuple[int, ...]  # every block_DV pick can return; coef_rows takes it as given
    pick: Callable[[LaunchShape], int]


_BLACKWELL_FWD = Rule(
    (64, 128),
    # blackwell/fused_fwd.py:861-864
    lambda shape: 128 if shape.grid >= int(FWD_TARGET_FRAC * shape.P) else 64,
)


def _hopper_fwd_pick(shape: LaunchShape) -> int:
    target = int(FWD_TARGET_FRAC * shape.P)
    if shape.grid >= target:
        return 128
    if shape.grid * 2 >= target:
        return 64
    return 32


_HOPPER_FWD = Rule((32, 64, 128), _hopper_fwd_pick)


@functools.lru_cache(maxsize=2)
def _correction_selector(backend: str):
    # Resolve the policy owned by the same architecture as the measured kernel.
    return importlib.import_module(
        f"flash_qla.ops.gated_delta_rule.chunk.{backend}.cp_fwd"
    ).select_correction_config


def _correction_pick(shape: LaunchShape, *, backend: str) -> int:
    return _correction_selector(backend)(
        shape.H, shape.n_raw, DV, num_sms=shape.P,
    )[1]


# Both layouts, seeded/unseeded scans and both directions share these choices.
# Each direction still has independently fitted latency coefficients.
_CORRECTION_CONFIGS = {
    "hopper": (("fp32", 16), ("tf32x3", 32), ("tf32x3", 64), ("tf32x3", 128)),
    "blackwell": (("fp32", 16), ("tf32x3", 32), ("tf32x3", 64), ("tf32x3", 128)),
}


def correction_configs(backend: str, kernel: str):
    return _CORRECTION_CONFIGS[backend]


_CORRECTION_RULES = {
    backend: Rule(tuple(sorted({width for _, width in configs})),
                  functools.partial(_correction_pick, backend=backend))
    for backend, configs in _CORRECTION_CONFIGS.items()
}

BLOCK_DV = {
    "hopper": {
        "fused_fwd": _HOPPER_FWD,                    # hopper/fused_fwd.py:730-736
        "correct_h0": _CORRECTION_RULES["hopper"], "correct_dht": _CORRECTION_RULES["hopper"],
    },
    "blackwell": {
        "fused_fwd": _BLACKWELL_FWD,
        "correct_h0": _CORRECTION_RULES["blackwell"], "correct_dht": _CORRECTION_RULES["blackwell"],
    },
    "blackwell_sm120": {
        "fused_fwd": 128,                            # blackwell_sm120/fused_fwd.py:726
        "correct_h0": 32, "correct_dht": 32,         # blackwell_sm120/cp_fwd.py:282
    },
}

BACKENDS = tuple(BLOCK_DV)

DV_TILED = tuple(dict.fromkeys(k for rules in BLOCK_DV.values() for k in rules))


def _rule(backend: str, kernel: str) -> Rule | int:
    if backend not in BLOCK_DV:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
    return BLOCK_DV[backend].get(kernel, DV)


def widths(backend: str, kernel: str) -> tuple[int, ...]:
    rule = _rule(backend, kernel)
    return rule.widths if isinstance(rule, Rule) else (rule,)


def launch_config(backend: str, kernel: str, shape: LaunchShape) -> LaunchConfig:
    rule = _rule(backend, kernel)
    if rule is _CORRECTION_RULES.get(backend):
        precision, block_DV = _correction_selector(backend)(
            shape.H, shape.n_raw, DV, num_sms=shape.P,
        )
        if (precision, block_DV) not in correction_configs(backend, kernel):
            raise ValueError(f"AutoCP has no coefficient row for {backend}/{precision}/DV{block_DV}")
        return LaunchConfig(kernel, block_DV, precision)
    if not isinstance(rule, Rule):
        return LaunchConfig(kernel, rule)
    block_DV = rule.pick(shape)
    if block_DV not in rule.widths:
        raise ValueError(
            f"{backend}/{kernel} picked block_DV={block_DV} for {shape}, which is "
            f"outside its declared widths {rule.widths}; no coefficient row was "
            f"fitted for it")
    return LaunchConfig(kernel, block_DV)


def coef_rows(backend: str, kernels) -> tuple[str, ...]:
    out: list[str] = []
    for kernel in kernels:
        configs = (correction_configs(backend, kernel)
                   if _rule(backend, kernel) is _CORRECTION_RULES.get(backend)
                   else tuple((None, width) for width in widths(backend, kernel)))
        for precision, width in configs:
            row = coef_row(kernel, width, precision)
            if row not in out:
                out.append(row)
    return tuple(out)


@functools.lru_cache(maxsize=1)
def current_arch() -> str:
    import tilelang
    version = tilelang.contrib.nvcc.get_target_compute_version()
    if version not in COMPUTE_VERSION_ARCH:
        raise ValueError(
            f"FlashQLA supports sm90, sm100, sm103 and sm120 only. "
            f"Found compute version: {version}")
    return COMPUTE_VERSION_ARCH[version]
