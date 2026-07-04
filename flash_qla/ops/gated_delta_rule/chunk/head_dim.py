# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Supported forward head dimensions for the GDN chunk kernels on Blackwell (SM100).

Single source of truth for the forward head-dim guards. Uses ``ValueError`` rather
than ``assert`` on purpose, since ``assert`` is stripped under ``python -O``.
"""

BLACKWELL_FWD_HEAD_DIM_K = (64, 128)
BLACKWELL_FWD_HEAD_DIM_V_MIN = 32
BLACKWELL_FWD_HEAD_DIM_V_MAX = 256
BLACKWELL_FWD_HEAD_DIM_V_MULTIPLE = 16


def blackwell_fwd_head_dim_v_supported(v: int) -> bool:
    return (
        v % BLACKWELL_FWD_HEAD_DIM_V_MULTIPLE == 0
        and BLACKWELL_FWD_HEAD_DIM_V_MIN <= v <= BLACKWELL_FWD_HEAD_DIM_V_MAX
    )


def validate_blackwell_forward_head_dims(head_dim_k: int, head_dim_v: int) -> None:
    if head_dim_k not in BLACKWELL_FWD_HEAD_DIM_K:
        raise ValueError(
            f"head_dim_k={head_dim_k} not supported on Blackwell forward; "
            f"must be one of {BLACKWELL_FWD_HEAD_DIM_K}."
        )
    if not blackwell_fwd_head_dim_v_supported(head_dim_v):
        raise ValueError(
            f"head_dim_v={head_dim_v} not supported on Blackwell forward; "
            f"must be a multiple of {BLACKWELL_FWD_HEAD_DIM_V_MULTIPLE} in "
            f"[{BLACKWELL_FWD_HEAD_DIM_V_MIN}, {BLACKWELL_FWD_HEAD_DIM_V_MAX}]."
        )
