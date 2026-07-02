# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Supported forward head dimensions for the GDN chunk kernels.

Single source of truth for the head-dim guards, established by a forward parity
sweep vs the fp64 reference on B200 (SM100). Kept as plain ``ValueError`` checks
(NOT ``assert``) on purpose: ``assert`` is stripped under ``python -O``, which
would silently re-admit the wrong-``head_dim_k`` widths documented below.

Blackwell (SM100, tcgen05 MMA), head_dim_k = the MMA **contraction** dim
(forward parity vs fp64 ref across the full config matrix — GQA, varlen, h0,
both state layouts):

    head_dim_k:  64 / 128                -> correct (all configs)
                 32                      -> correct only for state_v_first=False;
                                            SILENTLY WRONG for state_v_first=True
                 16 / 48 / 80 / 256      -> raise (no valid MMA atom / smem)
                 96 / 160 / 192          -> run but SILENTLY WRONG (rel-err ~0.4)

so head_dim_k is an **explicit set** {64, 128} — a modulo/range guard would let
the silently-wrong widths through, and even 32 is unsafe (layout-dependent). If
32 is ever needed, the vk (state_v_first) path must be fixed first. head_dim_v is
the free/output dim (walked by an explicit ``block_DV`` loop): flexible and never
silently wrong; validated for every multiple of 16 in [32, 256] across both
state layouts.

Hopper (SM90, wgmma) uses different MMA atoms and was NOT swept here, so it
stays at the original head_dim == 128.
"""

from __future__ import annotations

BLACKWELL_FWD_HEAD_DIM_K = (64, 128)
BLACKWELL_FWD_HEAD_DIM_V_MIN = 32
BLACKWELL_FWD_HEAD_DIM_V_MAX = 256
BLACKWELL_FWD_HEAD_DIM_V_MULTIPLE = 16

HOPPER_FWD_HEAD_DIM = 128


def blackwell_fwd_head_dim_v_supported(v: int) -> bool:
    return (
        v % BLACKWELL_FWD_HEAD_DIM_V_MULTIPLE == 0
        and BLACKWELL_FWD_HEAD_DIM_V_MIN <= v <= BLACKWELL_FWD_HEAD_DIM_V_MAX
    )


def validate_blackwell_forward_head_dims(head_dim_k: int, head_dim_v: int) -> None:
    """Raise ValueError if the (K, V) head dims aren't supported on Blackwell forward."""
    if head_dim_k not in BLACKWELL_FWD_HEAD_DIM_K:
        raise ValueError(
            f"FlashQLA forward: head_dim_k={head_dim_k} is not supported on Blackwell. "
            f"head_dim_k is the tcgen05 MMA contraction dim and must be one of "
            f"{BLACKWELL_FWD_HEAD_DIM_K}; other widths either raise or SILENTLY produce "
            f"incorrect results (e.g. 96/160/192)."
        )
    if not blackwell_fwd_head_dim_v_supported(head_dim_v):
        raise ValueError(
            f"FlashQLA forward: head_dim_v={head_dim_v} is not supported on Blackwell. "
            f"head_dim_v must be a multiple of {BLACKWELL_FWD_HEAD_DIM_V_MULTIPLE} in "
            f"[{BLACKWELL_FWD_HEAD_DIM_V_MIN}, {BLACKWELL_FWD_HEAD_DIM_V_MAX}] "
            f"(validated range)."
        )
