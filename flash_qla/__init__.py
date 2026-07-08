# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

__version__ = "0.1.1"

from flash_qla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule,
)
from flash_qla.ops.gated_delta_rule.fused_recurrent import (
    fused_recurrent_gdr_fwd,
    recurrent_gated_delta_rule,
    fused_recurrent_gdr_verify_fwd,
    recurrent_gated_delta_rule_verify,
)

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
    "fused_recurrent_gdr_fwd",
    "recurrent_gated_delta_rule",
    "fused_recurrent_gdr_verify_fwd",
    "recurrent_gated_delta_rule_verify",
]
