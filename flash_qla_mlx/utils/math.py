# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import mlx.core as mx


def l2norm(x: mx.array, dim: int = -1, eps: float = 1e-6) -> mx.array:
    inv_norm = mx.rsqrt((x * x).sum(axis=dim, keepdims=True) + eps)
    return x * inv_norm
