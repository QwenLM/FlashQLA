# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import mlx.core as mx


def unpack(
    x: mx.array,  # [1, sum_T, *dims]
    cu_seqlens: mx.array,
) -> mx.array:
    assert x.shape[0] == 1
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]              # [B]
    max_len = int(mx.max(seqlens).item())

    t = mx.arange(max_len, dtype=mx.int32)                   # [max_len]
    src = mx.expand_dims(cu_seqlens[:-1], 1) + t             # [B, max_len]
    src = mx.clip(src, 0, x.shape[1] - 1)

    out = x[0][src]                                          # [B, max_len, *dims]

    valid = t < mx.expand_dims(seqlens, 1)                   # [B, max_len]
    for _ in x.shape[2:]:
        valid = mx.expand_dims(valid, -1)
    return mx.where(valid, out, mx.zeros_like(out))


def pack(
    x: mx.array,  # [B, max_T, *dims]
    cu_seqlens: mx.array,
) -> mx.array:
    sum_T = int(cu_seqlens[-1].item())

    i = mx.arange(sum_T, dtype=mx.int32)                     # [sum_T]
    # b_idx[i] = batch element that packed position i belongs to
    b_idx = (
        mx.expand_dims(i, 1) >= mx.expand_dims(cu_seqlens[1:], 0)
    ).sum(axis=1).astype(mx.int32)                           # [sum_T]
    t_idx = (i - cu_seqlens[b_idx]).astype(mx.int32)         # [sum_T]

    return x[b_idx, t_idx][None]                             # [1, sum_T, *dims]


def pad_and_reshape(
    x: mx.array,
    dim: int,
    chunk_size: int = 64,
) -> mx.array:
    seq_len = x.shape[dim]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size > 0:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[dim] = (0, pad_size)
        x = mx.pad(x, pad_width)
    shape = list(x.shape)
    shape[dim : dim + 1] = [-1, chunk_size]
    return x.reshape(shape)


def fill_last_chunk_of_g(
    g: mx.array,  # [B, N, C, Hv]
    num_tokens: int,
    cu_seqlens: mx.array = None,
    chunk_size: int = 64,
    reverse: bool = False,
) -> mx.array:
    if cu_seqlens is None:
        last_chunk_size = num_tokens % chunk_size
        if last_chunk_size > 0:
            B, N, C, Hv = g.shape
            g_last = g[:, -1, :, :]  # [B, C, Hv]
            if reverse:
                update = (
                    g_last[:, last_chunk_size - 1 : last_chunk_size, :]
                    + g_last[:, -1:, :]
                )
                new_last = mx.concatenate(
                    [
                        g_last[:, : last_chunk_size - 1, :],
                        update,
                        g_last[:, last_chunk_size:, :],
                    ],
                    axis=1,
                )
            else:
                fill_val = mx.broadcast_to(
                    g_last[:, last_chunk_size - 1 : last_chunk_size, :],
                    (B, C - last_chunk_size, Hv),
                )
                new_last = mx.concatenate(
                    [g_last[:, :last_chunk_size, :], fill_val], axis=1
                )
            g = mx.concatenate([g[:, :-1, :, :], new_last[:, None, :, :]], axis=1)
    else:
        batch_size = cu_seqlens.shape[0] - 1
        new_g_list = []
        for i in range(batch_size):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seqlen = end - start
            last_chunk_idx = seqlen // chunk_size
            lcs = seqlen % chunk_size
            g_i = g[i]  # [N, C, Hv]
            if lcs > 0:
                g_i_last = g_i[last_chunk_idx]  # [C, Hv]
                if reverse:
                    update = (
                        g_i_last[lcs - 1 : lcs, :] + g_i_last[-1:, :]
                    )
                    new_chunk = mx.concatenate(
                        [
                            g_i_last[: lcs - 1, :],
                            update,
                            g_i_last[lcs:, :],
                        ],
                        axis=0,
                    )
                else:
                    fill_val = mx.broadcast_to(
                        g_i_last[lcs - 1 : lcs, :],
                        (chunk_size - lcs, g_i_last.shape[-1]),
                    )
                    new_chunk = mx.concatenate(
                        [g_i_last[:lcs, :], fill_val], axis=0
                    )
                g_i = mx.concatenate(
                    [
                        g_i[:last_chunk_idx, :, :],
                        new_chunk[None, :, :],
                        g_i[last_chunk_idx + 1 :, :, :],
                    ],
                    axis=0,
                )
            new_g_list.append(g_i)
        g = mx.stack(new_g_list, axis=0)
    return g


def prepare_chunk_offsets(cu_seqlens: mx.array, chunk_size: int) -> mx.array:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]              # [B]
    n_chunks = (seqlens + chunk_size - 1) // chunk_size      # ceiling division
    return mx.concatenate([
        mx.array([0], dtype=mx.int32),
        mx.cumsum(n_chunks).astype(mx.int32),
    ])
