# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import mlx.core as mx


def unpack(
    x: mx.array,  # [1, sum_T, *dims]
    cu_seqlens: mx.array,
) -> mx.array:
    assert x.shape[0] == 1
    batch_size = cu_seqlens.shape[0] - 1
    seqlens = [
        int((cu_seqlens[i + 1] - cu_seqlens[i]).item()) for i in range(batch_size)
    ]
    max_len = max(seqlens)
    rest = x.shape[2:]

    parts = []
    for i in range(batch_size):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        chunk = x[0, start:end]
        if end - start < max_len:
            pad_width = [(0, max_len - (end - start))] + [(0, 0)] * len(rest)
            chunk = mx.pad(chunk, pad_width)
        parts.append(chunk)
    return mx.stack(parts, axis=0)  # [B, max_len, *dims]


def pack(
    x: mx.array,  # [B, max_T, *dims]
    cu_seqlens: mx.array,
) -> mx.array:
    batch_size = cu_seqlens.shape[0] - 1
    parts = []
    for i in range(batch_size):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        parts.append(x[i, : end - start])
    packed = mx.concatenate(parts, axis=0)  # [sum_T, *dims]
    return packed[None]  # [1, sum_T, *dims]


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
    batch_size = cu_seqlens.shape[0] - 1
    offsets = [0]
    for i in range(batch_size):
        seqlen = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
        n_chunks = (seqlen + chunk_size - 1) // chunk_size
        offsets.append(offsets[-1] + n_chunks)
    return mx.array(offsets, dtype=mx.int32)
