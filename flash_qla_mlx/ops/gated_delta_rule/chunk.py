# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import mlx.core as mx

from flash_qla_mlx.utils import (
    pack,
    unpack,
    pad_and_reshape,
    fill_last_chunk_of_g,
    prepare_chunk_offsets,
    l2norm,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chunk_local_cumsum(
    g: mx.array,
    chunk_size: int = 64,
    cu_seqlens: mx.array = None,
    reverse: bool = False,
) -> mx.array:
    if cu_seqlens is not None:
        g = unpack(g, cu_seqlens)

    batch_size, num_tokens, num_heads = g.shape
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, H]

    if reverse:
        g = mx.flip(g, axis=2)
        g = mx.cumsum(g, axis=2)
        g = mx.flip(g, axis=2)
    else:
        g = mx.cumsum(g, axis=2)

    g = g.reshape(batch_size, -1, num_heads)[:, :num_tokens]

    if cu_seqlens is not None:
        g = pack(g, cu_seqlens)
    return g


def _kkt_fwd(
    k: mx.array,
    g: mx.array,
    beta: mx.array,
    cu_seqlens: mx.array = None,
    chunk_size: int = 64,
) -> mx.array:
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        g = unpack(g, cu_seqlens)
        beta = unpack(beta, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim = k.shape
    num_v_heads = g.shape[-1]

    if num_k_heads != num_v_heads:
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)      # [B, N, C, H, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)      # [B, N, C, H]
    beta = pad_and_reshape(beta, dim=1, chunk_size=chunk_size)  # [B, N, C, H]

    mask = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=0)
    decay_mask = mx.exp(g[:, :, :, None, :] - g[:, :, None, :, :])  # [B, N, C, C, H]
    decay_mask = mx.where(mask[None, None, :, :, None], mx.zeros_like(decay_mask), decay_mask)

    # attn[b, n, c, h, d] = (k_beta[b,n,c,h,:] . k[b,n,d,h,:]) * decay[b,n,c,d,h]
    attn = mx.einsum(
        "bnchk,bndhk->bnchd", k * beta[:, :, :, :, None], k
    ) * mx.swapaxes(decay_mask, -2, -1)
    attn = attn.reshape(batch_size, -1, num_v_heads, chunk_size)[:, :num_tokens]

    if cu_seqlens is not None:
        attn = pack(attn, cu_seqlens)
    return attn


def _kkt_solve(
    x: mx.array,
    cu_seqlens: mx.array = None,
    chunk_size: int = 64,
) -> mx.array:
    if cu_seqlens is not None:
        x = unpack(x, cu_seqlens)

    batch_size, num_tokens, num_heads, _ = x.shape

    # x: [B, T, H, D] -> [B, N, H, C, D] (negated, lower-tri solve)
    x = -pad_and_reshape(x, dim=1, chunk_size=chunk_size)  # [B, N, C, H, D]
    x = mx.swapaxes(x, 2, 3)  # [B, N, H, C, D]

    for i in range(1, chunk_size):
        row = x[..., i, :i]          # [B, N, H, i]
        sub = x[..., :i, :i]         # [B, N, H, i, i]
        new_val = row + (row[..., None] * sub).sum(axis=-2)
        x[..., i, :i] = new_val

    x = x + mx.eye(chunk_size, dtype=x.dtype)
    x = mx.swapaxes(x, 2, 3)        # [B, N, C, H, D]
    x = x.reshape(batch_size, -1, num_heads, chunk_size)[:, :num_tokens]

    if cu_seqlens is not None:
        x = pack(x, cu_seqlens)
    return x


def _kkt(
    k: mx.array,
    beta: mx.array,
    g: mx.array,
    cu_seqlens: mx.array = None,
    chunk_size: int = 64,
) -> mx.array:
    """Compute A = (I - L)^{-1} where L encodes the gated KKT system."""
    A = _kkt_fwd(k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    A = _kkt_solve(x=A, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    return A


def _w_u_fwd(
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    A: mx.array,
    g: mx.array,
    cu_seqlens: mx.array = None,
) -> tuple:
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        A = unpack(A, cu_seqlens)
        beta = unpack(beta, cu_seqlens)
        g = unpack(g, cu_seqlens)

    batch_size, num_tokens, _, chunk_size = A.shape
    _, _, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape

    if num_k_heads != num_v_heads:
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    k_beta = pad_and_reshape(
        k * (beta * mx.exp(g))[..., None], dim=1, chunk_size=chunk_size
    )  # [B, N, C, Hv, K]
    v_beta = pad_and_reshape(
        v * beta[..., None], dim=1, chunk_size=chunk_size
    )  # [B, N, C, Hv, V]
    A = pad_and_reshape(A, dim=1)  # [B, N, C, Hv, D]

    w = mx.einsum("bnchd,bndhk->bnchk", A, k_beta).reshape(
        batch_size, -1, num_v_heads, head_dim_k
    )[:, :num_tokens]
    u = mx.einsum("bnchd,bndhk->bnchk", A, v_beta).reshape(
        batch_size, -1, num_v_heads, head_dim_v
    )[:, :num_tokens]

    if cu_seqlens is not None:
        w = pack(w, cu_seqlens)
        u = pack(u, cu_seqlens)
    return w, u


def _chunk_gdr_fwd(
    k: mx.array,
    w: mx.array,
    u: mx.array,
    g: mx.array,
    initial_state: mx.array = None,
    cu_seqlens: mx.array = None,
    chunk_size: int = 64,
) -> tuple:
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        w = unpack(w, cu_seqlens)
        u = unpack(u, cu_seqlens)
        g = unpack(g, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = u.shape

    if num_k_heads != num_v_heads:
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    u = pad_and_reshape(u, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    if initial_state is None:
        last_state = mx.zeros(
            (batch_size, num_v_heads, head_dim_k, head_dim_v), dtype=g.dtype
        )
    else:
        last_state = initial_state.astype(g.dtype)

    h_list, vn_list = [], []
    for i in range(k.shape[1]):
        h_list.append(last_state)
        v_new = u[:, i] - mx.einsum("bchk,bhkv->bchv", w[:, i], last_state)
        vn_list.append(v_new)
        last_state = last_state * mx.exp(g[:, i, -1, :])[:, :, None, None]
        last_state = last_state + mx.einsum(
            "bchk,bchv->bhkv",
            k[:, i] * mx.exp(g[:, i, -1:, :, None] - g[:, i, :, :, None]),
            v_new,
        )

    h = mx.stack(h_list, axis=1)
    vn = (
        mx.stack(vn_list, axis=1)
        .reshape(batch_size, -1, num_v_heads, head_dim_v)[:, :num_tokens]
    )

    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        vn = pack(vn, cu_seqlens)
        h = pack(h, chunk_offsets)

    return h, vn, last_state


def _chunk_o_fwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    h: mx.array,
    g: mx.array,
    cu_seqlens: mx.array = None,
    scale: float = None,
    chunk_size: int = 64,
) -> mx.array:
    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        g = unpack(g, cu_seqlens)
        h = unpack(h, chunk_offsets)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape

    if num_k_heads != num_v_heads:
        q = mx.repeat(q, num_v_heads // num_k_heads, axis=2)
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    scale = scale if scale is not None else head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)

    q = q * scale

    mask = mx.triu(
        mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=1
    )
    decay_mask = mx.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = mx.where(mask[None, None, :, :, None], mx.zeros_like(decay_mask), decay_mask)

    attn = mx.einsum("bnchk,bndhk->bncdh", q, k) * decay_mask
    attn_inter = mx.einsum("bnchk,bnhkv->bnchv", q * mx.exp(g)[..., None], h)
    o = attn_inter + mx.einsum("bncdh,bndhv->bnchv", attn, v)

    o = o.reshape(batch_size, -1, num_v_heads, head_dim_v)[:, :num_tokens]
    if cu_seqlens is not None:
        o = pack(o, cu_seqlens)
    return o


# ---------------------------------------------------------------------------
# Backward helpers
# ---------------------------------------------------------------------------


def _chunk_dv_bwd(
    q: mx.array,
    k: mx.array,
    g: mx.array,
    do: mx.array,
    cu_seqlens: mx.array = None,
    scale: float = None,
    chunk_size: int = 64,
) -> mx.array:
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape

    if num_k_heads != num_v_heads:
        q = mx.repeat(q, num_v_heads // num_k_heads, axis=2)
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    scale = scale if scale is not None else head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)

    q = q * scale

    mask = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=1)
    decay_mask = mx.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = mx.where(mask[None, None, :, :, None], mx.zeros_like(decay_mask), decay_mask)

    attn = mx.einsum("bnchk,bndhk->bncdh", q, k) * decay_mask
    dv = mx.einsum("bncdh,bnchv->bndhv", attn, do)

    dv = dv.reshape(batch_size, -1, num_v_heads, head_dim_v)[:, :num_tokens]
    if cu_seqlens is not None:
        dv = pack(dv, cu_seqlens)
    return dv


def _chunk_gdr_bwd(
    q: mx.array,
    k: mx.array,
    w: mx.array,
    g: mx.array,
    do: mx.array,
    dv: mx.array,
    h0: mx.array = None,
    dht: mx.array = None,
    cu_seqlens: mx.array = None,
    scale: float = None,
    chunk_size: int = 64,
) -> tuple:
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        w = unpack(w, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)
        dv = unpack(dv, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape

    if num_k_heads != num_v_heads:
        q = mx.repeat(q, num_v_heads // num_k_heads, axis=2)
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    scale = scale if scale is not None else head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)
    dv = pad_and_reshape(dv, dim=1, chunk_size=chunk_size)
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    q = q * scale

    if dht is None:
        dstate = mx.zeros(
            (batch_size, num_v_heads, head_dim_k, head_dim_v), dtype=g.dtype
        )
    else:
        dstate = dht.astype(g.dtype)

    dstate_inter = mx.einsum(
        "bnchk,bnchv->bnhkv", q * mx.exp(g)[..., None], do
    )

    dh_list = []
    dv_list = list(mx.split(dv, dv.shape[1], axis=1))  # list of [B,1,C,Hv,V]
    for i in reversed(range(k.shape[1])):
        dh_list.insert(0, dstate)
        dv_i = dv_list[i][:, 0]  # [B, C, Hv, V]
        dv_i = dv_i + mx.einsum(
            "bchk,bhkv->bchv",
            k[:, i] * mx.exp(g[:, i, -1:, :, None] - g[:, i, :, :, None]),
            dstate,
        )
        dv_list[i] = dv_i
        dstate = dstate * mx.exp(g[:, i, -1, :])[:, :, None, None]
        dstate = (
            dstate
            + dstate_inter[:, i]
            - mx.einsum("bchk,bchv->bhkv", w[:, i], dv_i)
        )

    dh = mx.stack(dh_list, axis=1)

    dh0 = None if h0 is None else dstate
    dv = mx.stack(dv_list, axis=1).reshape(
        batch_size, -1, num_v_heads, head_dim_v
    )[:, :num_tokens]

    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        dv = pack(dv, cu_seqlens)
        dh = pack(dh, chunk_offsets)
    return dh, dh0, dv


def _chunk_dqkwg_bwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    w: mx.array,
    g: mx.array,
    h: mx.array,
    dv: mx.array,
    do: mx.array,
    dh: mx.array,
    cu_seqlens: mx.array = None,
    scale: float = None,
    chunk_size: int = 64,
) -> tuple:
    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        w = unpack(w, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)
        dv = unpack(dv, cu_seqlens)
        h = unpack(h, chunk_offsets)
        dh = unpack(dh, chunk_offsets)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape

    if num_k_heads != num_v_heads:
        q = mx.repeat(q, num_v_heads // num_k_heads, axis=2)
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    scale = scale if scale is not None else head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)
    dv = pad_and_reshape(dv, dim=1, chunk_size=chunk_size)
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    mask = mx.triu(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=1)
    decay_mask = mx.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = mx.where(mask[None, None, :, :, None], mx.zeros_like(decay_mask), decay_mask)

    dg_last = (h * dh).sum(axis=-1).sum(axis=-1)  # [B, N, Hv]
    ds = mx.einsum("bnchv,bndhv->bncdh", do, v)
    dq = mx.einsum("bnchv,bnhkv->bnchk", do, h)
    dk = mx.einsum("bnchv,bnhkv->bnchk", v, dh)
    dw = -mx.einsum("bnchv,bnhkv->bnchk", dv, h)

    g_last = g[:, :, -1]
    dg_last = dg_last * mx.exp(g_last)
    dq = dq * mx.exp(g)[..., None] * scale
    dg = (q * dq).sum(axis=-1)
    dk = dk * mx.exp(g_last[:, :, None, :, None] - g)[..., None]
    dg = dg - (k * dk).sum(axis=-1)
    dg_last = dg_last + (k * dk).sum(axis=-1).sum(axis=-2)
    ds = ds * decay_mask * scale
    ds2 = ds * mx.einsum("bnchk,bndhk->bncdh", q, k)
    dg = dg + ds2.sum(axis=-2)
    dg = dg - ds2.sum(axis=-3)
    dq = dq + mx.einsum("bncdh,bndhk->bnchk", ds, k)
    dk = dk + mx.einsum("bncdh,bnchk->bndhk", ds, q)
    dg[:, :, -1] = dg[:, :, -1] + dg_last

    dg = fill_last_chunk_of_g(
        dg, num_tokens, cu_seqlens, chunk_size=chunk_size, reverse=True
    )
    dq = dq.reshape(batch_size, -1, num_v_heads, head_dim_k)[:, :num_tokens]
    dk = dk.reshape(batch_size, -1, num_v_heads, head_dim_k)[:, :num_tokens]
    dw = dw.reshape(batch_size, -1, num_v_heads, head_dim_k)[:, :num_tokens]
    dg = dg.reshape(batch_size, -1, num_v_heads)[:, :num_tokens]

    if cu_seqlens is not None:
        dq = pack(dq, cu_seqlens)
        dk = pack(dk, cu_seqlens)
        dw = pack(dw, cu_seqlens)
        dg = pack(dg, cu_seqlens)
    return dq, dk, dw, dg


def _chunk_wy_bwd(
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    A: mx.array,
    g: mx.array,
    dw: mx.array,
    du: mx.array,
    dk1: mx.array,
    dg1: mx.array,
    cu_seqlens: mx.array = None,
    chunk_size: int = 64,
) -> tuple:
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        beta = unpack(beta, cu_seqlens)
        A = unpack(A, cu_seqlens)
        g = unpack(g, cu_seqlens)
        dw = unpack(dw, cu_seqlens)
        du = unpack(du, cu_seqlens)
        dk1 = unpack(dk1, cu_seqlens)
        dg1 = unpack(dg1, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape
    chunk_size_A = A.shape[-1]

    if num_k_heads != num_v_heads:
        k = mx.repeat(k, num_v_heads // num_k_heads, axis=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size_A)
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size_A)
    beta = pad_and_reshape(beta, dim=1, chunk_size=chunk_size_A)
    A = pad_and_reshape(A, dim=1)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size_A)
    dw = pad_and_reshape(dw, dim=1, chunk_size=chunk_size_A)
    du = pad_and_reshape(du, dim=1, chunk_size=chunk_size_A)
    dk1 = pad_and_reshape(dk1, dim=1, chunk_size=chunk_size_A)
    dg1 = pad_and_reshape(dg1, dim=1, chunk_size=chunk_size_A)

    dA = mx.einsum("bnchk,bndhk->bnchd", dw, k * (beta * mx.exp(g))[..., None])
    dk_beta_g = mx.einsum("bnchd,bnchk->bndhk", A, dw)
    dk = dk_beta_g * (beta * mx.exp(g))[..., None]
    db = (dk_beta_g * k * mx.exp(g)[..., None]).sum(axis=-1)
    dg = (dk_beta_g * k * (mx.exp(g) * beta)[..., None]).sum(axis=-1)

    dA = dA + mx.einsum("bnchv,bndhv->bnchd", du, v * beta[..., None])
    dv_beta = mx.einsum("bnchd,bnchv->bndhv", A, du)
    dv = dv_beta * beta[..., None]
    db = db + (dv_beta * v).sum(axis=-1)

    mask = mx.triu(mx.ones((chunk_size_A, chunk_size_A), dtype=mx.bool_), k=0)
    decay_mask = mx.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = mx.where(
        mask[None, None, :, :, None], mx.zeros_like(decay_mask), decay_mask
    )
    decay_mask = mx.swapaxes(decay_mask, -2, -1)

    dA = mx.where(mask[None, None, :, None, :], mx.zeros_like(dA), dA)
    dA = mx.einsum("bndhc,bndhe->bnche", A, dA)
    dA = mx.einsum("bnchd,bnehd->bnche", dA, A)
    dA = -dA * decay_mask

    A_kkt = mx.einsum("bnchk,bndhk->bnchd", k * beta[..., None], k)
    dk_beta = mx.einsum("bnchd,bndhk->bnchk", dA, k)
    db = db + (dk_beta * k).sum(axis=-1)
    dk = dk + mx.einsum("bnchd,bnchk->bndhk", dA, k * beta[..., None])
    dk = dk + dk_beta * beta[..., None]
    dk = dk + dk1

    dg = dg + (dA * A_kkt).sum(axis=-1) - mx.swapaxes(
        (dA * A_kkt).sum(axis=-3), -1, -2
    )
    dg = dg + dg1

    dk = dk.reshape(batch_size, -1, num_v_heads, head_dim_k)[:, :num_tokens]
    dv = dv.reshape(batch_size, -1, num_v_heads, head_dim_k)[:, :num_tokens]
    db = db.reshape(batch_size, -1, num_v_heads)[:, :num_tokens]
    dg = dg.reshape(batch_size, -1, num_v_heads)[:, :num_tokens]

    if cu_seqlens is not None:
        dk = pack(dk, cu_seqlens)
        dv = pack(dv, cu_seqlens)
        db = pack(db, cu_seqlens)
        dg = pack(dg, cu_seqlens)
    return dk, dv, db, dg


def _group_reduce_vector(buffer: mx.array, Hg: int) -> mx.array:
    batch_size, num_tokens, H, K = buffer.shape
    return buffer.reshape(batch_size, num_tokens, Hg, H // Hg, K).sum(axis=3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_fwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float = None,
    initial_state: mx.array = None,
    cu_seqlens: mx.array = None,
    output_final_state: bool = True,
    output_h: bool = False,
) -> tuple:
    """
    Forward pass of the Gated Delta Rule.

    Args:
        q:              [B, T, Hk, K]
        k:              [B, T, Hk, K]
        v:              [B, T, Hv, V]
        g:              [B, T, Hv]  (log-decay, negative values)
        beta:           [B, T, Hv]
        scale:          Optional softmax scale (defaults to K**-0.5)
        initial_state:  Optional [B, Hv, K, V]
        cu_seqlens:     Optional [S+1] int32, for variable-length inputs
        output_final_state: Whether to return the final state
        output_h:       Whether to return the chunk-level states h

    Returns:
        (g_cumsum, A, o, h, final_state)
        h and final_state may be None if not requested.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    chunk_size = 64

    g_cumsum = _chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    A = _kkt(k=k, beta=beta, g=g_cumsum, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    w, u = _w_u_fwd(k=k, v=v, beta=beta, A=A, g=g_cumsum, cu_seqlens=cu_seqlens)
    h, vn, final_state = _chunk_gdr_fwd(
        k=k, w=w, u=u, g=g_cumsum,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    )
    o = _chunk_o_fwd(
        q=q, k=k, v=vn, h=h, g=g_cumsum,
        cu_seqlens=cu_seqlens, scale=scale, chunk_size=chunk_size,
    )

    return (
        g_cumsum,
        A,
        o,
        h if output_h else None,
        final_state if output_final_state else None,
    )


def chunk_gated_delta_rule_bwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    A: mx.array,
    do: mx.array,
    dht: mx.array = None,
    scale: float = None,
    initial_state: mx.array = None,
    cu_seqlens: mx.array = None,
) -> tuple:
    """
    Backward pass of the Gated Delta Rule.

    Args:
        q, k, v, g, beta, A:  Saved from forward (g is the cumsum version)
        do:           Gradient of the output [B, T, Hv, V]
        dht:          Gradient of the final state [B, Hv, K, V] or None
        scale, initial_state, cu_seqlens: same as forward

    Returns:
        (dq, dk, dv, db, dg, dh0)
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    chunk_size = 64

    w, u = _w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu_seqlens)
    h, vn, _ = _chunk_gdr_fwd(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    )
    dv = _chunk_dv_bwd(
        q=q, k=k, g=g, do=do, cu_seqlens=cu_seqlens, scale=scale, chunk_size=chunk_size
    )
    dh, dh0, dv = _chunk_gdr_bwd(
        q=q, k=k, w=w, g=g, do=do, dv=dv,
        h0=initial_state, dht=dht,
        cu_seqlens=cu_seqlens, scale=scale, chunk_size=chunk_size,
    )
    dq, dk1, dw, dg1 = _chunk_dqkwg_bwd(
        q=q, k=k, v=vn, w=w, g=g, h=h,
        dv=dv, do=do, dh=dh,
        cu_seqlens=cu_seqlens, scale=scale, chunk_size=chunk_size,
    )
    dk, dv_out, db, dg = _chunk_wy_bwd(
        k=k, v=v, beta=beta, A=A, g=g,
        dw=dw, du=dv, dk1=dk1, dg1=dg1,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    )

    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq = _group_reduce_vector(dq, Hg)
        dk = _group_reduce_vector(dk, Hg)

    dg = _chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv_out, db, dg, dh0


def chunk_gated_delta_rule(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float = None,
    initial_state: mx.array = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: mx.array = None,
    head_first: bool = False,
) -> tuple:
    """
    Gated Delta Rule: end-to-end forward with MLX-native gradient support.

    Args:
        q:              [B, T, Hk, K]
        k:              [B, T, Hk, K]
        v:              [B, T, Hv, V]
        g:              [B, T, Hv]
        beta:           [B, T, Hv]
        scale:          Optional softmax scale
        initial_state:  Optional [B, Hv, K, V]
        output_final_state: Return final recurrent state
        use_qk_l2norm_in_kernel: L2-normalize q and k before computation
        cu_seqlens:     Optional variable-length sequence boundaries
        head_first:     Not supported (must be False)

    Returns:
        (o, final_state)  — final_state is None if output_final_state=False
    """
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0, (
        "num_v_heads must be divisible by num_k_heads."
    )

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"Batch size must be 1 when using cu_seqlens, got {q.shape[0]}."
        )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    # Define custom forward/backward for correct gradient computation.
    # We use closures to capture scale, cu_seqlens, and the presence of h0.
    _scale = scale
    _cu_seqlens = cu_seqlens

    if initial_state is not None:
        @mx.custom_function
        def _fn(q, k, v, g, beta, h0):
            g_out, A, o, _, fs = chunk_gated_delta_rule_fwd(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=_scale, initial_state=h0,
                cu_seqlens=_cu_seqlens,
                output_final_state=True, output_h=False,
            )
            fs_safe = fs if fs is not None else mx.zeros_like(h0)
            return o, fs_safe, g_out, A

        @_fn.vjp
        def _fn_vjp(primals, cotangents, outputs):
            q, k, v, g, beta, h0 = primals
            do, dfs, _, _ = cotangents
            _, _, g_out, A = outputs
            dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
                q=q, k=k, v=v, g=g_out, beta=beta, A=A,
                do=do, dht=dfs, scale=_scale,
                initial_state=h0, cu_seqlens=_cu_seqlens,
            )
            return dq, dk, dv, dg, db, dh0

        o, fs, _, _ = _fn(q, k, v, g, beta, initial_state)
    else:
        @mx.custom_function
        def _fn(q, k, v, g, beta):
            g_out, A, o, _, fs = chunk_gated_delta_rule_fwd(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=_scale, initial_state=None,
                cu_seqlens=_cu_seqlens,
                output_final_state=output_final_state, output_h=False,
            )
            _dummy = mx.zeros((1,), dtype=q.dtype)
            return o, _dummy, g_out, A

        @_fn.vjp
        def _fn_vjp(primals, cotangents, outputs):
            q, k, v, g, beta = primals
            do, _, _, _ = cotangents
            _, _, g_out, A = outputs
            dq, dk, dv, db, dg, _ = chunk_gated_delta_rule_bwd(
                q=q, k=k, v=v, g=g_out, beta=beta, A=A,
                do=do, dht=None, scale=_scale,
                initial_state=None, cu_seqlens=_cu_seqlens,
            )
            return dq, dk, dv, dg, db

        o, _, _, _ = _fn(q, k, v, g, beta)
        fs = None

    return o, fs if output_final_state else None
