# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch

from flash_qla.utils import (
    pad_and_reshape,
    pack,
    unpack,
    fill_last_chunk_of_g,
    prepare_chunk_offsets,
)


def torch_l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    raw_shape = x.shape
    x = x.view(-1, raw_shape[-1])
    sum_sq = (x * x).sum(dim=dim, keepdim=True) + eps
    rstd = torch.rsqrt(sum_sq)
    y = (x * rstd).to(x.dtype)
    y = y.view(raw_shape)
    rstd = rstd.view(*raw_shape[:-1])
    return y, rstd


def torch_l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    assert dim == -1 or dim == len(dy.shape) - 1
    assert y.stride(-1) == 1
    assert dy.stride(-1) == 1
    raw_shape = dy.shape
    y = y.view(-1, raw_shape[-1])
    dy = dy.view(-1, raw_shape[-1])
    rstd = rstd.view(-1, 1)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    dx = dy * rstd - dot * y * rstd
    dx = dx.view(raw_shape)
    return dx


def torch_cumsum(
    x: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
    reverse: bool = False,
):
    if cu_seqlens is not None:
        x = unpack(x, cu_seqlens)

    raw_shape = x.shape

    x = pad_and_reshape(x, dim=1, chunk_size=chunk_size)

    if reverse:
        x = torch.flip(x, dims=(2,))
        x = x.cumsum(dim=2)
        x = torch.flip(x, dims=(2,))
    else:
        x = x.cumsum(dim=2)
    x = x.reshape(raw_shape[0], -1, *raw_shape[2:])
    x = x[:, :raw_shape[1]]

    if cu_seqlens is not None:
        x = pack(x, cu_seqlens)
    return x


def torch_kkt_fwd(
    k: torch.Tensor,  # [B, T, Hk, K]
    g: torch.Tensor,  # [B, T, Hv]
    beta: torch.Tensor,  # [B, T, Hv]
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        g = unpack(g, cu_seqlens)
        beta = unpack(beta, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim = k.shape
    num_v_heads = g.shape[-1]

    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, H, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, H]
    beta = pad_and_reshape(beta, dim=1, chunk_size=chunk_size)  # [B, N, C, H]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device)
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)
    # decay_mask = torch.where(mask[None, None, :, :, None], decay_mask, 0.0)
    attn = torch.einsum(
        "bnchk, bndhk -> bnchd", k * beta.unsqueeze(-1), k
    ) * decay_mask.swapaxes(-2, -1)  # [B, N, C, H, D]
    attn = attn.reshape(batch_size, -1, num_v_heads, chunk_size)[:, :num_tokens]

    if cu_seqlens is not None:
        attn = pack(attn, cu_seqlens)
    return attn


def torch_solve(
    x: torch.Tensor,  # [B, T, H, D]
    cu_seqlens: torch.Tensor = None,
):
    if cu_seqlens is not None:
        x = unpack(x, cu_seqlens)

    batch_size, num_tokens, num_heads, chunk_size = x.shape

    x = -pad_and_reshape(x, dim=1, chunk_size=chunk_size).swapaxes(
        2, 3
    )  # [B, N, H, C, D]

    for i in range(1, chunk_size):
        row = x[..., i, :i].clone()
        sub = x[..., :i, :i].clone()
        x[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    x += torch.eye(chunk_size, dtype=x.dtype, device=x.device)
    x = x.swapaxes(2, 3).reshape((batch_size, -1, num_heads, chunk_size))[
        :, :num_tokens
    ]

    if cu_seqlens is not None:
        x = pack(x, cu_seqlens)
    return x


def torch_w_u_fwd(
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, V]
    g: torch.Tensor,  # [B, T, Hv]
    beta: torch.Tensor,  # [B, T, Hv]
    A: torch.Tensor,  # [B, T, Hv, D]
    cu_seqlens: torch.Tensor = None,
):
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
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k_beta = pad_and_reshape(
        k * beta.unsqueeze(-1) * g.exp().unsqueeze(-1), dim=1, chunk_size=chunk_size
    )  # [B, N, C, Hv, K]
    v_beta = pad_and_reshape(
        v * beta.unsqueeze(-1), dim=1, chunk_size=chunk_size
    )  # [B, N, C, Hv, V]
    A = pad_and_reshape(A, dim=1,chunk_size=chunk_size)

    w = torch.einsum("bnchd, bndhk -> bnchk", A, k_beta).reshape(
        (batch_size, -1, num_v_heads, head_dim_k)
    )[:, :num_tokens]
    u = torch.einsum("bnchd, bndhk -> bnchk", A, v_beta).reshape(
        (batch_size, -1, num_v_heads, head_dim_v)
    )[:, :num_tokens]

    if cu_seqlens is not None:
        w = pack(w, cu_seqlens)
        u = pack(u, cu_seqlens)
    return w, u


def torch_chunk_gdr_fwd(
    k: torch.Tensor,  # [B, T, Hk, K]
    w: torch.Tensor,  # [B, T, Hv, K]
    u: torch.Tensor,  # [B, T, Hv, V]
    g: torch.Tensor,  # [B, T, Hv]
    initial_state: torch.Tensor = None,  # [B, Hv, K, V]
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        w = unpack(w, cu_seqlens)
        u = unpack(u, cu_seqlens)
        g = unpack(g, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = u.shape

    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    u = pad_and_reshape(u, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    if initial_state is None:
        last_state = torch.zeros(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            dtype=g.dtype,
            device=g.device,
        )
    else:
        last_state = initial_state.to(g.dtype, copy=True)

    h, vn = [], []
    for i in range(k.shape[1]):
        h.append(last_state)
        v_new = u[:, i] - torch.einsum("bchk, bhkv -> bchv", w[:, i], last_state)
        vn.append(v_new)
        last_state = last_state * g[:, i, -1, :, None, None].exp()
        last_state = last_state + torch.einsum(
            "bchk, bchv -> bhkv",
            k[:, i] * (g[:, i, -1:, :, None] - g[:, i, :, :, None]).exp(),
            v_new,
        )
    h = torch.stack(h, dim=1).contiguous()
    vn = (
        torch.stack(vn, dim=1)
        .reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
        .contiguous()
    )

    if cu_seqlens is not None:
        vn = pack(vn, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        h = pack(h, chunk_offsets)

    return h, vn, last_state


def torch_chunk_o_fwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, K]
    h: torch.Tensor,  # [B, N, Hv, K, V]
    g: torch.Tensor,  # [B, T, Hv]
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        g = unpack(g, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        h = unpack(h, chunk_offsets)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape

    if num_k_heads != num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    scale = scale or head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    q = q * scale

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device),
        diagonal=1,
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(
        mask[None, None, :, :, None], 0.0
    )  # [B, N, C, D, Hv]

    attn = torch.einsum("bnchk, bndhk -> bncdh", q, k) * decay_mask
    attn_inter = torch.einsum("bnchk, bnhkv -> bnchv", q * g.exp().unsqueeze(-1), h)
    o = attn_inter + torch.einsum("bncdh, bndhv -> bnchv", attn, v)

    o = o.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    if cu_seqlens is not None:
        o = pack(o, cu_seqlens)
    return o


def torch_chunk_dv_bwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    g: torch.Tensor,  # [B, T, Hv]
    do: torch.Tensor,  # [B, T, Hv, V]
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape

    if num_k_heads != num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    scale = scale or head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    q = q * scale

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device),
        diagonal=1,
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(
        mask[None, None, :, :, None], 0.0
    )  # [B, N, C, D, Hv]

    attn = torch.einsum("bnchk, bndhk -> bncdh", q, k) * decay_mask
    dv = torch.einsum("bncdh, bnchv -> bndhv", attn, do)

    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    if cu_seqlens is not None:
        dv = pack(dv, cu_seqlens)
    return dv


def torch_chunk_gdr_bwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    w: torch.Tensor,  # [B, T, Hv, K]
    g: torch.Tensor,  # [B, T, Hv]
    do: torch.Tensor,  # [B, T, Hv, V]
    dv: torch.Tensor,  # [B, T, Hv, V]
    h0: torch.Tensor = None,  # [B, Hv, K, V]
    dht: torch.Tensor = None,  # [B, Hv, K, V]
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
):
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
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    scale = scale or head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    dv = pad_and_reshape(dv, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    q = q * scale

    if dht is None:
        dstate = torch.zeros(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            dtype=g.dtype,
            device=g.device,
        )
    else:
        dstate = dht.to(g.dtype, copy=True)
    dstate_inter = torch.einsum("bnchk, bnchv -> bnhkv", q * g.exp().unsqueeze(-1), do)

    dh = []
    for i in reversed(range(k.shape[1])):
        dh.insert(0, dstate)
        dv[:, i] += torch.einsum(
            "bchk, bhkv -> bchv",
            k[:, i] * (g[:, i, -1:, :, None] - g[:, i, :, :, None]).exp(),
            dstate,
        )
        dstate = dstate * g[:, i, -1, :, None, None].exp()
        dstate = (
            dstate
            + dstate_inter[:, i]
            - torch.einsum("bchk, bchv -> bhkv", w[:, i], dv[:, i])
        )
    dh = torch.stack(dh, dim=1).contiguous()

    dh0 = None if h0 is None else dstate
    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]
    if cu_seqlens is not None:
        dv = pack(dv, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        dh = pack(dh, chunk_offsets)
    return dh, dh0, dv


def torch_chunk_dqkwg_bwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, V]
    w: torch.Tensor,  # [B, T, Hv, K]
    g: torch.Tensor,  # [B, T, Hv]
    h: torch.Tensor,  # [B, N, Hv, K, V]
    dv: torch.Tensor,  # [B, T, Hv, V]
    do: torch.Tensor,  # [B, T, Hv, V]
    dh: torch.Tensor,  # [B, N, Hv, K, V]
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        w = unpack(w, cu_seqlens)
        g = unpack(g, cu_seqlens)
        do = unpack(do, cu_seqlens)
        dv = unpack(dv, cu_seqlens)
        chunk_offsets, _ = prepare_chunk_offsets(cu_seqlens, chunk_size)
        h = unpack(h, chunk_offsets)
        dh = unpack(dh, chunk_offsets)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = do.shape

    if num_k_heads != num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    scale = scale or head_dim_k ** (-0.5)

    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    dv = pad_and_reshape(dv, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device),
        diagonal=1,
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(
        mask[None, None, :, :, None], 0.0
    )  # [B, N, C, D, Hv]

    dg_last = (h * dh).sum(dim=-1).sum(dim=-1)  # [B, N, Hv]
    ds = torch.einsum("bnchv, bndhv -> bncdh", do, v)
    dq = torch.einsum("bnchv, bnhkv -> bnchk", do, h)
    dk = torch.einsum("bnchv, bnhkv -> bnchk", v, dh)
    dw = -torch.einsum("bnchv, bnhkv -> bnchk", dv, h)

    g_last = g[:, :, -1]
    dg_last *= g_last.exp()
    dq = dq * g.unsqueeze(-1).exp() * scale
    dg = (q * dq).sum(dim=-1)  # [B, N, C, Hv]
    dk = dk * (g_last.unsqueeze(-2) - g).unsqueeze(-1).exp()
    dg -= (k * dk).sum(dim=-1)
    dg_last += (k * dk).sum(dim=-1).sum(dim=-2)
    ds *= decay_mask * scale
    ds2 = ds * torch.einsum("bnchk, bndhk -> bncdh", q, k)
    dg += ds2.sum(dim=-2)
    dg -= ds2.sum(dim=-3)
    dq += torch.einsum("bncdh, bndhk -> bnchk", ds, k)
    dk += torch.einsum("bncdh, bnchk -> bndhk", ds, q)
    dg[:, :, -1] += dg_last

    dg = fill_last_chunk_of_g(
        dg, num_tokens, cu_seqlens, chunk_size=chunk_size, reverse=True
    )
    dq = dq.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dk = dk.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dw = dw.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dg = dg.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    if cu_seqlens is not None:
        dq = pack(dq, cu_seqlens)
        dk = pack(dk, cu_seqlens)
        dw = pack(dw, cu_seqlens)
        dg = pack(dg, cu_seqlens)
    return dq, dk, dw, dg


def torch_chunk_wy_bwd(
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, V]
    beta: torch.Tensor,  # [B, T, Hv]
    A: torch.Tensor,  # [B, T, Hv, D]
    g: torch.Tensor,  # [B, T, Hv]
    dw: torch.Tensor,  # [B, T, Hv, K]
    du: torch.Tensor,  # [B, T, Hv, V]
    dk1: torch.Tensor,  # [B, T, Hv, K]
    dg1: torch.Tensor,  # [B, T, Hv]
    cu_seqlens: torch.Tensor = None,
):
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
    chunk_size = A.shape[-1]

    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    beta = pad_and_reshape(beta, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    A = pad_and_reshape(A, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, D]
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    dw = pad_and_reshape(dw, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    du = pad_and_reshape(du, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, V]
    dk1 = pad_and_reshape(dk1, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv, K]
    dg1 = pad_and_reshape(dg1, dim=1, chunk_size=chunk_size)  # [B, N, C, Hv]
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    dA = torch.einsum("bnchk, bndhk -> bnchd", dw, k * (beta * g.exp()).unsqueeze(-1))
    dk_beta_g = torch.einsum("bnchd, bnchk -> bndhk", A, dw)
    dk = dk_beta_g * (beta * g.exp()).unsqueeze(-1)
    db = (dk_beta_g * k * g.exp().unsqueeze(-1)).sum(dim=-1)
    dg = (dk_beta_g * k * (g.exp() * beta).unsqueeze(-1)).sum(dim=-1)

    dA += torch.einsum("bnchv, bndhv -> bnchd", du, v * beta.unsqueeze(-1))
    dv_beta = torch.einsum("bnchd, bnchv -> bndhv", A, du)
    dv = dv_beta * beta.unsqueeze(-1)
    db += (dv_beta * v).sum(dim=-1)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device)
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0).swapaxes(
        -2, -1
    )
    dA = dA.masked_fill(mask[None, None, :, None, :], 0.0)
    dA = torch.einsum("bndhc, bndhe -> bnche", A, dA)
    dA = torch.einsum("bnchd, bnehd -> bnche", dA, A)
    dA = -dA * decay_mask

    A = torch.einsum("bnchk, bndhk -> bnchd", k * beta.unsqueeze(-1), k)
    dk_beta = torch.einsum("bnchd, bndhk -> bnchk", dA, k)
    db += (dk_beta * k).sum(dim=-1)
    dk += torch.einsum("bnchd, bnchk -> bndhk", dA, k * beta.unsqueeze(-1))
    dk += dk_beta * beta.unsqueeze(-1)
    dk += dk1

    dg += (dA * A).sum(dim=-1) - (dA * A).sum(dim=-3).swapaxes(-1, -2)
    dg += dg1

    # TODO: NOTE: GVA
    dk = dk.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    dv = dv.reshape((batch_size, -1, num_v_heads, head_dim_k))[:, :num_tokens]
    db = db.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    dg = dg.reshape((batch_size, -1, num_v_heads))[:, :num_tokens]
    if cu_seqlens is not None:
        dk = pack(dk, cu_seqlens)
        dv = pack(dv, cu_seqlens)
        db = pack(db, cu_seqlens)
        dg = pack(dg, cu_seqlens)
    return dk, dv, db, dg


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,  # [B, T, Hk, K]
    k: torch.Tensor,  # [B, T, Hk, K]
    v: torch.Tensor,  # [B, T, Hv, K]
    g: torch.Tensor,  # [B, T, Hv]
    beta: torch.Tensor,  # [B, T, Hv]
    cu_seqlens: torch.Tensor = None,
    initial_state: torch.Tensor = None,
    scale: float = None,
    chunk_size: int = 64,
):
    scale = scale or q.shape[-1] ** (-0.5)
    g = torch_cumsum(
        x=g,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    A = torch_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    A = torch_solve(
        x=A,
        cu_seqlens=cu_seqlens,
    )
    w, u = torch_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, vn, final_state = torch_chunk_gdr_fwd(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = torch_chunk_o_fwd(
        q=q,
        k=k,
        v=vn,
        h=h,
        g=g,
        cu_seqlens=cu_seqlens,
        scale=scale,
        chunk_size=chunk_size,
    )
    return g, o, A, h, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 64,
):
    w, u = torch_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, vn, _ = torch_chunk_gdr_fwd(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dv = torch_chunk_dv_bwd(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = torch_chunk_gdr_bwd(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk1, dw, dg1 = torch_chunk_dqkwg_bwd(
        q=q,
        k=k,
        v=vn,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk, dv, db, dg = torch_chunk_wy_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        dk1=dk1,
        dg1=dg1,
        cu_seqlens=cu_seqlens,
    )
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        B, T, _, K = dq.shape
        dq = torch.sum(dq.reshape(B, T, Hg, -1, K), dim=3)
        dk = torch.sum(dk.reshape(B, T, Hg, -1, K), dim=3)
    dg = torch_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = torch_l2norm_fwd(q)
            k, k_rstd = torch_l2norm_fwd(k)

        g, o, A, _, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q, k, q_rstd, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, k, q_rstd, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = torch_l2norm_bwd(q, q_rstd, dq)
            dk = torch_l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )

    return o, final_state
