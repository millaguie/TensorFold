"""The Qwen forward on RDNA: projections, attention and linear-attention layers, MLP or experts, caches."""

from __future__ import annotations

import torch

from tensorfold.rocm.model.qwen_math import (
    SPAN,
    DevicePos,
    Packed,
    Spec,
    apply_rope,
    causal_attend,
    causal_conv,
    gated_delta,
    gather_rows,
    normalize_qk,
    rms_norm,
)


def _project(x: torch.Tensor, packed: Packed, linear) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    y = linear(flat, packed)
    if y.dtype != x.dtype and not packed.partial:     # a rank's share stays fp32 until the ranks are summed
        y = y.to(dtype=x.dtype)
    return y.reshape(*x.shape[:-1], -1)


def _project_pair(x: torch.Tensor, first: Packed, second: Packed, linear):
    """Two projections of one activation. Falls back to two solo launches."""

    owner = getattr(linear, "__self__", None)
    pair = getattr(owner, "linear_pair", None) if owner is not None else None
    if pair is None:
        return _project(x, first, linear), _project(x, second, linear)
    left, right = pair(x, first, second)
    if left.dtype != x.dtype:
        left = left.to(dtype=x.dtype)
    if right.dtype != x.dtype:
        right = right.to(dtype=x.dtype)
    return left.reshape(*x.shape[:-1], -1), right.reshape(*x.shape[:-1], -1)


def _project_group(x: torch.Tensor, packeds: tuple, linear):
    """One launch for several projections of a short activation. None keeps the solo or pair path."""

    owner = getattr(linear, "__self__", None)
    group = getattr(owner, "linear_group", None) if owner is not None else None
    if group is None:
        return None
    outs = group(x, packeds)
    if outs is None:
        return None
    return tuple(y.reshape(*x.shape[:-1], -1) if y.dtype == x.dtype else y.to(dtype=x.dtype).reshape(*x.shape[:-1], -1)
                 for y in outs)


def _add_residual(x: torch.Tensor, y: torch.Tensor) -> None:
    """``x += y`` in place; a rank's fp32 sum is added to a widened residual and rounded once. In place is one add
    kernel where ``x[...] = x + y`` added and then copied back, with the same rounding."""

    if y.dtype == torch.float32 and x.dtype != torch.float32:
        x.copy_(x.float() + y)
    else:
        x.add_(y)


def forward_hidden(model, tokens: torch.Tensor, caches: list | None, linear, pos0: int,
                   act_dtype: torch.dtype | None = None, *, exact_short: bool = False, reduce=None,
                   at: DevicePos | None = None):
    """One prefill or decode step; ``reduce`` sums tp shares, ``at`` reads the position on the device."""

    spec = model.spec
    x = gather_rows(model.embed, tokens, dtype=act_dtype)
    if x.device != tokens.device:
        x = x.to(device=tokens.device)
    if not x.is_contiguous():
        x = x.contiguous()
    fresh = caches is None
    new_caches = []
    length = x.shape[1]
    for index, layer in enumerate(model.layers):
        cache = None if fresh else caches[index]
        # One span of the residual. The whole prompt is not a second fp32 copy.
        for start in range(0, length, SPAN):
            stop = min(length, start + SPAN)
            normed = rms_norm(x[:, start:stop], layer.input_norm, spec.eps)
            if spec.full(index):
                y, cache = _attention(spec, layer, normed, cache, linear, pos0 + start, exact_short, at)
            else:
                y, cache = _linear_attn(spec, layer, normed, cache, linear, exact_short, at is not None)
            if reduce is not None:
                y = reduce(y)
            _add_residual(x[:, start:stop], y)
            y = _mlp(spec, layer, rms_norm(x[:, start:stop], layer.post_norm, spec.eps), linear)
            if reduce is not None:
                y = reduce(y)
            _add_residual(x[:, start:stop], y)
        new_caches.append(cache)
    return rms_norm(x, model.final_norm, spec.eps), new_caches


def _mlp(spec: Spec, layer, x: torch.Tensor, linear) -> torch.Tensor:
    if layer.moe is not None:
        return _moe_mlp(layer.moe, x)
    # Gate and up are the wide tensors. A long prefill computes them a chunk at a time.
    batch, length, hidden = x.shape
    flat = x.reshape(-1, hidden)
    if flat.shape[0] <= SPAN:
        grouped = _project_group(x, (layer.gate, layer.up), linear)
        if grouped is None:
            gate, up = _project_pair(x, layer.gate, layer.up, linear)
        else:
            gate, up = grouped
        return _project(torch.nn.functional.silu(gate) * up, layer.down, linear)
    out = torch.empty(flat.shape, dtype=torch.float32 if layer.down.partial else flat.dtype, device=flat.device)
    for start in range(0, flat.shape[0], SPAN):
        stop = min(start + SPAN, flat.shape[0])
        piece = flat[start:stop]
        gate = _project(piece, layer.gate, linear)
        up = _project(piece, layer.up, linear)
        out[start:stop] = _project(torch.nn.functional.silu(gate) * up, layer.down, linear)
    return out.view(batch, length, hidden)


def _moe_mlp(routed, x: torch.Tensor) -> torch.Tensor:
    """Routed experts plus the shared one over the flattened rows, in the activation dtype."""

    from tensorfold.rocm.model.moe import run

    batch, length, hidden = x.shape
    y = run(x.reshape(-1, hidden), routed, prefill=length > 1)
    if routed.partial:                     # a tp rank's fp32 share: the all-reduce rounds once, after the sum
        return y.view(batch, length, hidden)
    return y.view(batch, length, hidden).to(dtype=x.dtype)


def _gated_norm(y: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> torch.Tensor:
    """``rms_norm(y) * silu(z)`` in ``dtype``; on the device one kernel with these ops' roundings."""

    width = y.shape[-1]
    if (y.is_cuda and y.dtype == torch.float32 and z.dtype == torch.bfloat16 and dtype == torch.bfloat16
            and width <= 512 and weight is not None and weight.numel() == width):
        from tensorfold.rocm.kernels.act import gated_rms

        out = gated_rms(y.reshape(-1, width).contiguous(), weight.reshape(width).float().contiguous(),
                        z.reshape(-1, width).contiguous(), eps)
        if out is not None:
            return out.view(z.shape)
    # silu(z) widened first: the same product as the promoting multiply, on the same-dtype kernel.
    y = rms_norm(y, weight, eps) * torch.nn.functional.silu(z).float()
    return y if y.dtype == dtype else y.to(dtype=dtype)


def _linear_span(spec: Spec, layer, x: torch.Tensor, conv_state, rec, linear, exact: bool, in_place: bool = False):
    batch, length, _ = x.shape
    grouped = _project_group(x, (layer.qkv, layer.z, layer.a, layer.b), linear)
    if grouped is None:
        qkv = _project(x, layer.qkv, linear)
        z = _project(x, layer.z, linear)
        a = _project(x, layer.a, linear)
        b = _project(x, layer.b, linear)
    else:
        qkv, z, a, b = grouped
    z = z.view(batch, length, spec.value_heads, spec.value_dim)
    mixed, conv_state = causal_conv(qkv, layer.conv, conv_state, exact=exact, in_place=in_place)
    q, k, v = mixed.split((spec.key_width, spec.key_width, spec.value_width), dim=-1)
    q = q.view(batch, length, spec.key_heads, spec.key_dim)
    k = k.view(batch, length, spec.key_heads, spec.key_dim)
    v = v.view(batch, length, spec.value_heads, spec.value_dim)
    q, k = normalize_qk(q, k, spec.key_dim, spec.eps)
    y, rec = gated_delta(q, k, v, a, b, layer.a_log, layer.dt_bias, rec, fused=not exact and length == 1)
    y = _gated_norm(y, z, layer.gnorm, spec.eps, x.dtype)
    return _project(y.reshape(batch, length, -1), layer.out, linear), conv_state, rec


def _linear_attn(spec: Spec, layer, x: torch.Tensor, cache, linear, exact: bool, in_place: bool = False):
    batch, length, hidden = x.shape
    conv_state = None if cache is None else cache["conv"]
    rec = None if cache is None else cache["state"]
    if length <= SPAN:
        y, conv_state, rec = _linear_span(spec, layer, x, conv_state, rec, linear, exact, in_place)
        return y, {"conv": conv_state, "state": rec}
    y = torch.empty(batch, length, hidden, dtype=torch.float32 if layer.out.partial else x.dtype, device=x.device)
    for start in range(0, length, SPAN):
        stop = min(length, start + SPAN)
        y[:, start:stop], conv_state, rec = _linear_span(
            spec, layer, x[:, start:stop], conv_state, rec, linear, exact)
    return y, {"conv": conv_state, "state": rec}


def _attention(spec: Spec, layer, x: torch.Tensor, cache, linear, pos0: int, exact: bool,
               at: DevicePos | None = None):
    batch, length, hidden = x.shape
    if length <= SPAN:
        return _attention_span(spec, layer, x, cache, linear, pos0, exact, at)
    y = torch.empty(batch, length, hidden, dtype=x.dtype, device=x.device)
    for start in range(0, length, SPAN):
        stop = min(length, start + SPAN)
        y[:, start:stop], cache = _attention_span(
            spec, layer, x[:, start:stop], cache, linear, pos0 + start, exact)
    return y, cache


def _attention_span(spec: Spec, layer, x: torch.Tensor, cache, linear, pos0: int, exact: bool,
                    at: DevicePos | None = None):
    batch, length, _ = x.shape
    grouped = _project_group(x, (layer.q, layer.k, layer.v), linear)
    if grouped is None:
        qg = _project(x, layer.q, linear)
        keys, values = _project_pair(x, layer.k, layer.v, linear)
    else:
        qg, keys, values = grouped
    qg = qg.view(batch, length, spec.heads, spec.head_dim * 2)
    queries, gate = qg.split(spec.head_dim, dim=-1)
    keys = keys.view(batch, length, spec.kv_heads, spec.head_dim)
    values = values.view(batch, length, spec.kv_heads, spec.head_dim)
    queries = rms_norm(queries, layer.q_norm, spec.eps).permute(0, 2, 1, 3)
    keys = rms_norm(keys, layer.k_norm, spec.eps).permute(0, 2, 1, 3)
    values = values.permute(0, 2, 1, 3)
    queries = apply_rope(queries, pos0, spec.rope_theta, spec.rotary_dim, exact=exact, at=at)
    keys = apply_rope(keys, pos0, spec.rope_theta, spec.rotary_dim, exact=exact, at=at)
    if at is not None:
        if length != 1 or cache is None or "len" not in cache:
            raise ValueError("a device position is one token over a fixed key/value buffer")
        cache["k"].index_copy_(2, at.i64, keys.to(dtype=cache["k"].dtype))
        cache["v"].index_copy_(2, at.i64, values.to(dtype=cache["v"].dtype))
        kept_k, kept_v, new_cache = cache["k"], cache["v"], cache
    elif cache is not None and "len" in cache:
        end = cache["len"] + length
        if end > cache["k"].shape[2]:
            raise RuntimeError("kv cache is shorter than the tokens written into it")
        cache["k"][:, :, cache["len"]:end] = keys.to(dtype=cache["k"].dtype)
        cache["v"][:, :, cache["len"]:end] = values.to(dtype=cache["v"].dtype)
        cache["len"] = end
        kept_k = cache["k"][:, :, :end]
        kept_v = cache["v"][:, :, :end]
        new_cache = cache
    elif cache is None:
        kept_k, kept_v = keys, values
        new_cache = {"k": kept_k, "v": kept_v}
    else:
        kept_k = torch.cat((cache["k"], keys), dim=2)
        kept_v = torch.cat((cache["v"], values), dim=2)
        new_cache = {"k": kept_k, "v": kept_v}
    query = queries if queries.dtype == torch.float32 else queries.float()
    if at is not None:
        from tensorfold.rocm.kernels.attention import causal_at

        attended = causal_at(query, kept_k, kept_v, spec.head_dim ** -0.5, at.i32)
    else:
        attended = _attend(query, kept_k, kept_v, spec.head_dim ** -0.5, pos0, exact)
    attended = attended.permute(0, 2, 1, 3).reshape(batch, length, -1)
    gated = attended * torch.sigmoid(gate.reshape(batch, length, -1).float())
    if gated.dtype != x.dtype:
        gated = gated.to(dtype=x.dtype)
    return _project(gated, layer.o, linear), new_cache


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int,
            prefill: bool = False) -> torch.Tensor:
    """Device attention reads the cache dtype directly; ``prefill`` rows take the prefill kernel at any length."""

    # Even head sizes take the device kernels; odd ones the chunked matmul.
    even = q.shape[-1] <= 256 and q.shape[-1] % 2 == 0 and q.shape[1] % k.shape[1] == 0
    if q.is_cuda and k.is_cuda and even:
        from tensorfold.rocm.kernels.attention import causal

        return causal(q, k, v, scale, q_pos0, prefill=prefill)
    return causal_attend(q, k, v, scale, q_pos0)


def _blank_caches(model, batch: int, total: int, device: torch.device, cache_dtype: torch.dtype = torch.float32,
                  ) -> list:
    """One cache per request. Full attention keeps a fixed key/value buffer; linear attention keeps its own state."""

    spec = model.spec
    caches = []
    for index in range(spec.n_layers):
        if spec.full(index):
            shape = (batch, spec.kv_heads, total, spec.head_dim)
            caches.append({
                "k": torch.empty(shape, device=device, dtype=cache_dtype),
                "v": torch.empty(shape, device=device, dtype=cache_dtype),
                "len": 0,
            })
        else:
            caches.append({"conv": None, "state": None})
    return caches


def greedy(model, prompts: list[list[int]], n_new: int, linear, device: torch.device,
           after_token=None, cache_dtype: torch.dtype = torch.float32, *, reduce=None,
           gather=None) -> list[list[int]]:
    """``n_new`` tokens for every prompt in one batch; ``after_token(step)`` runs once a step is queued."""

    if n_new < 1:
        raise ValueError("n_new must be positive")
    tokens = torch.tensor(prompts, dtype=torch.long, device=device)
    if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
        raise ValueError("prompts must be a non-empty rectangular batch")
    batch, length = tokens.shape
    caches = _blank_caches(model, batch, length + n_new, device, cache_dtype)
    out = [[] for _ in range(batch)]

    def commit(step: int, nxt: torch.Tensor) -> None:
        # The copy lands the ids on the host. The clock, if any, starts only after that sync.
        ids = [int(token) for token in nxt.tolist()]
        if after_token is not None:
            after_token(step)
        for row, token in enumerate(ids):
            out[row].append(token)

    if after_token is not None:
        after_token(-1)
    def pick(hidden: torch.Tensor) -> torch.Tensor:
        logits = _project(hidden[:, -1], model.output_head(), linear)
        return torch.argmax(logits if gather is None else gather(logits), dim=-1)

    hidden, caches = forward_hidden(model, tokens, caches, linear, 0, cache_dtype, reduce=reduce)
    nxt = pick(hidden)
    commit(0, nxt)
    for step in range(1, n_new):
        hidden, caches = forward_hidden(model, nxt.view(-1, 1), caches, linear, length + step - 1, cache_dtype,
                                        reduce=reduce)
        nxt = pick(hidden)
        commit(step, nxt)
    if any(len(row) != n_new for row in out):
        raise RuntimeError("generation stopped before the requested token count")
    return out
