"""Qwen3.5 text math for the ROCm forward; the projection is passed in."""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass

import torch

_FMAF = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6").fmaf
_FMAF.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_FMAF.restype = ctypes.c_float

# Tokens resident in one prefill step. A multiple of the 16-wide matmul tile.
SPAN = 2048


@dataclass
class Packed:
    """One affine matrix; ``partial`` marks a tp slice along K that returns an fp32 share."""

    words: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor
    bits: int
    group: int
    partial: bool = False


@dataclass
class Dense:
    """An unquantized projection a conversion kept in float (an MTP head's fc): fp32 (N, K)."""

    weight: torch.Tensor
    partial: bool = False


@dataclass
class GptqPacked:
    """One W4A16 GPTQ / AWQ projection for the RDNA2 fp16 dot; ``v2`` is AWQ's literal zeros."""

    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor | None = None
    v2: bool = False


@dataclass
class Spec:
    hidden: int
    intermediate: int
    n_layers: int
    heads: int
    kv_heads: int
    head_dim: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv: int
    vocab: int
    eps: float
    rope_theta: float
    rotary_dim: int
    full_every: int
    bits: int
    group: int
    experts: int = 0
    top_k: int = 0
    moe_width: int = 0

    def full(self, index: int) -> bool:
        return (index + 1) % self.full_every == 0

    @property
    def key_width(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def value_width(self) -> int:
        return self.value_heads * self.value_dim


def _rms_rows(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    return y if y.dtype == x.dtype else y.to(dtype=x.dtype)


def _rms_torch(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    width = x.shape[-1]
    flat = x.reshape(-1, width)
    if flat.shape[0] <= SPAN:
        return _rms_rows(flat, weight, eps).reshape(x.shape)
    out = torch.empty_like(flat)
    for start in range(0, flat.shape[0], SPAN):
        stop = min(start + SPAN, flat.shape[0])
        out[start:stop] = _rms_rows(flat[start:stop], weight, eps)
    return out.reshape(x.shape)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """fp32 RMSNorm. Device rows take one HIP block each at any row count, so a row's bits do not depend on its span."""

    width = x.shape[-1]
    rows = x.numel() // width
    weight_ok = weight is None or weight.numel() == width
    if x.is_cuda and weight_ok and rows >= 1 and 1 <= width <= 8192:
        from tensorfold.rocm.kernels.act import rms

        # The kernel reads and writes the activation dtype and computes in fp32: no cast either side.
        flat = x.reshape(rows, width)
        if flat.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            flat = flat.float()
        scale = None if weight is None else weight.reshape(width).float().contiguous()
        y = rms(flat.contiguous(), scale, eps).reshape(x.shape)
        return y if y.dtype == x.dtype else y.to(dtype=x.dtype)
    return _rms_torch(x, weight, eps)


_FILLED: dict[tuple, torch.Tensor] = {}


def _filled(width: int, value: float, device: torch.device) -> torch.Tensor:
    """A constant fp32 weight, made once per shape (a prefill makes it before any decode graph is captured)."""

    key = (width, value, str(device))
    weight = _FILLED.get(key)
    if weight is None:
        weight = _FILLED[key] = torch.full((width,), value, dtype=torch.float32, device=device)
    return weight


def normalize_qk(q: torch.Tensor, k: torch.Tensor, head_k: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize q and k with ``head_k ** -0.5`` folded in as the norm weight (same bits)."""

    inv = head_k ** -0.5
    rms_eps = eps * inv * inv
    return (rms_norm(q, _filled(q.shape[-1], inv * inv, q.device), rms_eps),
            rms_norm(k, _filled(k.shape[-1], inv, k.device), rms_eps))


class DevicePos:
    """A decode step's position on the device, so one captured step serves every position."""

    def __init__(self, device: torch.device) -> None:
        self.i32 = torch.zeros(1, dtype=torch.int32, device=device)
        self.i64 = torch.zeros(1, dtype=torch.int64, device=device)

    def set(self, pos: int) -> None:
        self.i32.fill_(pos)
        self.i64.fill_(pos)


def apply_rope(x: torch.Tensor, pos0: int, theta: float, rotary_dim: int, *, exact: bool = False,
               at: DevicePos | None = None) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` features; ``exact`` keeps the prefill formula, ``at`` a device position."""

    width = x.shape[-1]
    rows = x.numel() // width
    short = not exact and x.is_cuda and x.shape[-2] == 1 and rows <= 256 and width <= 8192
    if short and 0 < rotary_dim <= width and rotary_dim % 2 == 0:
        from tensorfold.rocm.kernels.act import rope_decode

        flat = x.reshape(rows, width).float().contiguous()
        y = rope_decode(flat, pos0 if at is None else at.i32, rotary_dim, theta).reshape(x.shape)
        return y if y.dtype == x.dtype else y.to(dtype=x.dtype)
    half = rotary_dim // 2
    freq = 1.0 / (theta ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    if at is not None:
        if x.shape[2] != 1:
            raise ValueError("a device position is one row")
        pos = at.i64.to(torch.float32)
    else:
        pos = torch.arange(pos0, pos0 + x.shape[2], device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]
    cos = ang.cos()[None, None]
    sin = ang.sin()[None, None]
    x1 = x[..., :half].float()
    x2 = x[..., half:rotary_dim].float()
    rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    if rotary_dim == x.shape[-1]:
        full = rot
    else:
        full = torch.cat((rot, x[..., rotary_dim:].float()), dim=-1)
    return full if full.dtype == x.dtype else full.to(dtype=x.dtype)


def causal_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int) -> torch.Tensor:
    """Causal attention. Query position ``q_pos0 + i`` reads keys ``0 .. q_pos0 + i``. Computed in query chunks."""

    heads = q.shape[1]
    kv_heads = k.shape[1]
    if heads != kv_heads:
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
    length = q.shape[2]
    span = k.shape[2]
    pieces = []
    step = 128
    key_pos = torch.arange(span, device=q.device)
    for start in range(0, length, step):
        stop = min(length, start + step)
        scores = torch.matmul(q[:, :, start:stop].float(), k.float().transpose(-1, -2)) * scale
        pos = torch.arange(start, stop, device=q.device) + q_pos0
        scores = scores.masked_fill(key_pos.view(1, 1, 1, span) > pos.view(1, 1, -1, 1), float("-inf"))
        pieces.append(torch.matmul(torch.softmax(scores, dim=-1), v.float()))
    return torch.cat(pieces, dim=2)


_CONV_TYPES = (torch.float32, torch.float16, torch.bfloat16)


def causal_conv(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor | None, *,
                exact: bool = False, in_place: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv; ``in_place`` keeps the state's buffer.

    A device prompt takes one kernel with the loop's arithmetic (the same bits), ``exact`` or not; ``exact`` only keeps
    a one-token step off the decode kernel, whose silu rounds differently.
    """

    batch, length, channels = x.shape
    kernel = weight.shape[1]
    if not exact and x.is_cuda and length == 1 and 1 <= kernel <= 8:
        from tensorfold.rocm.kernels.act import conv_decode

        if state is None:
            state = torch.zeros(batch, kernel - 1, channels, device=x.device, dtype=torch.float32)
        else:
            state = state.to(dtype=torch.float32).contiguous()
        sample = x.reshape(batch, 1, channels).float().contiguous()
        y = conv_decode(sample, weight.float().contiguous(), state)
        return y.view(batch, 1, channels), state
    if state is None:
        state = x.new_zeros(batch, kernel - 1, channels)
    if (x.is_cuda and length > 1 and 1 <= kernel <= 8 and length <= 65535 and batch <= 65535
            and x.dtype in _CONV_TYPES):
        # The loop below in one kernel, the same products and adds in the same order: the same bits.
        from tensorfold.rocm.kernels.act import conv_prefill

        y = conv_prefill(x.contiguous(), weight.float().contiguous(), state.float().contiguous())
        kept = kernel - 1
        if length >= kept:
            tail = x[:, length - kept:].float()
        else:
            tail = torch.cat((state.float(), x.float()), dim=1)[:, length:]
        if in_place and state.dtype == torch.float32:
            state.copy_(tail)
            return y, state
        return y, tail.contiguous()
    window = torch.cat((state.float(), x.float()), dim=1)
    out = torch.zeros(batch, length, channels, device=x.device, dtype=torch.float32)
    taps = weight.float()
    for tap in range(kernel):
        out.add_(window[:, tap:tap + length] * taps[:, tap].view(1, 1, channels))
    if in_place and state.dtype == torch.float32:
        state.copy_(window[:, length:])
        return torch.nn.functional.silu(out), state
    return torch.nn.functional.silu(out), window[:, length:].contiguous()


def _gate_beta(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor]:
    beta = torch.sigmoid(b.float())
    gate = torch.exp(-torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float()))
    return gate, beta


def _lanes(x: torch.Tensor, span: int) -> torch.Tensor:
    """Pack the last axis as ``(span, 32)`` with index ``lane + i * 32``. Unused lanes stay zero."""

    width = span * 32
    if x.shape[-1] != width:
        padded = x.new_zeros(*x.shape[:-1], width)
        padded[..., :x.shape[-1]] = x
        x = padded
    return x.reshape(*x.shape[:-1], span, 32)


def _warp0(partial: torch.Tensor) -> torch.Tensor:
    """Lane 0 after a xor-shuffle of 16, 8, 4, 2, 1. Each lane adds ``x[i] + x[i ^ mask]``."""

    x = partial
    index = torch.arange(32, device=partial.device)
    for mask in (16, 8, 4, 2, 1):
        x = x + x[..., index ^ mask]
    return x[..., 0]


def gated_delta_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor,
                          beta: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's reduction, in PyTorch. Decay, then the value residual, then the q readout."""

    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if key_dim not in (16, 128) or value_heads % key_heads != 0:
        raise ValueError("dk is 16 or 128 and value heads are a multiple of key heads")
    span = 1 if key_dim < 32 else key_dim // 32
    repeat = value_heads // key_heads
    q = q.float().contiguous()
    k = k.float().contiguous()
    v = v.float().contiguous()
    gate = gate.float().reshape(batch, length, value_heads)
    beta = beta.float().reshape(batch, length, value_heads)
    state = state.float().contiguous().clone()
    y = torch.empty(batch, length, value_heads, value_dim, device=q.device, dtype=torch.float32)
    heads = torch.arange(value_heads, device=q.device) // repeat
    for t in range(length):
        scaled = state * gate[:, t].view(batch, value_heads, 1, 1)
        kt = k[:, t].index_select(1, heads)
        qt = q[:, t].index_select(1, heads)
        st_l = _lanes(scaled, span)
        k_l = _lanes(kt, span).unsqueeze(2)
        partial = torch.zeros(batch, value_heads, value_dim, 32, device=q.device, dtype=torch.float32)
        for i in range(span):
            partial = partial + st_l[..., i, :] * k_l[..., i, :]
        delta = (v[:, t] - _warp0(partial)) * beta[:, t].view(batch, value_heads, 1)
        for i in range(span):
            st_l[..., i, :] = st_l[..., i, :] + k_l[..., i, :] * delta.unsqueeze(-1)
        state = st_l.reshape(batch, value_heads, value_dim, span * 32)[..., :key_dim].contiguous()
        q_l = _lanes(qt, span).unsqueeze(2)
        acc = torch.zeros_like(partial)
        for i in range(span):
            acc = acc + st_l[..., i, :] * q_l[..., i, :]
        y[:, t] = _warp0(acc)
    return y, state


def gated_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                a_log: torch.Tensor, dt_bias: torch.Tensor, state: torch.Tensor | None, *, fused: bool = False,
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar-gate delta rule over fp32 state; ``fused`` takes gate and beta from one HIP launch."""

    if fused and a.is_cuda:
        from tensorfold.rocm.kernels.act import gdn_gate

        gate, beta = gdn_gate(a, b, a_log, dt_bias)
    else:
        gate, beta = _gate_beta(a, b, a_log, dt_bias)
    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if state is None:
        state = torch.zeros(batch, value_heads, value_dim, key_dim, device=q.device, dtype=torch.float32)
    else:
        state = state.float()
    qf = q.float().contiguous()
    kf = k.float().contiguous()
    vf = v.float().contiguous()
    gate = gate.reshape(batch, length, value_heads).contiguous()
    beta = beta.reshape(batch, length, value_heads).contiguous()
    if qf.is_cuda:
        from tensorfold.rocm.kernels.gated_delta import recurrence

        return recurrence(qf, kf, vf, gate, beta, state.contiguous())
    return gated_delta_reference(qf, kf, vf, gate, beta, state)


def _codes(words: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """Unpack codes the way the HIP reader does, including a code that crosses two words."""

    words64 = words.to(torch.int64) & 0xFFFFFFFF
    index = torch.arange(k, device=words.device)
    bit = index * bits
    word = bit // 32
    shift = bit % 32
    shape = (*words.shape[:-1], k)
    low = torch.gather(words64, -1, word.expand(shape))
    nxt = (word + 1).clamp(max=words.shape[-1] - 1)
    high = torch.gather(words64, -1, nxt.expand(shape))
    high = torch.where((shift + bits > 32) & (word + 1 < words.shape[-1]), high, torch.zeros_like(high))
    value = (low >> shift) | (high << ((32 - shift) & 31))
    return value & ((1 << bits) - 1)


def _dequant_rows(packed: Packed, ids: torch.Tensor) -> torch.Tensor:
    """Fp32 rows for a 1-D id list. The packed table is not expanded."""

    words = packed.words[ids]
    groups = packed.scale.shape[1]
    k = groups * packed.group
    if packed.bits == 8:
        # Eight-bit codes are little-endian bytes of the int32 words, one code per byte.
        codes = words.contiguous().view(torch.uint8)[..., :k].to(torch.float32)
    else:
        codes = _codes(words, packed.bits, k).to(torch.float32)
    codes = codes.to(torch.bfloat16).to(torch.float32)
    scale = packed.scale[ids].to(torch.float32).unsqueeze(-1)
    bias = packed.bias[ids].to(torch.float32).unsqueeze(-1)
    return (codes.view(ids.shape[0], groups, packed.group) * scale + bias).reshape(ids.shape[0], k)


def gather_rows(packed: Packed, ids: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dequantize selected embedding rows from packed codes. The table itself stays packed."""

    ids = ids.to(packed.words.device)
    k = packed.scale.shape[1] * packed.group
    flat = ids.reshape(-1)
    out = torch.empty(flat.shape[0], k, dtype=torch.float32 if dtype is None else dtype, device=ids.device)
    for start in range(0, flat.shape[0], SPAN):
        stop = min(start + SPAN, flat.shape[0])
        piece = _dequant_rows(packed, flat[start:stop])
        out[start:stop] = piece if piece.dtype == out.dtype else piece.to(dtype=out.dtype)
    return out.view(*ids.shape, k)


def affine_reference(x: torch.Tensor, packed: Packed) -> torch.Tensor:
    """Scalar affine formula with libm fmaf. Group order matches the GEMV kernel. Does not call the extension."""

    values = x.detach().float().to(torch.bfloat16).float().cpu()
    codes = _codes(packed.words.cpu(), packed.bits, values.shape[-1]).float().to(torch.bfloat16).float()
    scale = packed.scale.float().cpu()
    bias = packed.bias.float().cpu()
    rows, k = values.shape
    cols = codes.shape[0]
    acc = torch.zeros(rows, cols)
    group = packed.group
    for start in range(0, k, group):
        dot = torch.zeros(rows, cols)
        summed = torch.zeros(rows)
        block_x = values[:, start:start + group]
        block_q = codes[:, start:start + group]
        for t in range(group):
            xv = block_x[:, t].contiguous().numpy()
            qv = block_q[:, t].contiguous().numpy()
            # Broadcast the column of x across outputs and call fmaf in C order.
            prod = torch.empty(rows, cols)
            flat_x = torch.from_numpy(xv).unsqueeze(1).expand(rows, cols).reshape(-1).numpy()
            flat_q = torch.from_numpy(qv).unsqueeze(0).expand(rows, cols).reshape(-1).numpy()
            flat_d = dot.reshape(-1).numpy()
            out = prod.reshape(-1).numpy()
            fmaf = _FMAF
            for i in range(out.shape[0]):
                out[i] = fmaf(float(flat_x[i]), float(flat_q[i]), float(flat_d[i]))
            dot = torch.from_numpy(out.copy()).view(rows, cols)
            summed = summed + block_x[:, t]
        g = start // group
        acc = _fma_cols(dot, scale[:, g], acc)
        acc = _fma_rows(summed, bias[:, g], acc)
    return acc.to(torch.bfloat16)


def _fma_cols(dot: torch.Tensor, scale: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    rows, cols = dot.shape
    out = torch.empty(rows, cols)
    flat_a = dot.reshape(-1).numpy()
    flat_b = scale.expand(rows, cols).reshape(-1).numpy()
    flat_c = acc.reshape(-1).numpy()
    dest = out.reshape(-1).numpy()
    fmaf = _FMAF
    for i in range(dest.shape[0]):
        dest[i] = fmaf(float(flat_a[i]), float(flat_b[i]), float(flat_c[i]))
    return torch.from_numpy(dest.copy()).view(rows, cols)


def _fma_rows(summed: torch.Tensor, bias: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    rows, cols = acc.shape
    out = torch.empty(rows, cols)
    flat_a = summed.unsqueeze(1).expand(rows, cols).reshape(-1).numpy()
    flat_b = bias.expand(rows, cols).reshape(-1).numpy()
    flat_c = acc.reshape(-1).numpy()
    dest = out.reshape(-1).numpy()
    fmaf = _FMAF
    for i in range(dest.shape[0]):
        dest[i] = fmaf(float(flat_a[i]), float(flat_b[i]), float(flat_c[i]))
    return torch.from_numpy(dest.copy()).view(rows, cols)
