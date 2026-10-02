"""The 27B prefill's small kernels, one program a row, writing FP8 matmul inputs as e4m3 straight from fp32."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _src(BLOCK: tl.constexpr):
    """Stored position m holds input src: 32-input blocks in fragment order (see ``qmm.quantize_rows``)."""

    m = tl.arange(0, BLOCK)
    j = m % 4
    return (m // 32) * 32 + ((m % 32) // 16) * 16 + ((m % 16) // 4) * 2 + (j % 2) + (j // 2) * 8


@triton.jit
def _quantize(v, row, X8, XS, A, K: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr, TILED: tl.constexpr = False):
    m = tl.arange(0, BLOCK)
    a = tl.maximum(tl.max(tl.abs(v), 0), 1e-30) / 448.0
    if TILED:   # WMMA fragment order (RDNA4's qmm8_rocm.cu): 16 rows x 16 positions a 256-byte fragment, by lane
        at = (((row // 16) * (K // 16) + m // 16) * 32 + row % 16 + 16 * ((m % 16) // 8)) * 8 + m % 8
    else:
        at = row * K + m
    tl.store(X8 + at, (v / a).to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m < K)
    g = tl.sum(tl.reshape(v, (BLOCK // GS, GS)), 1) / a
    gi = tl.arange(0, BLOCK // GS)
    tl.store(XS + row * (K // GS) + gi, g.to(tl.bfloat16), mask=gi < K // GS)
    tl.store(A + row, a)


@triton.jit
def _add_rmsnorm(X, R, W, H, X8, XS, A, eps, D: tl.constexpr, HAS_R: tl.constexpr, GS: tl.constexpr,
                 BLOCK: tl.constexpr, TILED: tl.constexpr):
    """h = x + r (bf16, the residual); rmsnorm(h) * w, quantized."""

    row = tl.program_id(0)
    src = _src(BLOCK)
    ok = src < D
    x = tl.load(X + row * D + src, mask=ok, other=0.0).to(tl.float32)
    if HAS_R:
        h = (x + tl.load(R + row * D + src, mask=ok, other=0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(H + row * D + src, h, mask=ok)
        x = h.to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
    _quantize(x * inv * tl.load(W + src, mask=ok, other=0.0).to(tl.float32), row, X8, XS, A, D, GS, BLOCK, TILED)


@triton.jit
def _swiglu(GATE, UP, X8, XS, A, N: tl.constexpr, GS: tl.constexpr, BLOCK: tl.constexpr, TILED: tl.constexpr):
    row = tl.program_id(0)
    src = _src(BLOCK)
    ok = src < N
    g = tl.load(GATE + row * N + src, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(UP + row * N + src, mask=ok, other=0.0).to(tl.float32)
    _quantize(g * tl.sigmoid(g) * u, row, X8, XS, A, N, GS, BLOCK, TILED)


@triton.jit
def _gated_norm(Yr, Z, W, X8, XS, A, eps, VH: tl.constexpr, DV: tl.constexpr, GS: tl.constexpr,
                HEADS: tl.constexpr, TILED: tl.constexpr):
    """silu(z) * rmsnorm(y) * w a value head (HEADS: VH rounded up to a power of two), quantized."""

    row = tl.program_id(0)
    src = tl.reshape(_src(HEADS * DV), (HEADS, DV))
    ok = src < VH * DV
    y = tl.load(Yr + row * VH * DV + src, mask=ok, other=0.0).to(tl.float32)
    z = tl.load(Z + row * VH * DV + src, mask=ok, other=0.0).to(tl.float32)
    w = tl.load(W + src % DV, mask=ok, other=0.0).to(tl.float32)
    yn = y * (1.0 / tl.sqrt(tl.sum(y * y, 1) / DV + eps))[:, None] * w
    _quantize(tl.reshape(z * tl.sigmoid(z) * yn, (HEADS * DV,)), row, X8, XS, A, VH * DV, GS, HEADS * DV, TILED)


@triton.jit
def _gate_mul(O, QG, X8, XS, A, H: tl.constexpr, D: tl.constexpr, GS: tl.constexpr, HEADS: tl.constexpr,
              TILED: tl.constexpr):
    """Attention output times sigmoid(gate) from the [q | gate] rows, quantized."""

    row = tl.program_id(0)
    src = tl.reshape(_src(HEADS * D), (HEADS, D))
    ok = src < H * D
    head, d = src // D, src % D
    v = tl.load(O + (row * H + head) * D + d, mask=ok, other=0.0).to(tl.float32) * tl.sigmoid(
        tl.load(QG + (row * H + head) * 2 * D + D + d, mask=ok, other=0.0).to(tl.float32))
    _quantize(tl.reshape(v, (HEADS * D,)), row, X8, XS, A, H * D, GS, HEADS * D, TILED)


def tiled() -> bool:
    """e4m3 rows in WMMA fragment order: RDNA4 with qmm8_rocm.cu's prompt matmul (TF_ROCM_PREFILL8_KERNEL=hip)."""

    from tensorfold.cuda.build import gfx12
    from tensorfold.cuda.kernels.qmm_groups import prefill8_kernel

    return gfx12() and prefill8_kernel() == "hip"


def _outputs(rows: int, k: int, device, gs: int = 64):
    """(e4m3 rows, group sums, row scales). Tiled rows are (row tiles, K / 16, 256) bytes: fragment (r // 16, m // 16)
    holds rows 16 (r // 16) .. + 15 at positions 16 (m // 16) .. + 15, lane l's 8 bytes at 8 l being row l % 16,
    positions 8 (l // 16) .. + 7. A last partial tile's padding rows are left unwritten (they reach no output)."""

    x8 = (torch.empty(((rows + 15) // 16, k // 16, 256), dtype=torch.uint8, device=device) if tiled() else
          torch.empty((rows, k), dtype=torch.uint8, device=device))
    return (x8,
            torch.empty((rows, k // gs), dtype=torch.bfloat16, device=device),
            torch.empty((rows,), dtype=torch.float32, device=device))


def _warps(block: int) -> int:
    return max(4, min(32, block // 1024))


def add_rmsnorm(x: torch.Tensor, r: torch.Tensor | None, w: torch.Tensor, eps: float):
    """(h = x + r, quantized rmsnorm(h) * w); without ``r``, h is x itself."""

    rows, d = x.shape
    h = torch.empty_like(x) if r is not None else x
    q = _outputs(rows, d, x.device)
    block = triton.next_power_of_2(d)
    _add_rmsnorm[(rows,)](x, r if r is not None else x, w, h, *q, eps, D=d, HAS_R=r is not None, GS=64,
                          BLOCK=block, TILED=q[0].dim() == 3, num_warps=_warps(block))
    return h, q


def swiglu(gate: torch.Tensor, up: torch.Tensor):
    rows, n = gate.shape
    q = _outputs(rows, n, gate.device)
    block = triton.next_power_of_2(n)
    _swiglu[(rows,)](gate, up, *q, N=n, GS=64, BLOCK=block, TILED=q[0].dim() == 3, num_warps=_warps(block))
    return q


def gated_norm(y: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float):
    rows, vh, dv = y.shape
    q = _outputs(rows, vh * dv, y.device)
    heads = triton.next_power_of_2(vh)
    _gated_norm[(rows,)](y, z, w, *q, eps, VH=vh, DV=dv, GS=64, HEADS=heads, TILED=q[0].dim() == 3,
                         num_warps=_warps(heads * dv))
    return q


def gate_mul(o: torch.Tensor, qg: torch.Tensor, *, heads: int, head_dim: int):
    rows = o.shape[0]
    q = _outputs(rows, heads * head_dim, o.device)
    hp = triton.next_power_of_2(heads)
    _gate_mul[(rows,)](o, qg, *q, H=heads, D=head_dim, GS=64, HEADS=hp, TILED=q[0].dim() == 3,
                       num_warps=_warps(hp * head_dim))
    return q
