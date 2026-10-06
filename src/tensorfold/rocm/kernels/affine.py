"""Packed MLX affine projections. The kernel reads the integer words and applies each group's scale and bias."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)


SCALE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _tables(scale: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and bias as stored when they are fp32, bf16 or fp16 of one type; the kernel widens them exactly."""

    if scale.dtype != bias.dtype or scale.dtype not in SCALE_DTYPES:
        scale, bias = scale.to(torch.float32), bias.to(torch.float32)
    return scale.contiguous(), bias.contiguous()


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.kernels.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_affine",
                sources=[str(here / name) for name in ("affine.cpp", "affine_gemv.hip", "affine_wmma.hip",
                                                       "affine_wmma_pair.hip", "affine_dot2.hip", "affine_tiles.hip")],
                extra_include_paths=[str(here)], verbose=False)


def matmul(x: torch.Tensor, words: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, *, bits: int,
           group: int, schedule: str = "auto", f32: bool = False, dot2_split: bool | None = None) -> torch.Tensor:
    """``x`` (M, K) BF16, or FP16 on RDNA2, times packed words (N, K * bits / 32)."""

    from tensorfold.rocm.kernels.build import WMMA, gfx_name

    if bits not in BITS or group not in GROUPS:
        raise ValueError("RDNA affine weights require 2/3/4/5/6/8 bits and groups of 32/64/128")
    which = {"auto": 0, "gemv": 1, "wmma": 2, "decode": 3}[schedule]
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda or not x.is_contiguous():
        raise ValueError("affine inputs must be a contiguous BF16 or FP16 matrix on the device")
    if x.dtype == torch.float16 and gfx_name() in WMMA:
        raise ValueError("FP16 activations are the RDNA2 schedule")
    m, k = x.shape
    if k % group != 0 or (k * bits) % 32 != 0:
        raise ValueError("K must be whole groups and whole packed words")
    if words.ndim != 2 or words.dtype != torch.int32 or words.shape[1] != k * bits // 32:
        raise ValueError("packed words must be int32 of shape (N, K * bits / 32)")
    n = words.shape[0]
    groups = k // group
    scale, bias = _tables(scale, bias)
    if scale.shape != (n, groups) or bias.shape != scale.shape:
        raise ValueError("scale and bias must be (N, K / group)")
    if not all(t.is_cuda and t.device == x.device for t in (words, scale, bias)):
        raise ValueError("affine operands must share the input's device")
    # None follows the column grid. False is one launch. True splits K on group boundaries.
    split_mode = 0 if dot2_split is None else (2 if dot2_split else 1)
    # The RDNA2 decode tile rounds its fp16 output itself: the same bits as the fp32 result cast after.
    half = (not f32 and x.dtype == torch.float16 and m <= 8 and which == 0 and split_mode != 2
            and gfx_name() not in WMMA)
    # The gfx11 / gfx12 decode tile rounds its bf16 output itself, as the cast after would.
    brain = (not f32 and x.dtype == torch.bfloat16 and m <= 8 and which == 0 and split_mode != 2
             and gfx_name() in WMMA)
    out = torch.empty((m, n), dtype=torch.float16 if half else torch.bfloat16 if brain else torch.float32,
                      device=x.device)
    _ext().affine(x, words.contiguous(), scale, bias, out, bits, group, which, split_mode)
    return out if f32 or half or brain else out.to(x.dtype)


def matmul_routed(x: torch.Tensor, words: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, items: torch.Tensor,
                  members: torch.Tensor, *, pairs: int, x_div: int, rows: int, bits: int, group: int) -> torch.Tensor:
    """Every item (expert, first, count) of a plan in one launch: (pairs, N) fp32 by pair id."""

    if bits not in BITS or group not in GROUPS:
        raise ValueError("RDNA affine weights require 2/3/4/5/6/8 bits and groups of 32/64/128")
    scale, bias = _tables(scale, bias)
    out = torch.empty((pairs, words.shape[1]), dtype=torch.float32, device=x.device)
    _ext().affine_routed(x.contiguous(), words, scale, bias, out, items, members, x_div, rows, bits, group)
    return out


def _one_type(tables: list[torch.Tensor]) -> list[torch.Tensor]:
    """One launch reads one scale type. Mixed sides widen to fp32."""

    if len({t.dtype for t in tables}) == 1:
        return tables
    return [t.to(torch.float32) for t in tables]


def _as_affine(words, scale, bias, k, bits, group):
    if words.dtype != torch.int32 or words.ndim != 2 or words.shape[1] != k * bits // 32:
        raise ValueError("packed words must be int32 of shape (N, K * bits / 32)")
    n = words.shape[0]
    groups = k // group
    scale, bias = _tables(scale, bias)
    if scale.shape != (n, groups) or bias.shape != scale.shape:
        raise ValueError("scale and bias must be (N, K / group)")
    return words.contiguous(), scale, bias, n


def matmul_pair(x: torch.Tensor, words_a: torch.Tensor, scale_a: torch.Tensor, bias_a: torch.Tensor,
                words_b: torch.Tensor, scale_b: torch.Tensor, bias_b: torch.Tensor, *, bits: int, group: int,
                f32: bool = False):
    """Two packed products that share ``x``. Each side matches a solo WMMA launch. 8-bit only."""

    from tensorfold.rocm.kernels.build import WMMA, gfx_name

    if x.dtype != torch.bfloat16 or gfx_name() not in WMMA or bits != 8:
        raise ValueError("the paired matmul is the BF16 WMMA 8-bit schedule")
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("affine inputs must be a contiguous BF16 matrix on the device")
    m, k = x.shape
    if k % group != 0:
        raise ValueError("K must be whole groups")
    wa, sa, ba, na = _as_affine(words_a, scale_a, bias_a, k, bits, group)
    wb, sb, bb, nb = _as_affine(words_b, scale_b, bias_b, k, bits, group)
    if na != nb:
        raise ValueError("a paired matmul needs both sides to share N")
    sa, ba, sb, bb = _one_type([sa, ba, sb, bb])
    out_a = torch.empty((m, na), dtype=torch.float32, device=x.device)
    out_b = torch.empty((m, nb), dtype=torch.float32, device=x.device)
    _ext().affine_pair(x, wa, sa, ba, out_a, wb, sb, bb, out_b, bits, group)
    if f32:
        return out_a, out_b
    return out_a.to(x.dtype), out_b.to(x.dtype)


def matmul_rows(x: torch.Tensor, packeds: tuple, *, bits: int, group: int):
    """Up to four products of a one- or two-row BF16 ``x`` in one launch of the gfx11 / gfx12 row tile, bf16 out.

    Each output has the bits of its solo ``matmul``. None when a product would not take the row tile alone.
    """

    if not 1 <= len(packeds) <= 4 or x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return None
    m, k = x.shape
    if m > 2 or k % group or bits not in BITS or group not in GROUPS:
        return None
    words, scale, bias, outs = [], [], [], []
    for packed in packeds:
        w, s, b, n = _as_affine(*packed, k, bits, group)
        words.append(w)
        scale.append(s)
        bias.append(b)
        outs.append(torch.empty((m, n), dtype=torch.bfloat16, device=x.device))
    if len({t.dtype for t in scale + bias}) != 1:
        return None
    if not _ext().affine_rows(x, words, scale, bias, outs, bits, group):
        return None
    return tuple(outs)


def matmul_group(x: torch.Tensor, packeds: tuple, *, bits: int, group: int, f32: bool = False):
    """Up to four packed products that share ``x``. Each side matches a solo WMMA launch."""

    from tensorfold.rocm.kernels.build import WMMA, gfx_name

    if not 1 <= len(packeds) <= 4:
        raise ValueError("a grouped matmul takes 1 to 4 weights")
    if x.dtype != torch.bfloat16 or gfx_name() not in WMMA or bits != 8:
        raise ValueError("the grouped matmul is the BF16 WMMA 8-bit schedule")
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("affine inputs must be a contiguous BF16 matrix on the device")
    m, k = x.shape
    if k % group != 0:
        raise ValueError("K must be whole groups")
    words, scale, bias, outs = [], [], [], []
    for packed in packeds:
        w, s, b, n = _as_affine(*packed, k, bits, group)
        words.append(w)
        scale.append(s)
        bias.append(b)
        outs.append(torch.empty((m, n), dtype=torch.float32, device=x.device))
    tables = _one_type(scale + bias)
    scale, bias = tables[:len(scale)], tables[len(scale):]
    _ext().affine_group(x, words, scale, bias, outs, bits, group)
    if f32:
        return tuple(outs)
    return tuple(y.to(x.dtype) for y in outs)
