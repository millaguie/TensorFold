"""RDNA4's MXFP4 matmuls (``mx4_rocm.cu``) and the weight layout they read."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_mx4_rocm_v14", sources=[str(here / "mx4_rocm.cpp"), str(here / "mx4_rocm.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def to_tiles(weight: torch.Tensor) -> torch.Tensor:
    """(N, K / 2) fp4 bytes -> (N / 16, K / 32, 16, 16): a group's 16 bytes for 16 columns side by side."""

    n, kb = weight.shape
    if n % 16 or kb % 16:
        raise ValueError(f"MXFP4 tiles take N and K / 2 in multiples of 16, not {tuple(weight.shape)}")
    return weight.view(n // 16, 16, kb // 16, 16).permute(0, 2, 1, 3).contiguous()


def from_tiles(tiles: torch.Tensor) -> torch.Tensor:
    t16, kg, _, _ = tiles.shape
    return tiles.permute(0, 2, 1, 3).reshape(t16 * 16, kg * 16)


def prompt(x: tuple[torch.Tensor, torch.Tensor, torch.Tensor], tiles: torch.Tensor, scales_t: torch.Tensor,
           ref: torch.Tensor, n: int, *, f32: bool = False, tn: int | None = None) -> torch.Tensor:
    """prefill_glue's TILED e4m3 fragments (with row scales) times the MXFP4 weight."""

    x8, _, a = x
    if x8.dim() != 3:
        raise ValueError("MXFP4 prompts take prefill_glue's fragment-ordered rows (RDNA4 with qmm8_rocm)")
    m = a.shape[0]
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x8.device)
    _ext().prompt(x8, a, tiles, scales_t, ref, n, out, tn or (4 if n >= 1024 else 2))
    return out


_parts: dict[torch.device, torch.Tensor] = {}


_counts: dict[torch.device, torch.Tensor] = {}


def _count(device: torch.device, n: int) -> torch.Tensor:
    """Arrival counters, one a 128-column block, zero between calls (the last slice of a block resets its own)."""

    c = _counts.get(device)
    blocks = (n + 127) // 128
    if c is None or c.numel() < blocks:
        c = torch.zeros(max(blocks, 2048), dtype=torch.int32, device=device)
        _counts[device] = c
    return c


def _part(device: torch.device, need: int) -> torch.Tensor:
    part = _parts.get(device)
    if part is None or part.numel() < need:
        part = torch.empty(max(need, 1), dtype=torch.float32, device=device)
        _parts[device] = part
    return part


def decode(x: torch.Tensor, tiles: torch.Tensor, scales_t: torch.Tensor, ref: torch.Tensor, n: int, *,
           f32: bool = False) -> torch.Tensor:
    """bf16 rows times the MXFP4 weight, 48 rows a launch: a row's bits do not depend on the row count."""

    x = x.to(torch.bfloat16).contiguous()
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    ext = _ext()
    part = _part(x.device, ext.decode_slices(n, k // 32) * min(m, 48) * n)
    for a in range(0, m, 48):
        ext.decode(x[a:a + 48], tiles, scales_t, ref, n, out[a:a + 48], part, _count(x.device, n))
    return out


def b16(x: torch.Tensor, weight: torch.Tensor, *, f32: bool = False) -> torch.Tensor:
    """bf16 rows times a bf16 (N, K) weight as stored, 48 rows a launch, row-count invariant."""

    x = x.to(torch.bfloat16).contiguous()
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    ext = _ext()
    part = _part(x.device, ext.decode_slices(n, k // 32) * min(m, 48) * n)
    for a in range(0, m, 48):
        ext.decode_b16(x[a:a + 48], weight, out[a:a + 48], part, _count(x.device, n))
    return out


def host_rows(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Rows ``ids`` of a bf16 table in pinned host memory, gathered on the GPU over PCIe."""

    flat = ids.reshape(-1).to(torch.int64).contiguous()
    out = torch.empty((flat.numel(), table.shape[1]), dtype=torch.bfloat16, device=ids.device)
    _ext().host_rows(table, flat, out)
    return out.view(*ids.shape, table.shape[1])
