"""RDNA4's MXFP4 matmuls (``mx4_rocm.cu``) and the weight layout they read."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_mx4_rocm_v1", sources=[str(here / "mx4_rocm.cpp"), str(here / "mx4_rocm.cu")],
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
