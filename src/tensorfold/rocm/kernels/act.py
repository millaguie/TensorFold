"""Short-row RMSNorm, length-1 causal conv, and length-1 RoPE. Prefill keeps the PyTorch ops."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.rocm.kernels.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_rocm_act",
                sources=[str(here / "act.cpp"), str(here / "act.hip")],
                extra_include_paths=[str(here)], verbose=False)


def rms(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``x`` is (rows, width) fp32, fp16 or bf16; ``y`` has its dtype, computed in fp32. ``weight`` is fp32."""

    y = torch.empty_like(x)
    _ext().rms(x, weight if weight is not None else torch.empty(0, device=x.device), y, float(eps))
    return y


def conv_decode(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Length-1 depthwise conv. ``state`` is updated in place."""

    y = torch.empty_like(x)
    _ext().conv_decode(x, weight, state, y)
    return y


def conv_prefill(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """A prompt's depthwise conv and silu, fp32 out. ``state`` [B, K - 1, C] fp32 is only read."""

    y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _ext().conv_prefill(x, weight, state, y)
    return y


def rope_decode(x: torch.Tensor, pos: "int | torch.Tensor", rotary: int, theta: float) -> torch.Tensor:
    """Rotate the first ``rotary`` columns of one position: an int, or an int32 device scalar read at run time."""

    y = torch.empty_like(x)
    if isinstance(pos, torch.Tensor):
        _ext().rope_decode(x, y, 0, int(rotary), float(theta), pos)
    else:
        _ext().rope_decode(x, y, int(pos), int(rotary), float(theta), None)
    return y


def moe_router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> None:
    """``out`` [R, E + 1] fp32 = ``x`` [R, D] fp16 or bf16 . ``rows`` [E + 1, D] fp32, a row's bits whatever R."""

    _ext().moe_router(x, rows, out)


def moe_select(logits: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, top_k: int,
               items: torch.Tensor | None = None, members: torch.Tensor | None = None) -> None:
    """The routing rule in one launch; with ``items`` (one row) the plan too: item k is pair k alone."""

    _ext().moe_select(logits, pick, wts, int(top_k), items, members)


def moe_act(both: torch.Tensor, dtype: torch.dtype, limit: float = 0.0) -> torch.Tensor:
    """``both`` [P, 2 NI] fp32 (gate, then up) -> silu(gate) * up [P, NI] in ``dtype``."""

    out = torch.empty((both.shape[0], both.shape[1] // 2), dtype=dtype, device=both.device)
    _ext().moe_act(both, out, float(limit))
    return out


def moe_combine(y: torch.Tensor, wts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``y`` [R, S, D] fp32 times ``wts`` [R, S], the slots summed in order and rounded once to ``dtype``."""

    out = torch.empty((y.shape[0], y.shape[2]), dtype=dtype, device=y.device)
    _ext().moe_combine(y, wts, out)
    return out


def gdn_gate(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """The Gated DeltaNet decay gate and beta in one launch, fp32 in ``a``'s shape."""

    a, b = a.contiguous(), b.contiguous()
    gate = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    _ext().gdn_gate(a, b, a_log.float().contiguous(), dt_bias.float().contiguous(), gate, beta)
    return gate, beta
