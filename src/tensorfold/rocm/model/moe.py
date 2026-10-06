"""Softmax top-k MoE with a sigmoid-gated shared expert: the CUDA routing rule on RDNA."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tensorfold.rocm.kernels import act
from tensorfold.rocm.model import experts as grouped


def router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> None:
    """``out`` [R, E + 1] fp32 = ``x`` [R, D] . ``rows`` [E + 1, D]; a row's logits do not depend on R."""

    act.moe_router(x, rows, out)


def select_rows(logits: torch.Tensor, buf: "MoEBuffers", top_k: int, experts: int) -> None:
    """Each row's top-k experts and weights by the CUDA rule; slot ``top_k`` is the shared expert."""

    if logits.shape[1] != experts + 1:
        raise ValueError(f"logits carry {logits.shape[1]} columns, want {experts + 1}")
    act.moe_select(logits, buf.pick, buf.wts, top_k)


def select(logits: torch.Tensor, buf: "MoEBuffers", top_k: int, experts: int,
           tile: int = grouped.PREFILL_TILE) -> None:
    """Each row's experts and weights, then the pairs grouped by expert, ``tile`` pairs an item at most."""

    select_rows(logits, buf, top_k, experts)
    grouped.route(buf.pick[: logits.shape[0]], buf.plan, tile)


def _plan(x: torch.Tensor, ex: grouped.Experts, buf: "MoEBuffers", top_k: int, experts: int,
          prefill: bool = False) -> None:
    """Pick each row's experts and group the pairs. One row on the routed kernels: the pick writes the plan."""

    rows = x.shape[0]
    if rows == 1 and grouped.one_launch(ex, x):
        act.moe_select(buf.logits[:1], buf.pick, buf.wts, top_k, buf.plan.items, buf.plan.members)
        buf.plan.tile, buf.plan.count = 1, buf.slots
        return
    select_rows(buf.logits[:rows], buf, top_k, experts)
    grouped.route(buf.pick[:rows], buf.plan, grouped.tile_for(ex, rows, prefill))


def moe(x: torch.Tensor, rows32: torch.Tensor, ex: grouped.Experts, buf: "MoEBuffers", top_k: int,
        experts: int, *, prefill: bool = False) -> torch.Tensor:
    """Route ``x`` [R, D] and run its experts: [R, k + 1, D] fp32, slot k the shared expert."""

    rows = x.shape[0]
    router(x, rows32, buf.logits[:rows])
    _plan(x, ex, buf, top_k, experts, prefill=prefill)
    hidden = grouped.gate_up(x, ex, buf.plan, rows)
    return grouped.down(hidden, ex, buf.plan, rows).view(rows, buf.slots, ex.dims)


def combine(y: torch.Tensor, wts: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``y`` [R, S, D] fp32 and ``wts`` [R, S] fp32 -> [R, D] ``dtype``, the slots summed in order, rounded once."""

    return act.moe_combine(y.contiguous(), wts.contiguous(), dtype)


@dataclass
class MoEBuffers:
    """Static scratch for up to ``rows`` rows; ``prefill`` picks the experts' prompt arithmetic."""

    rows: int
    slots: int
    logits: torch.Tensor
    pick: torch.Tensor
    wts: torch.Tensor
    plan: grouped.Plan

    def __init__(self, rows: int, cfg, device: torch.device | str, *, prefill: bool = False) -> None:
        # Not inference tensors: these buffers outlive the inference_mode block that makes them.
        with torch.inference_mode(False):
            slots = cfg.num_experts_per_tok + 1
            self.rows, self.slots = rows, slots
            self.logits = torch.empty((rows, cfg.num_experts + 1), dtype=torch.float32, device=device)
            self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
            self.wts = torch.empty((rows, slots), dtype=torch.float32, device=device)
            self.plan = grouped.Plan(rows, slots, cfg.num_experts + 1, device, prefill=prefill)


@dataclass
class Routed:
    """A layer's router rows (shared gate last) and experts; ``remap`` and ``partial`` mark a tp rank's share."""

    router: torch.Tensor
    experts: grouped.Experts
    top_k: int
    remap: torch.Tensor | None = None
    partial: bool = False
    rows32: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The router kernel reads a bf16 router as stored (it widens exactly) at the widths it tiles; others widen
        # once here. Same fp32 values either way.
        if self.router.dtype == torch.bfloat16 and self.router.shape[1] in (1024, 2048, 4096):
            self.rows32 = self.router.contiguous()
        else:
            self.rows32 = self.router.float().contiguous()

    @property
    def count(self) -> int:
        return self.router.shape[0] - 1


class _Shape:
    def __init__(self, m: Routed) -> None:
        self.num_experts_per_tok, self.num_experts = m.top_k, m.count
        self.moe_intermediate_size, self.hidden_size = m.experts.width, m.experts.dims


_scratch: dict[tuple, MoEBuffers] = {}


def run(x: torch.Tensor, m: Routed, *, prefill: bool = False) -> torch.Tensor:
    """``x`` [R, D] -> [R, D]: the top-k experts plus the shared one; a tp rank returns its fp32 share."""

    rows = x.shape[0]
    size = 1 << max(4, (rows - 1).bit_length())
    key = (size, m.count, m.top_k, m.experts.width, m.experts.dims, prefill, x.device)
    buf = _scratch.get(key)
    if buf is None:
        buf = _scratch[key] = MoEBuffers(size, _Shape(m), x.device, prefill=prefill)
    if m.remap is None:
        y = moe(x.contiguous(), m.rows32, m.experts, buf, m.top_k, m.count, prefill=prefill)
        return combine(y, buf.wts[:rows], x.dtype)
    return _run_share(x.contiguous(), m, buf, prefill)


def _run_share(x: torch.Tensor, m: Routed, buf: MoEBuffers, prefill: bool) -> torch.Tensor:
    """One rank's experts: every rank picks the same pairs, runs the ones it holds and sums them in slot order."""

    rows = x.shape[0]
    router(x, m.rows32, buf.logits[:rows])
    select_rows(buf.logits[:rows], buf, m.top_k, m.count)
    local = m.remap[buf.pick[:rows].long()]
    mine = local >= 0
    skip = m.experts.count                         # an id past the rank's experts: its items do no work
    picks = torch.where(mine, local, torch.full_like(local, skip)).to(torch.int32).contiguous()
    grouped.route(picks, buf.plan, grouped.tile_for(m.experts, rows, prefill))
    items = buf.plan.items[:buf.plan.count]
    items[:, 2] = torch.where(items[:, 0] == skip, 0, items[:, 2])     # no mask write: no host sync in a graph
    hidden = grouped.gate_up(x, m.experts, buf.plan, rows)
    y = grouped.down(hidden, m.experts, buf.plan, rows).view(rows, buf.slots, -1)
    y = torch.where(mine.unsqueeze(-1), y, torch.zeros((), dtype=y.dtype, device=y.device))
    return combine(y, buf.wts[:rows], torch.float32)
