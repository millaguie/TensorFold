"""Qwen3.5 / Qwen3.6 text model on RDNA: the projection dispatch, and the model's load and tp entry points."""

from __future__ import annotations

import torch

from tensorfold.rocm.model import qwen_math
from tensorfold.rocm.model.checkpoint import load, load_mtp_head  # noqa: F401 - the package's entry points
from tensorfold.rocm.model.forward import greedy
from tensorfold.rocm.model.model import FullLayer, LinearLayer, MTPHead, TextModel  # noqa: F401
from tensorfold.rocm.model.qwen_math import Dense, GptqPacked, Packed, Spec  # noqa: F401 - Spec for callers
from tensorfold.rocm.model.slicing import slice_for_tp  # noqa: F401

_RDNA2 = {f"gfx103{i}" for i in range(7)}


def activation_dtype(gfx: str) -> torch.dtype:
    """FP16 on RDNA2, where the affine schedule is the FP16 dot. BF16 on an RDNA3 WMMA part."""

    from tensorfold.rocm.kernels.build import WMMA

    if gfx in _RDNA2:
        return torch.float16
    if gfx in WMMA:
        return torch.bfloat16
    raise RuntimeError(f"no activation dtype for {gfx}")


class Engine:
    """The forward's projections: ``schedule`` picks the affine kernel, ``rccl`` the tp ring."""

    def __init__(self, model: TextModel, schedule: str = "auto", dtype: torch.dtype | None = None, rccl=None):
        self.model = model
        self.schedule = schedule
        self.dtype = dtype
        self.rccl = rccl
        self.projections = 0

    def linear(self, flat: torch.Tensor, packed: Packed) -> torch.Tensor:
        from tensorfold.rocm.kernels import affine as affine_mod

        if self.dtype is None:
            from tensorfold.rocm.kernels.build import gfx_name

            self.dtype = activation_dtype(gfx_name())
        flat = flat.reshape(-1, flat.shape[-1]).contiguous()
        if isinstance(packed, GptqPacked):
            return self._gptq(flat, packed)
        if isinstance(packed, Dense):
            return torch.nn.functional.linear(flat.float(), packed.weight).to(self.dtype)
        flat = flat.to(dtype=self.dtype)
        self._note(flat, packed)
        words, scale, bias = packed.words, packed.scale, packed.bias
        schedule = self.schedule
        # A tensor-parallel share along K stays fp32 until the ranks are summed.
        kwargs = {"bits": packed.bits, "group": packed.group, "schedule": schedule, "f32": packed.partial}
        span = qwen_math.SPAN
        if flat.shape[0] <= span:
            return affine_mod.matmul(flat, words, scale, bias, **kwargs)
        # One output buffer. Keeping every chunk and then concatenating doubles a long prefill.
        out = torch.empty(flat.shape[0], words.shape[0], dtype=torch.float32 if packed.partial else flat.dtype,
                          device=flat.device)
        for start in range(0, flat.shape[0], span):
            stop = min(start + span, flat.shape[0])
            out[start:stop] = affine_mod.matmul(flat[start:stop], words, scale, bias, **kwargs)
        return out

    def linear_pair(self, x: torch.Tensor, first: Packed, second: Packed):
        """Gate and up, or k and v: one shared activation load when both widths match."""

        if not isinstance(first, Packed) or not isinstance(second, Packed):
            return self.linear(x, first), self.linear(x, second)
        from tensorfold.rocm.kernels import affine as affine_mod
        from tensorfold.rocm.kernels.build import WMMA, gfx_name

        if self.dtype is None:
            self.dtype = activation_dtype(gfx_name())
        flat = x.reshape(-1, x.shape[-1]).to(dtype=self.dtype).contiguous()
        same = (first.words.shape[0] == second.words.shape[0] and first.bits == second.bits == 8
                and first.group == second.group and self.schedule == "wmma" and self.dtype == torch.bfloat16
                and gfx_name() in WMMA)
        if not same:
            return self.linear(flat, first), self.linear(flat, second)
        self._note(flat, first)
        self._note(flat, second)
        left, right = affine_mod.matmul_pair(
            flat, first.words, first.scale, first.bias, second.words, second.scale, second.bias,
            bits=first.bits, group=first.group)
        return left, right

    def linear_group(self, x: torch.Tensor, packeds: tuple):
        """Several projections of one short activation. None means the caller uses solo or pair launches."""

        if not all(isinstance(p, Packed) for p in packeds):
            return None
        from tensorfold.rocm.kernels import affine as affine_mod
        from tensorfold.rocm.kernels.build import WMMA, gfx_name

        if self.dtype is None:
            self.dtype = activation_dtype(gfx_name())
        flat = x.reshape(-1, x.shape[-1]).to(dtype=self.dtype).contiguous()
        # A decode step's projections take one launch of the row tile, each with its solo launch's bits.
        if (2 <= len(packeds) <= 4 and flat.shape[0] <= 2 and self.schedule == "auto"
                and self.dtype == torch.bfloat16 and gfx_name() in WMMA
                and all(not p.partial and p.bits == packeds[0].bits and p.group == packeds[0].group
                        for p in packeds)):
            outs = affine_mod.matmul_rows(flat, tuple((p.words, p.scale, p.bias) for p in packeds),
                                          bits=packeds[0].bits, group=packeds[0].group)
            if outs is not None:
                for packed in packeds:
                    self._note(flat, packed)
                return outs
        same = (2 <= len(packeds) <= 4 and flat.shape[0] <= 16 and self.schedule == "wmma"
                and self.dtype == torch.bfloat16 and gfx_name() in WMMA
                and all(p.bits == 8 and p.group == packeds[0].group for p in packeds))
        if not same:
            return None
        for packed in packeds:
            self._note(flat, packed)
        return affine_mod.matmul_group(
            flat, tuple((p.words, p.scale, p.bias) for p in packeds), bits=8, group=packeds[0].group)

    def _note(self, flat: torch.Tensor, packed: Packed) -> None:
        words = packed.words
        k = flat.shape[-1]
        expect = k * packed.bits // 32
        if words.dtype != torch.int32 or words.ndim != 2 or words.shape[1] != expect:
            raise RuntimeError("a projection handed the matmul a weight that is not packed int32 words")
        self.projections += 1

    def _gptq(self, flat: torch.Tensor, packed: GptqPacked) -> torch.Tensor:
        """A W4A16 GPTQ projection: the RDNA2 fp16 dot, tiled by the prefill row counts."""

        from tensorfold.rocm.kernels import qgemm

        if packed.g_idx is not None:
            raise ValueError("the RDNA W4A16 path does not carry an act-order permutation yet")
        x = flat.to(dtype=torch.float16).contiguous()
        qweight, qzeros, scales = packed.qweight, packed.qzeros, packed.scales
        span = qwen_math.SPAN
        if x.shape[0] <= span:
            return qgemm.matmul(x, qweight, qzeros, scales, use_v2_format=packed.v2, prefill=x.shape[0] > 16)
        out = torch.empty(x.shape[0], qweight.shape[1], dtype=torch.float16, device=x.device)
        for start in range(0, x.shape[0], span):
            stop = min(start + span, x.shape[0])
            out[start:stop] = qgemm.matmul(x[start:stop], qweight, qzeros, scales, use_v2_format=packed.v2,
                                           prefill=True)
        return out

    def generate(self, prompts: list[list[int]], n_new: int, after_token=None) -> list[list[int]]:
        device = self.model.embed.words.device
        if self.dtype is None:
            from tensorfold.rocm.kernels.build import gfx_name

            self.dtype = activation_dtype(gfx_name())
        hooks = {}
        if self.rccl is not None and self.rccl.world > 1:
            from functools import partial

            from tensorfold.rocm.model.qwen_tp import all_reduce_local, vocab_gather

            hooks = {"reduce": partial(all_reduce_local, self.rccl), "gather": partial(vocab_gather, self.rccl)}
        with torch.inference_mode():
            return greedy(self.model, prompts, n_new, self.linear, device, after_token=after_token,
                          cache_dtype=self.dtype, **hooks)


