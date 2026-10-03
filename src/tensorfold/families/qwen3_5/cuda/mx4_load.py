"""A Quark MXFP4 checkpoint of a Qwen3.5/3.8 dense model (``quant_method: quark``, fp4 weights in groups of 32 with
e8m0 scales, the format vLLM-radiance serves on RDNA4) into ``Weights``: every projection an ``Mx4``.

The checkpoint is a Hugging Face export, so two things differ from the MLX conversions the rest of the family reads:
  * RMSNorm weights are stored as offsets (the model multiplies by 1 + w); MLX stores 1 + w. The input, post-attention,
    q/k and final norms get their +1 here in fp32, as MLX's conversion does. The GDN gated norm is not an offset.
  * conv1d is (C, 1, kernel), reshaped to (C, kernel).
The vision tower and the MTP head are not read (DFlash2 drafts; image input is not served on ROCm).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .weights import GDN, Attention, Config, Layer, Plain, Weights

# e2m1 codes 0-15 (bit 3 the sign): 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
GROUP = 32


def quark_mxfp4(model_dir: str | Path) -> bool:
    """Whether ``model_dir`` holds a Quark MXFP4 export (fp4 weights, e8m0 scales, groups of 32)."""

    path = Path(model_dir) / "config.json"
    if not path.exists():
        return False
    q = json.loads(path.read_text()).get("quantization_config") or {}
    w = (q.get("global_quant_config") or {}).get("weight") or {}
    return (q.get("quant_method") == "quark" and w.get("dtype") == "fp4" and w.get("scale_format") == "e8m0"
            and w.get("group_size") == GROUP)


@dataclass
class Mx4:
    """An MXFP4 projection: two e2m1 codes a byte (low nibble the even input), an e8m0 scale per 32 inputs, kept in
    the layout ``tensorfold.cuda.kernels.mx4`` reads: a group's 16 bytes for 16 columns side by side, the scales by
    group, and each column's reference exponent (its largest scale)."""

    tiles: torch.Tensor       # (N / 16, K / 32, 16, 16) uint8
    scales_t: torch.Tensor    # (K / 32, N) uint8: the group's scale is 2^(e - 127)
    ref: torch.Tensor         # (N,) int32
    layout: str = "mx4"

    @classmethod
    def from_checkpoint(cls, weight: torch.Tensor, scale: torch.Tensor) -> "Mx4":
        from tensorfold.cuda.kernels.mx4 import to_tiles

        return cls(to_tiles(weight), scale.t().contiguous(), scale.max(1).values.to(torch.int32).contiguous())

    @property
    def n(self) -> int:
        return int(self.scales_t.shape[1])

    @property
    def k(self) -> int:
        return int(self.scales_t.shape[0]) * GROUP

    def nbytes(self) -> int:
        return self.tiles.numel() + self.scales_t.numel() + self.ref.numel() * 4

    def stored(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The checkpoint's (N, K / 2) bytes and (N, K / 32) scales again."""

        from tensorfold.cuda.kernels.mx4 import from_tiles

        return from_tiles(self.tiles), self.scales_t.t()

    def rows(self, index: torch.Tensor) -> "Mx4":
        weight, scale = self.stored()
        return Mx4.from_checkpoint(weight.index_select(0, index), scale.index_select(0, index))

    def dequantize(self) -> torch.Tensor:
        """The (N, K) weight in bf16: every e2m1 value times a power of two is exact there."""

        weight, scale = self.stored()
        lut = torch.tensor(E2M1, dtype=torch.float32, device=weight.device)
        codes = torch.stack([weight & 0x0F, weight >> 4], -1).reshape(self.n, self.k)
        return (lut[codes.long()] * torch.exp2(scale.float() - 127.0).repeat_interleave(GROUP, 1)).to(torch.bfloat16)

    def prefill8(self, x: tuple) -> torch.Tensor:
        """FP8 prompt fragments on RDNA4's folded-scale kernel (W4A8, as vLLM-radiance serves this checkpoint)."""

        from tensorfold.cuda.kernels.mx4 import prompt

        return prompt(x, self.tiles, self.scales_t, self.ref, self.n)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Decode and verify rows on RDNA4's MXFP4 kernel: the same row bits at any row count."""

        from tensorfold.cuda.kernels.mx4 import decode

        return decode(x, self.tiles, self.scales_t, self.ref, self.n)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The bf16 weight through torch's matmul (not row-count invariant)."""

        return (x.to(torch.bfloat16) @ self.dequantize().t()).to(torch.bfloat16)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


def _offset_norm(t: torch.Tensor) -> torch.Tensor:
    return (t.float() + 1.0).to(torch.bfloat16)


def load_mx4(model_dir: str | Path, device: str = "cuda") -> Weights:
    from safetensors import safe_open

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    if cfg.experts:
        raise ValueError("MXFP4 loading covers the dense Qwen3.5/3.8 models")
    files = sorted(model_dir.glob("*.safetensors"))
    handles = {}
    for f in files:
        h = safe_open(str(f), "pt", device="cpu")
        for k in h.keys():
            handles[k] = h
    used: set[str] = set()
    lm = "model.language_model."

    def get(name: str) -> torch.Tensor:
        used.add(name)
        return handles[name].get_tensor(name)

    def mx(name: str) -> Mx4:
        w, s = get(name + ".weight"), get(name + ".weight_scale")
        if w.dtype != torch.uint8 or s.dtype != torch.uint8 or s.shape[1] * GROUP != w.shape[1] * 2:
            raise ValueError(f"{name}: expected (N, K / 2) fp4 bytes with (N, K / 32) e8m0 scales")
        return Mx4.from_checkpoint(w.to(device), s.to(device))

    def dense(name: str) -> torch.Tensor:
        return get(name).contiguous().to(device)

    layers = []
    for i in range(cfg.layers):
        p = f"{lm}layers.{i}."
        gdn = attn = None
        if cfg.is_linear(i):
            gdn = GDN(qkv=mx(p + "linear_attn.in_proj_qkv"), z=mx(p + "linear_attn.in_proj_z"),
                      b=mx(p + "linear_attn.in_proj_b"), a=mx(p + "linear_attn.in_proj_a"),
                      out=mx(p + "linear_attn.out_proj"),
                      conv=dense(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                      A_log=dense(p + "linear_attn.A_log").float().contiguous(),
                      dt_bias=dense(p + "linear_attn.dt_bias").float().contiguous(),
                      norm=dense(p + "linear_attn.norm.weight"))
        else:
            attn = Attention(q=mx(p + "self_attn.q_proj"), k=mx(p + "self_attn.k_proj"), v=mx(p + "self_attn.v_proj"),
                             o=mx(p + "self_attn.o_proj"),
                             q_norm=_offset_norm(dense(p + "self_attn.q_norm.weight")),
                             k_norm=_offset_norm(dense(p + "self_attn.k_norm.weight")))
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=_offset_norm(dense(p + "input_layernorm.weight")),
                            post_norm=_offset_norm(dense(p + "post_attention_layernorm.weight")), gdn=gdn, attn=attn,
                            gate=mx(p + "mlp.gate_proj"), up=mx(p + "mlp.up_proj"), down=mx(p + "mlp.down_proj")))
    w = Weights(config=cfg, embed=Plain(dense(lm + "embed_tokens.weight").to(torch.bfloat16)), layers=layers,
                norm=_offset_norm(dense(lm + "norm.weight")), head=Plain(dense("lm_head.weight").to(torch.bfloat16)),
                quant="mx4")
    half = cfg.rope_dims // 2
    inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    w.inv_freq = inv.to(torch.float32).to(device)
    left = [k for k in handles if k not in used and not k.startswith(("model.visual.", "mtp."))]
    if left:
        raise ValueError(f"unused MXFP4 checkpoint tensors: {left[:5]} ...")
    return w


__all__ = ["E2M1", "GROUP", "Mx4", "load_mx4", "quark_mxfp4"]
