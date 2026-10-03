"""DFlash2's shared group sums (``_kv_sums``): a packed [k | v] matmul given them has the bits it has computing its own."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.dflash2 import quantize4  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm import group_sums  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import matmul, tile  # noqa: E402


@pytest.mark.parametrize("rows", [1, 3, 12, 16, 40])
def test_a_projection_given_the_shared_sums_keeps_its_bits(rows):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    q = tile(quantize4(torch.randn(2048, 5120, generator=gen, device="cuda").bfloat16() / 64))
    x = torch.randn(rows, 5120, generator=gen, device="cuda").bfloat16()
    shared = group_sums(x)                       # once for every layer's projection, as add_taps does
    assert torch.equal(matmul(x, q, shared), matmul(x, q))
    other = tile(quantize4(torch.randn(2048, 5120, generator=gen, device="cuda").bfloat16() / 64))
    assert torch.equal(matmul(x, other, shared), matmul(x, other))      # a second layer reuses them
