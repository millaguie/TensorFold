"""RDNA4's MXFP4 matmuls (mx4_rocm.cu): exact on their inputs, row- and chunk-invariant, stacked views equal members."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import gfx12  # noqa: E402

if not gfx12():
    pytest.skip("mx4_rocm.cu: RDNA4's WMMA", allow_module_level=True)

from tensorfold.cuda.kernels import mx4 as kmx  # noqa: E402
from tensorfold.families.qwen3_5.cuda.mx4_load import Mx4, stack  # noqa: E402


def _weight(n: int, k: int, seed: int, spread: int = 6) -> Mx4:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(0, 256, (n, k // 2), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    scales = torch.randint(120 - spread, 121, (n, k // 32), generator=gen, device="cuda", dtype=torch.int32)
    return Mx4.from_checkpoint(words, scales.to(torch.uint8))


def _fragments(x8: torch.Tensor) -> torch.Tensor:
    m, k = x8.shape
    mt = (m + 15) // 16
    padded = torch.zeros((mt * 16, k), dtype=torch.uint8, device=x8.device)
    padded[:m] = x8
    return padded.view(mt, 16, k // 16, 2, 8).permute(0, 2, 3, 1, 4).reshape(mt, k // 16, 256).contiguous()


def _rows(m: int, k: int, seed: int):
    """e4m3 prompt rows in prefill_glue's stored order (fragments), their row scales and their values by input."""

    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=gen, device="cuda")
    a = x.abs().amax(1) / 448.0
    x8 = (x / a[:, None]).to(torch.float8_e4m3fn)
    i = torch.arange(k, device="cuda")
    src = (i // 32) * 32 + ((i % 32) // 16) * 16 + ((i % 16) // 4) * 2 + (i % 4 % 2) + (i % 4 // 2) * 8
    return x8[:, src].contiguous().view(torch.uint8), a.contiguous(), x8.double() * a.double()[:, None]


@pytest.mark.parametrize("n,k", [(48, 5120), (1024, 5120), (17408, 5120), (5120, 17408), (5120, 6144)])
def test_prompt_matmul_is_exact_and_chunk_invariant(n, k):
    q = _weight(n, k, n + k)
    stored, a, values = _rows(300, k, n)
    whole = kmx.prompt((_fragments(stored), None, a), q.tiles, q.scales_t, q.ref, n, f32=True)
    ref = values @ q.dequantize().double().t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 1e-5     # every shift here is within e4m3's reach
    for size in (1, 7, 16, 100):
        parts = [kmx.prompt((_fragments(stored[s:s + size]), None, a[s:s + size].contiguous()), q.tiles, q.scales_t,
                            q.ref, n, f32=True) for s in range(0, 300, size)]
        assert torch.equal(whole, torch.cat(parts)), size
    for tn in (2, 4):                                  # tile widths: speed only
        assert torch.equal(kmx.prompt((_fragments(stored), None, a), q.tiles, q.scales_t, q.ref, n, f32=True, tn=tn),
                           whole)


@pytest.mark.parametrize("n,k", [(48, 5120), (1024, 5120), (17408, 5120), (5120, 17408), (5120, 6144)])
def test_decode_rows_are_exact_and_row_count_invariant(n, k):
    q = _weight(n, k, 2 * n + k, spread=20)               # bf16 widening is exact at any shift
    x = torch.randn(60, k, device="cuda").bfloat16()
    whole = kmx.decode(x, q.tiles, q.scales_t, q.ref, n, f32=True)
    ref = x.double() @ q.dequantize().double().t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 1e-5   # fp32 sums over K
    for m in (1, 3, 12, 16, 17, 32, 33, 40):
        assert torch.equal(kmx.decode(x[:m], q.tiles, q.scales_t, q.ref, n, f32=True), whole[:m]), m


def test_stacked_views_give_their_members_bits():
    parts = [_weight(n, 5120, 7 + n) for n in (1024, 512, 48, 48)]
    alone = [(p.tiles.clone(), p.scales_t.clone(), p.ref.clone(), p.n) for p in parts]
    whole, views = stack(parts)
    x = torch.randn(12, 5120, device="cuda").bfloat16()
    stacked = kmx.decode(x, whole.tiles, whole.scales_t, whole.ref, whole.n, f32=True)
    stored, a, _ = _rows(100, 5120, 3)
    frag = (_fragments(stored), None, a)
    stacked8 = kmx.prompt(frag, whole.tiles, whole.scales_t, whole.ref, whole.n, f32=True)
    col = 0
    for (tiles, scales_t, ref, n), view in zip(alone, views):
        own = kmx.decode(x, tiles, scales_t, ref, n, f32=True)
        assert torch.equal(own, stacked[:, col:col + n])
        assert torch.equal(own, kmx.decode(x, view.tiles, view.scales_t, view.ref, n, f32=True))
        own8 = kmx.prompt(frag, tiles, scales_t, ref, n, f32=True)
        assert torch.equal(own8, stacked8[:, col:col + n])
        assert torch.equal(own8, kmx.prompt(frag, view.tiles, view.scales_t, view.ref, n, f32=True))
        col += n


def test_bf16_head_rows_are_exact_and_row_count_invariant():
    w = torch.randn(4096, 5120, device="cuda").bfloat16()
    x = torch.randn(50, 5120, device="cuda").bfloat16()
    whole = kmx.b16(x, w, f32=True)
    ref = x.double() @ w.double().t()
    assert ((whole.double() - ref).norm() / ref.norm()).item() < 1e-5   # fp32 sums over K
    for m in (1, 4, 12, 17, 32, 33, 48):
        assert torch.equal(kmx.b16(x[:m], w, f32=True), whole[:m]), m


def test_host_table_rows_are_the_stored_rows():
    table = torch.randn(5000, 5120).bfloat16().pin_memory()
    ids = torch.tensor([[0, 4999, 17], [17, 3, 2500]], device="cuda")
    got = kmx.host_rows(ids, table)
    assert got.shape == (2, 3, 5120) and got.is_cuda
    assert torch.equal(got.cpu(), table[ids.cpu()])
    assert kmx.host_rows(torch.empty(0, dtype=torch.int64, device="cuda"), table).shape == (0, 5120)
