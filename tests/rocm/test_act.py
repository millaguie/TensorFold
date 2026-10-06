"""Short-row norm, conv, and RoPE match the PyTorch formulas. Spanning affine codes match the bit reader."""

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.model.checkpoint import _affine_quant  # noqa: E402
from tensorfold.rocm.model.qwen_math import _codes, _rms_torch, apply_rope, causal_conv, rms_norm  # noqa: E402


def _code(row, k, bits):
    bit = k * bits
    word, shift = divmod(bit, 32)
    low = int(row[word].item()) & 0xFFFFFFFF
    high = int(row[word + 1].item()) & 0xFFFFFFFF if shift + bits > 32 and word + 1 < row.numel() else 0
    value = low >> shift
    if shift + bits > 32:
        value |= high << ((32 - shift) & 31)
    return value & ((1 << bits) - 1)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_codes_match_the_bit_reader(bits):
    g = torch.Generator().manual_seed(bits + 20)
    words = torch.randint(-(2**31), 2**31, (4, 40), dtype=torch.int32, generator=g)
    got = _codes(words, bits, 96)
    for row in range(words.shape[0]):
        for k in range(96):
            assert int(got[row, k]) == _code(words[row], k, bits)


def test_affine_quant_accepts_the_mlx_widths():
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            assert _affine_quant({"mode": "affine", "bits": bits, "group_size": group}) == (bits, group)
    with pytest.raises(ValueError):
        _affine_quant({"mode": "affine", "bits": 7, "group_size": 64})


def test_short_rms_matches_the_formula():
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(4, 8, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, generator=g, device="cuda")
    got = rms_norm(x, weight, 1e-6)
    ref = _rms_torch(x, weight, 1e-6)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
    bare = rms_norm(x, None, 1e-6)
    assert torch.allclose(bare, _rms_torch(x, None, 1e-6), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_rows_do_not_depend_on_the_row_count(dtype):
    x = torch.randn(1, 600, 16, 128, device="cuda").to(dtype)
    weight = torch.randn(128, device="cuda")
    whole = rms_norm(x, weight, 1e-6)
    for start, stop in ((0, 1), (0, 16), (16, 22), (22, 600)):
        assert torch.equal(rms_norm(x[:, start:stop].contiguous(), weight, 1e-6), whole[:, start:stop])


def test_decode_conv_matches_the_loop():
    g = torch.Generator(device="cuda").manual_seed(4)
    weight = torch.randn(32, 4, generator=g, device="cuda")
    state = torch.randn(2, 3, 32, generator=g, device="cuda")
    x = torch.randn(2, 1, 32, generator=g, device="cuda", dtype=torch.float16)
    # The host loop is the reference, so run it on a clone before the HIP path updates state.
    host_state = state.clone()
    window = torch.cat((host_state.float(), x.float()), dim=1)
    out = torch.zeros(2, 1, 32, device="cuda")
    for tap in range(4):
        out = out + window[:, tap:tap + 1] * weight[:, tap].view(1, 1, 32)
    ref = torch.nn.functional.silu(out)
    got, new_state = causal_conv(x, weight, state.clone())
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(new_state, window[:, 1:].contiguous(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [1, 37, 128, 256, 300, 512])
def test_short_rows_keep_the_block_kernels_bits(dtype, width):
    """Rows up to 512 wide take one wave a row with the block kernel's lane-order sum, recomputed here up to the last
    bit of torch's rsqrt; a row's bits do not depend on how many rows share the launch."""

    g = torch.Generator(device="cuda").manual_seed(8)
    x = (torch.randn(70, width, generator=g, device="cuda") * 4).to(dtype)
    weight = torch.randn(width, generator=g, device="cuda")
    got = rms_norm(x, weight, 1e-6)
    v = x.float()
    lanes = torch.zeros(70, 32, device="cuda")
    for start in range(0, width, 32):
        part = v[:, start:start + 32]
        lanes[:, :part.shape[1]] += part * part
    for mask in (16, 8, 4, 2, 1):
        lanes = lanes + lanes[:, torch.arange(32, device="cuda") ^ mask]
    inv = torch.rsqrt(lanes[:, :1] / width + 1e-6)
    want = ((v * inv) * weight).to(dtype)
    ulp = {torch.float32: 2.0 ** -23, torch.float16: 2.0 ** -10, torch.bfloat16: 2.0 ** -7}[dtype]
    assert torch.allclose(got.float(), want.float(), rtol=ulp * 2, atol=1e-6)
    for start, stop in ((0, 1), (1, 9), (9, 70)):
        assert torch.equal(rms_norm(x[start:stop].contiguous(), weight, 1e-6), got[start:stop])


def _conv_loop(x, weight, state):
    window = torch.cat((state.float(), x.float()), dim=1)
    out = torch.zeros(x.shape, device="cuda")
    for tap in range(weight.shape[1]):
        out.add_(window[:, tap:tap + x.shape[1]] * weight[:, tap].view(1, 1, -1))
    return torch.nn.functional.silu(out), window[:, x.shape[1]:].contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("length", [2, 3, 37, 600])
def test_prompt_conv_keeps_the_loops_bits(dtype, length):
    g = torch.Generator(device="cuda").manual_seed(6)
    weight = torch.randn(96, 4, generator=g, device="cuda")
    state = torch.randn(2, 3, 96, generator=g, device="cuda") * 8
    x = (torch.randn(2, length, 96, generator=g, device="cuda") * 8).to(dtype)
    ref, ref_state = _conv_loop(x, weight, state)
    got, new_state = causal_conv(x, weight, state.clone(), exact=True)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
    assert torch.equal(new_state, ref_state)
    kept = state.clone()
    got, same = causal_conv(x, weight, kept, exact=True, in_place=True)
    assert same is kept and torch.equal(kept, ref_state)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


def test_prompt_conv_resumed_equals_fresh():
    g = torch.Generator(device="cuda").manual_seed(7)
    weight = torch.randn(64, 4, generator=g, device="cuda")
    x = torch.randn(1, 50, 64, generator=g, device="cuda", dtype=torch.bfloat16)
    whole, _ = causal_conv(x, weight, None, exact=True)
    head, state = causal_conv(x[:, :2], weight, None, exact=True)
    tail, _ = causal_conv(x[:, 2:], weight, state, exact=True)
    assert torch.equal(torch.cat((head, tail), dim=1), whole)


def test_decode_rope_matches_the_formula():
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(2, 4, 1, 64, generator=g, device="cuda", dtype=torch.bfloat16)
    got = apply_rope(x, 17, 10_000_000.0, 32)
    half = 16
    freq = 1.0 / (10_000_000.0 ** (torch.arange(half, device="cuda", dtype=torch.float32) / half))
    ang = 17 * freq
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:32]
    rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos, xf[..., 32:]), dim=-1)
    assert got.dtype == x.dtype
    assert torch.allclose(got, rot.to(dtype=got.dtype), rtol=1e-4, atol=1e-4)


def test_rms_in_the_activation_dtype_matches_fp32_then_cast():
    """FP16 and BF16 rows give the bits of the fp32 kernel followed by one cast, which is what the forward ran."""

    from tensorfold.rocm.kernels.act import rms

    g = torch.Generator(device="cuda").manual_seed(9)
    weight = torch.randn(5120, generator=g, device="cuda")
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn(3, 5120, generator=g, device="cuda").to(dtype)
        assert torch.equal(rms(x, weight, 1e-6), rms(x.float(), weight, 1e-6).to(dtype))
        assert torch.equal(rms(x, None, 1e-6), rms(x.float(), None, 1e-6).to(dtype))


@pytest.mark.parametrize("width", [128, 100, 256, 512])
def test_gated_norm_keeps_the_torch_ops_bits(width):
    """rms_norm(y) * silu(z) in one kernel: the norm's, silu's, the multiply's and the cast's roundings."""

    from tensorfold.rocm.model.forward import _gated_norm

    g = torch.Generator(device="cuda").manual_seed(width)
    y = torch.randn(3, 5, 32, width, generator=g, device="cuda") * 3
    z = (torch.randn(3, 5, 32, width, generator=g, device="cuda") * 6).to(torch.bfloat16)
    weight = torch.randn(width, generator=g, device="cuda")
    want = (rms_norm(y, weight, 1e-6) * torch.nn.functional.silu(z).float()).to(torch.bfloat16)
    got = _gated_norm(y, z, weight, 1e-6, torch.bfloat16)
    assert got.dtype == torch.bfloat16 and got.shape == z.shape
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
