"""Packed affine on RDNA: a row's bits do not depend on how many rows share the launch."""

import ctypes
import ctypes.util

import numpy as np
import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    pytest.skip("RDNA only", allow_module_level=True)

from tensorfold.rocm.kernels.affine import matmul, matmul_group, matmul_pair  # noqa: E402
from tensorfold.rocm.kernels.build import WMMA, gfx_name  # noqa: E402

ROWS = [1, 2, 7, 15, 16, 17, 31, 32]

_FMAF = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6").fmaf
_FMAF.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_FMAF.restype = ctypes.c_float


def _fmaf(a, b, c):
    """float32 fused multiply-add. torch's addcmul is a separate multiply and add, so it is not this rounding."""

    a, b, c = np.broadcast_arrays(np.asarray(a, np.float32), np.asarray(b, np.float32), np.asarray(c, np.float32))
    fa = np.ascontiguousarray(a, np.float32).reshape(-1)
    fb = np.ascontiguousarray(b, np.float32).reshape(-1)
    fc = np.ascontiguousarray(c, np.float32).reshape(-1)
    out = np.empty(fa.shape, np.float32)
    fmaf = _FMAF
    for i in range(out.shape[0]):
        out[i] = fmaf(fa[i], fb[i], fc[i])
    return out.reshape(a.shape)


def _code(row, k, bits):
    bit = k * bits
    word, shift = divmod(bit, 32)
    low = int(row[word].item()) & 0xFFFFFFFF
    high = int(row[word + 1].item()) & 0xFFFFFFFF if shift + bits > 32 and word + 1 < row.numel() else 0
    value = low >> shift
    if shift + bits > 32:
        value |= high << ((32 - shift) & 31)
    return value & ((1 << bits) - 1)


def _pack(n, k, bits, group, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    codes = torch.randint(0, 1 << bits, (n, k), generator=g)
    words = torch.zeros((n, k * bits // 32), dtype=torch.int64)
    for col in range(k):
        value = (codes[:, col] & ((1 << bits) - 1)).to(torch.int64)
        bit = col * bits
        word, shift = divmod(bit, 32)
        words[:, word] |= value << shift
        if shift + bits > 32:
            words[:, word + 1] |= value >> (32 - shift)
    scale = torch.rand((n, k // group), generator=g) * 0.2 + 0.02
    bias = torch.randn((n, k // group), generator=g) * 0.05
    return codes, words.to(torch.int32), scale, bias


def _reference(x, codes, scale, bias, group):
    """The GEMV formula: BF16 rounding of each code, then one product at a time, then the group scale and bias."""

    q = codes.to(torch.float32).to(torch.bfloat16).to(torch.float32).cpu().numpy()
    xf = x.float().cpu().numpy()
    sc = scale.float().cpu().numpy()
    bi = bias.float().cpu().numpy()
    acc = np.zeros((xf.shape[0], q.shape[0]), np.float32)
    for start in range(0, xf.shape[1], group):
        dot = np.zeros((xf.shape[0], q.shape[0]), np.float32)
        summed = np.zeros((xf.shape[0],), np.float32)
        for t in range(group):
            xv = xf[:, start + t]
            dot = _fmaf(xv[:, None], q[:, start + t][None, :], dot)
            summed = np.add(summed, xv, dtype=np.float32)
        g = start // group
        acc = _fmaf(dot, sc[:, g][None, :], acc)
        acc = _fmaf(summed[:, None], bi[:, g][None, :], acc)
    return torch.from_numpy(np.ascontiguousarray(acc))


def _skip_wmma(schedule):
    if schedule == "wmma" and gfx_name() not in WMMA:
        pytest.skip("this RDNA part has no WMMA; auto stays on the GEMV")


def test_visible_device_when_pinned():
    import os
    want = os.environ.get("EXPECT_BUS")
    if not want:
        return
    bus = getattr(torch.cuda.get_device_properties(0), "pci_bus_id", None)
    # hip properties expose the bus as an int on ROCm builds; skip the compare when this torch does not.
    if isinstance(bus, int):
        assert f"{bus:02x}" == want.lower(), f"visible bus {bus:02x}, expected {want}"


def test_pack_round_trip():
    codes, words, _, _ = _pack(4, 96, 3, 32, 4)
    for n in range(codes.shape[0]):
        for k in range(codes.shape[1]):
            assert _code(words[n], k, 3) == int(codes[n, k])


@pytest.mark.parametrize("bits,group,k", [(8, 64, 128), (4, 128, 256), (3, 32, 96), (5, 32, 128)])
def test_gemv_matches_the_formula(bits, group, k):
    codes, words, scale, bias = _pack(40, k, bits, group, 10 + bits)
    g = torch.Generator(device="cuda").manual_seed(10 + bits)
    x = torch.randn((8, k), generator=g, device="cuda", dtype=torch.bfloat16)
    got = matmul(x, words.cuda(), scale.cuda(), bias.cuda(), bits=bits, group=group, schedule="gemv", f32=True)
    assert torch.equal(got.cpu(), _reference(x, codes, scale, bias, group))


@pytest.mark.parametrize("schedule", ["auto", "gemv", "wmma"])
def test_rows_do_not_depend_on_row_count(schedule):
    _skip_wmma(schedule)
    _, words, scale, bias = _pack(48, 192, 8, 64, 3)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((max(ROWS), 192), generator=g, device="cuda", dtype=torch.bfloat16)
    alone = torch.cat([matmul(x[r:r + 1], words, scale, bias, bits=8, group=64, schedule=schedule, f32=True)
                       for r in range(max(ROWS))])
    for m in ROWS:
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=8, group=64, schedule=schedule, f32=True), alone[:m])
    perm = torch.randperm(max(ROWS), device="cuda")
    assert torch.equal(matmul(x[perm], words, scale, bias, bits=8, group=64, schedule=schedule, f32=True), alone[perm])


@pytest.mark.parametrize("schedule", ["auto", "gemv", "wmma"])
def test_word_spanning_rows_do_not_depend_on_row_count(schedule):
    """3-bit codes cross the 32-bit word. The same row in a wide launch matches the row alone."""

    _skip_wmma(schedule)
    _, words, scale, bias = _pack(32, 96, 3, 32, 9)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    x = torch.randn((17, 96), device="cuda", dtype=torch.bfloat16)
    alone = matmul(x[:1], words, scale, bias, bits=3, group=32, schedule=schedule, f32=True)
    assert torch.equal(matmul(x, words, scale, bias, bits=3, group=32, schedule=schedule, f32=True)[:1], alone)


def test_fp16_is_the_rdna2_schedule():
    """A WMMA part keeps one BF16 formula. RDNA2 FP16 auto matches its gemv and does not depend on M."""

    _, words, scale, bias = _pack(48, 192, 8, 64, 11)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn((max(ROWS), 192), generator=g, device="cuda", dtype=torch.float16)
    if gfx_name() in WMMA:
        with pytest.raises(ValueError, match="RDNA2"):
            matmul(x[:1], words, scale, bias, bits=8, group=64)
        return
    with pytest.raises(RuntimeError):
        matmul(x[:1], words, scale, bias, bits=8, group=64, schedule="wmma", f32=True)
    stored = matmul(x[:1], words, scale, bias, bits=8, group=64)
    assert stored.dtype == torch.float16
    alone = torch.cat([matmul(x[r:r + 1], words, scale, bias, bits=8, group=64, schedule="auto", f32=True)
                       for r in range(max(ROWS))])
    assert not torch.equal(alone, torch.zeros_like(alone))
    gemv = torch.cat([matmul(x[r:r + 1], words, scale, bias, bits=8, group=64, schedule="gemv", f32=True)
                      for r in range(max(ROWS))])
    assert torch.equal(alone, gemv)
    for m in ROWS:
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=8, group=64, f32=True), alone[:m])
    perm = torch.randperm(max(ROWS), device="cuda")
    assert torch.equal(matmul(x[perm], words, scale, bias, bits=8, group=64, f32=True), alone[perm])


def test_pair_matches_two_wmma_launches():
    if gfx_name() not in WMMA:
        pytest.skip("the paired matmul is the WMMA schedule")
    _, words_a, scale_a, bias_a = _pack(64, 256, 8, 64, 21)
    _, words_b, scale_b, bias_b = _pack(64, 256, 8, 64, 22)
    tensors = [t.cuda() for t in (words_a, scale_a, bias_a, words_b, scale_b, bias_b)]
    words_a, scale_a, bias_a, words_b, scale_b, bias_b = tensors
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    for rows in (1, 8, 17):
        got_a, got_b = matmul_pair(x[:rows], words_a, scale_a, bias_a, words_b, scale_b, bias_b, bits=8, group=64,
                                   f32=True)
        one_a = matmul(x[:rows], words_a, scale_a, bias_a, bits=8, group=64, schedule="wmma", f32=True)
        one_b = matmul(x[:rows], words_b, scale_b, bias_b, bits=8, group=64, schedule="wmma", f32=True)
        assert torch.equal(got_a, one_a)
        assert torch.equal(got_b, one_b)


@pytest.mark.parametrize("group,k", [(32, 128), (64, 256), (128, 256)])
def test_group_matches_solo_wmma(group, k):
    """Columns of different widths in one launch match the same columns launched alone."""

    if gfx_name() not in WMMA:
        pytest.skip("the grouped matmul is the WMMA schedule")
    packs = []
    for n, seed in ((40, 31), (16, 32), (64, 33)):
        packs.append(_pack(n, k, 8, group, seed))
    tensors = []
    for _, words, scale, bias in packs:
        tensors.append((words.cuda(), scale.cuda(), bias.cuda()))
    x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
    for rows in (1, 8, 17):
        got = matmul_group(x[:rows], tensors, bits=8, group=group, f32=True)
        for (words, scale, bias), part in zip(tensors, got):
            solo = matmul(x[:rows], words, scale, bias, bits=8, group=group, schedule="wmma", f32=True)
            assert torch.equal(part, solo)


def test_wmma_partial_tile_matches_one_row():
    """The last 16-column tile is short. Its row still matches that row launched alone."""

    if gfx_name() not in WMMA:
        pytest.skip("partial-tile WMMA is the gfx11 schedule")
    _, words, scale, bias = _pack(40, 128, 8, 64, 5)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    alone = matmul(x[:1], words, scale, bias, bits=8, group=64, schedule="wmma", f32=True)
    wide = matmul(x, words, scale, bias, bits=8, group=64, schedule="wmma", f32=True)
    assert torch.equal(wide[:1], alone)


def test_fp16_split_k_matches_one_launch():
    """Group-boundary K splits fold in group order, so the bits match one launch, including M=1 against a wide grid."""

    if gfx_name() in WMMA:
        pytest.skip("FP16 activations are the RDNA2 schedule")
    specs = (
        (1, 512, 256, 8, 64, 3),
        (3, 1024, 512, 8, 64, 4),
        (8, 4096, 256, 8, 32, 5),
        (1, 640, 384, 5, 64, 6),
        (17, 768, 256, 3, 32, 7),
    )
    for rows, n, k, bits, group, seed in specs:
        _, words, scale, bias = _pack(n, k, bits, group, seed)
        words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(rows, k, generator=g, device="cuda", dtype=torch.float16)
        one = matmul(x, words, scale, bias, bits=bits, group=group, f32=True, dot2_split=False)
        split = matmul(x, words, scale, bias, bits=bits, group=group, f32=True, dot2_split=True)
        assert torch.equal(one, split)
    _, words, scale, bias = _pack(512, 256, 8, 64, 9)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    g = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(1024, 256, generator=g, device="cuda", dtype=torch.float16)
    alone = matmul(x[:1], words, scale, bias, bits=8, group=64, f32=True)
    wide = matmul(x, words, scale, bias, bits=8, group=64, f32=True)
    assert torch.equal(wide[:1], alone)


def test_fp16_word_spanning_rows_do_not_depend_on_row_count():
    if gfx_name() in WMMA:
        pytest.skip("FP16 activations are the RDNA2 schedule")
    _, words, scale, bias = _pack(32, 96, 3, 32, 13)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    x = torch.randn((17, 96), device="cuda", dtype=torch.float16)
    alone = matmul(x[:1], words, scale, bias, bits=3, group=32, f32=True)
    assert torch.equal(matmul(x, words, scale, bias, bits=3, group=32, f32=True)[:1], alone)


def _schedules():
    """Every schedule this part runs, with its activation dtype."""

    if gfx_name() in WMMA:
        return [(schedule, torch.bfloat16) for schedule in ("auto", "gemv", "wmma", "decode")]
    return [("auto", torch.float16), ("gemv", torch.bfloat16)]


@pytest.mark.parametrize("table", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bits,group,k", [(8, 64, 256), (4, 64, 256), (6, 32, 192), (3, 128, 384)])
def test_bf16_and_fp16_tables_match_the_same_values_in_fp32(table, bits, group, k):
    """Group tables stored as bf16 or fp16 widen exactly, so every schedule gives the fp32 table's bits."""

    _, words, scale, bias = _pack(48, k, bits, group, 41)
    words, scale, bias = words.cuda(), scale.to(table).cuda(), bias.to(table).cuda()
    wide_scale, wide_bias = scale.float(), bias.float()
    for schedule, act in _schedules():
        if schedule == "decode" and bits != 8:
            continue
        x = torch.randn(17, k, device="cuda", dtype=act)
        for rows in (1, 8, 17):
            if schedule == "decode" and rows > 16:
                continue
            splits = (None, True) if act == torch.float16 else (None,)
            for split in splits:
                kw = dict(bits=bits, group=group, schedule=schedule, f32=True, dot2_split=split)
                got = matmul(x[:rows], words, scale, bias, **kw)
                assert torch.equal(got, matmul(x[:rows], words, wide_scale, wide_bias, **kw)), (schedule, rows, split)


def test_pair_and_group_read_bf16_tables():
    """Paired and grouped launches read bf16 tables too. Sides of mixed types widen to fp32 together."""

    if gfx_name() not in WMMA:
        pytest.skip("the paired and grouped matmuls are the WMMA schedule")
    packs = [_pack(n, 256, 8, 64, seed) for n, seed in ((64, 51), (64, 52), (16, 53))]
    half = [(w.cuda(), s.to(torch.bfloat16).cuda(), b.to(torch.bfloat16).cuda()) for _, w, s, b in packs]
    wide = [(w, s.float(), b.float()) for w, s, b in half]
    x = torch.randn(9, 256, device="cuda", dtype=torch.bfloat16)
    got = matmul_pair(x, *half[0], *half[1], bits=8, group=64, f32=True)
    want = matmul_pair(x, *wide[0], *wide[1], bits=8, group=64, f32=True)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    want = matmul_group(x, wide, bits=8, group=64, f32=True)
    for sides in (half, [half[0], wide[1], half[2]]):
        got = matmul_group(x, sides, bits=8, group=64, f32=True)
        assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_fp16_decode_tile_matches_the_one_thread_kernel(bits, group):
    """The 1-8 row decode tile gives the one-thread kernel's bits, a tail round of fewer than 8 groups included."""

    if gfx_name() in WMMA:
        pytest.skip("FP16 activations are the RDNA2 schedule")
    k = group * 12
    for n in (77, 300):
        _, words, scale, bias = _pack(n, k, bits, group, bits * 1000 + group + n)
        words, scale, bias = words.cuda(), scale.to(torch.bfloat16).cuda(), bias.to(torch.bfloat16).cuda()
        x = torch.randn((8, k), device="cuda", dtype=torch.float16)
        alone = matmul(x[:1], words, scale, bias, bits=bits, group=group, f32=True)
        for m in range(1, 9):
            got = matmul(x[:m], words, scale, bias, bits=bits, group=group, f32=True)
            want = matmul(x[:m], words, scale, bias, bits=bits, group=group, schedule="gemv", f32=True)
            assert torch.equal(got, want), (n, m)
            assert torch.equal(got[:1], alone), (n, m)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("groups", [4, 8, 32, 80, 160])
def test_fp16_one_row_tile_matches_the_one_thread_kernel(bits, group, groups):
    """The one-row FP16 tile keeps the one-thread kernel's bits at every lane mapping; fp16 out is fp32 rounded."""

    if gfx_name() in WMMA:
        pytest.skip("FP16 activations are the RDNA2 schedule")
    k = group * groups
    n = 157
    _, words, scale, bias = _pack(n, k, bits, group, bits * 977 + group + groups)
    words, scale, bias = words.cuda(), scale.to(torch.bfloat16).cuda(), bias.to(torch.bfloat16).cuda()
    x = torch.randn((2, k), device="cuda", dtype=torch.float16)
    one = matmul(x[:1], words, scale, bias, bits=bits, group=group, f32=True)
    assert torch.equal(one, matmul(x[:1], words, scale, bias, bits=bits, group=group, schedule="gemv", f32=True))
    assert torch.equal(one, matmul(x, words, scale, bias, bits=bits, group=group, f32=True)[:1])
    half = matmul(x[:1], words, scale, bias, bits=bits, group=group)
    assert half.dtype == torch.float16 and torch.equal(half, one.half())


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_fp16_prefill_tile_matches_the_one_thread_kernel(bits, group):
    """The 128-row prefill tile reads every width and gives the one-thread kernel's bits at any row count."""

    if gfx_name() in WMMA:
        pytest.skip("FP16 activations are the RDNA2 schedule")
    k = group * 6
    _, words, scale, bias = _pack(70, k, bits, group, bits * 100 + group)
    words, scale, bias = words.cuda(), scale.cuda(), bias.cuda()
    x = torch.randn((300, k), device="cuda", dtype=torch.float16)
    want = matmul(x, words, scale, bias, bits=bits, group=group, schedule="gemv", f32=True)
    for m in (9, 17, 128, 300):
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=bits, group=group, f32=True), want[:m]), m


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_fp16_gemm_tile_matches_the_one_thread_kernel(bits, group):
    """From 64 rows the 128x128 GEMM tile runs: ragged rows and columns keep the one-thread kernel's bits."""

    if gfx_name() in WMMA:
        pytest.skip("FP16 activations are the RDNA2 schedule")
    k = group * 5
    _, words, scale, bias = _pack(200, k, bits, group, bits * 10 + group)
    words, scale, bias = words.cuda(), scale.to(torch.float16).cuda(), bias.to(torch.float16).cuda()
    x = torch.randn((260, k), device="cuda", dtype=torch.float16)
    want = matmul(x, words, scale, bias, bits=bits, group=group, schedule="gemv", f32=True)
    for m in (64, 129, 260):
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=bits, group=group, f32=True), want[:m]), m
    assert torch.equal(matmul(x[5:6], words, scale, bias, bits=bits, group=group, f32=True), want[5:6])


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("groups", [10, 8, 32])
def test_bf16_dot2_rows_keep_their_bits_on_gfx11(bits, group, groups):
    """gfx11 BF16 dot2 tiles: a row's bits are the same alone, in a short batch and in a prefill tile, and the
    decode tile's own bf16 rounding equals the cast of its fp32 result. 8 and 32 groups take the one-row tile's
    vector loads of the scales and biases and the prefill tile's staged slabs of eight groups; 10 their scalar
    loads."""

    if gfx_name() not in WMMA:
        pytest.skip("the BF16 dot2 tiles are the gfx11 / gfx12 schedule")
    k = group * groups
    codes, words, scale, bias = _pack(150, k, bits, group, bits * 31 + group)
    words, scale, bias = words.cuda(), scale.to(torch.bfloat16).cuda(), bias.to(torch.bfloat16).cuda()
    x = torch.randn((70, k), device="cuda", dtype=torch.bfloat16)
    tall = matmul(x, words, scale, bias, bits=bits, group=group, f32=True)
    for m in (1, 2, 5, 8):
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=bits, group=group, f32=True), tall[:m]), m
        assert torch.equal(matmul(x[:m], words, scale, bias, bits=bits, group=group), tall[:m].to(torch.bfloat16)), m
    want = _reference(x[:4], codes, scale.float().cpu(), bias.float().cpu(), group)
    assert torch.allclose(tall[:4].cpu(), want, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("m", [1, 2])
def test_rows_group_matches_solo_launches(m):
    """One launch of up to four products of a decode row: every output has its solo launch's bits."""

    from tensorfold.rocm.kernels.affine import matmul_rows

    if gfx_name() not in WMMA:
        pytest.skip("the grouped row tile is the gfx11 / gfx12 BF16 schedule")
    k, group = 2048, 64
    packs = []
    for i, n in enumerate((640, 128, 32, 7)):
        _, words, scale, bias = _pack(n, k, 4, group, 90 + i)
        packs.append((words.cuda(), scale.to(torch.bfloat16).cuda(), bias.to(torch.bfloat16).cuda()))
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    outs = matmul_rows(x, tuple(packs), bits=4, group=group)
    assert outs is not None
    for (words, scale, bias), out in zip(packs, outs):
        assert torch.equal(out, matmul(x, words, scale, bias, bits=4, group=group))


def test_bf16_decode_output_rounds_nan_as_torch_does():
    """The decode tile's own bf16 rounding of a NaN row is torch's cast: the quiet 0x7FC0, sign and payload dropped."""

    if gfx_name() not in WMMA:
        pytest.skip("the bf16 output is the gfx11 / gfx12 decode tile")
    _, words, scale, bias = _pack(96, 512, 4, 64, 77)
    words, scale, bias = words.cuda(), scale.to(torch.bfloat16).cuda(), bias.to(torch.bfloat16).cuda()
    x = torch.randn((2, 512), device="cuda", dtype=torch.bfloat16)
    x[0, 5] = -float("nan")
    want = matmul(x, words, scale, bias, bits=4, group=64, f32=True).to(torch.bfloat16)
    got = matmul(x, words, scale, bias, bits=4, group=64)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
