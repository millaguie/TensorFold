"""The host-RAM tier on a GPU: spills on the side stream, fence, restores, under the main stream's own writes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_cuda_host_tier import FakeState, _same  # noqa: E402

from tensorfold.cuda.host_tier import HostTier  # noqa: E402


def _state(gen, ids: list[int], rows: int, width: int) -> FakeState:
    """A state over ``ids``: attention rows that are a function of the ids (as a forward's are), random DeltaNet."""

    st = FakeState()
    st.pos, st.limit, st.rope_delta = len(ids), 1 << 20, 0
    st.conv = [torch.randn(3, 64, generator=gen, device="cuda").bfloat16(), None]
    st.rec = [torch.randn(4, 64, 64, generator=gen, device="cuda"), None]
    keys = torch.zeros((rows, 2, width), dtype=torch.bfloat16, device="cuda")
    keys[:len(ids)] = torch.tensor(ids, device="cuda", dtype=torch.float32)[:, None, None].bfloat16() + \
        torch.arange(width, device="cuda").bfloat16() / 8
    st.kv = [None, (keys, -keys)]
    return st


def _copy(st: FakeState) -> dict:
    return {"conv": st.conv[0].clone(), "rec": st.rec[0].clone(), "k": st.kv[1][0][:st.pos].clone(),
            "v": st.kv[1][1][:st.pos].clone()}


def _clobber(st: FakeState) -> None:
    """The main stream's next prefill writes over every buffer the spills read, at once after the fence."""

    for t in (st.conv[0], st.rec[0], st.kv[1][0], st.kv[1][1]):     # queued at once: a copy in flight would race
        t.view(torch.uint8).fill_(0xA5)


@pytest.mark.parametrize("pin", [True, False])
def test_spills_under_the_main_streams_writes_come_back_bit_for_bit(pin):
    gen = torch.Generator(device="cuda").manual_seed(5)
    tier = HostTier(4 << 30, "cuda", chunk=1 << 22, pin=pin)
    base = list(range(1, 8001))
    states = {}
    for name, ids in (("a", base), ("b", base + [9, 9, 9] * 1000), ("c", list(range(9000, 12000)))):
        st = _state(gen, ids, len(ids) + 64, 4096)        # 8-12K rows of 2 x 4096 bf16: 128-192 MB a buffer
        states[name] = (ids, st, _copy(st))
        assert tier.put(ids, st, None)
    tier.fence()                                           # the engine's contract before a prefill writes
    for _, st, _ in states.values():
        _clobber(st)
    for name, (ids, _, want) in states.items():
        got_ids, back, _ = tier.take(ids + [0], rows=len(ids) + 100)
        assert got_ids == ids, name
        got = _copy(back)
        assert all(_same(got[k], want[k]) for k in want), name
        assert back.kv[1][0].shape[0] == len(ids) + 100
    torch.cuda.synchronize()


def test_a_restore_then_a_spill_of_the_same_rows_keeps_both():
    """Restore an entry, write past its rows on the main stream, spill it again: the chained segment still serves."""

    gen = torch.Generator(device="cuda").manual_seed(6)
    tier = HostTier(1 << 30, "cuda", chunk=1 << 22)
    ids = list(range(1, 6001))
    st = _state(gen, ids, 7000, 512)
    want = _copy(st)
    assert tier.put(ids, st, None)
    tier.fence()
    _clobber(st)
    _, back, _ = tier.take(ids + [1])
    more = ids + list(range(20000, 20500))
    back.kv[1][0][len(ids):len(more)] = 1.0                # the resumed prefill's rows, past the restored ones
    back.kv[1][1][len(ids):len(more)] = -1.0
    back.pos = len(more)
    assert tier.put(more, back, None)
    tier.fence()
    _clobber(back)
    _, again, _ = tier.take(more + [1])
    assert _same(again.kv[1][0][:len(ids)], want["k"]) and _same(again.kv[1][1][:len(ids)], want["v"])
    assert bool((again.kv[1][0][len(ids):len(more)] == 1.0).all())
    torch.cuda.synchronize()
