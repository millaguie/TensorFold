"""The concurrent decoder restores a host-tier state into the rows its memory gate admitted, not the whole reply's."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.streams import PrefixCache  # noqa: E402
from tensorfold.families.qwen3_5.cuda.multi import MultiDecoder  # noqa: E402


class _Gate:
    waits = ends = 0

    def fits(self, nbytes: int) -> bool:
        return True


class _Tier:
    def __init__(self):
        self.asked = []

    def take(self, prompt, than=0, into=None, rows=None):
        self.asked.append(rows)
        return None                                       # nothing kept: the stream prefills from scratch


def test_a_restore_asks_for_the_admitted_rows():
    dec = object.__new__(MultiDecoder)
    dec.broken, dec.context, dec.vision, dec.world, dec.next_id = None, 0, None, 1, 0
    dec.max_rows, dec.row_bytes, dec.streams, dec.filling = 12, 1, {}, []
    dec.memory_gate, dec.tier, dec.cache = _Gate(), _Tier(), PrefixCache(8)
    dec._send = lambda values: None
    queued = []
    dec._queue = lambda s, hit: queued.append(s)
    s = SimpleNamespace(prompt=list(range(1, 2001)), count=46000, draft=True, sampling=None, constraint=None,
                        vision=None, waiting=False)
    dec.admit(s)
    assert queued == [s]
    assert dec.tier.asked == [dec._first(s)]               # what _room let in: prompt and a window, a step at a time
    assert dec._first(s) < dec._most(s) == 48000            # not prompt and reply, which the gate never saw
