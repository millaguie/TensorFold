"""tools/prompt_precision_cuda.py scores rows as tools/prompt_precision.py does."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import prompt_precision  # noqa: E402
import prompt_precision_cuda  # noqa: E402


def test_the_torch_score_matches_the_numpy_one():
    gen = np.random.default_rng(3)
    ours, theirs = prompt_precision_cuda.Score(), prompt_precision.Score()
    for source in ("wikitext", "code", "wikitext"):
        ref = gen.normal(size=(6, 50)).astype(np.float32)
        path = ref + gen.normal(scale=0.3, size=ref.shape).astype(np.float32)
        targets = gen.integers(0, 50, size=6)
        ours.add(torch.from_numpy(ref), torch.from_numpy(path), torch.from_numpy(targets), source)
        theirs.add(ref, path, targets, source)
    ours.add(torch.from_numpy(ref[:1]), torch.from_numpy(path[:1]), None, "chat")    # a last row: KL and top-1 only
    theirs.add(ref[:1], path[:1], None, "chat")
    assert ours.rows == theirs.rows and ours.top == theirs.top
    assert ours.kl == pytest.approx(theirs.kl, rel=1e-12)
    assert ours.line() == theirs.line()
