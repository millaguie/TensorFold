"""A quick speed-and-bits check of the ROCm engine on one card: one load, prefill at a few lengths, a greedy decode.

Prints the prefill and decode rates and a hash of the generated tokens. Run it before and after a kernel change:
the rates must not drop and, unless the change says which sums move, the hashes must stay the same.

  python3 tools/quick_rocm.py MODEL_DIR [--lengths 2048,8192] [--tokens 128] [--profile]
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

PROMPT = ("Write a Python function that parses an ISO 8601 date string without using datetime, "
          "with docstrings and three pytest tests.")


def sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--lengths", default="2048,8192")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--profile", action="store_true", help="print the prefill's top kernels at the last length")
    ap.add_argument("--profile-decode", action="store_true", help="print a 33-token greedy decode's top kernels")
    a = ap.parse_args()

    from tokenizers import Tokenizer

    from tensorfold.rocm.serving.engine import QwenEngine

    t0 = time.perf_counter()
    eng = QwenEngine.load(a.model)
    tok = Tokenizer.from_file(str(a.model / "tokenizer.json"))
    print(f"load {time.perf_counter() - t0:.1f} s on {torch.cuda.get_device_properties(0).gcnArchName}", flush=True)

    g = torch.Generator().manual_seed(0)
    lengths = [int(x) for x in a.lengths.split(",")]
    filler = torch.randint(1000, 100000, (max(lengths),), generator=g).tolist()
    eng.prefill_caches(filler[:256])
    sync()
    for n in lengths:
        t = time.perf_counter()
        eng.prefill_caches(filler[:n])
        sync()
        dt = time.perf_counter() - t
        print(f"prefill {n:6d}: {n / dt:7.0f} tok/s ({dt:.2f} s)", flush=True)

    ids = tok.encode(f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
    for draft in (False, True):
        out: list[int] = []
        t = time.perf_counter()
        eng.generate(ids, a.tokens, None, lambda new: out.extend(new), stop_eos=False, draft=draft)
        sync()
        dt = time.perf_counter() - t
        sha = hashlib.sha256(str(out).encode()).hexdigest()[:12]
        print(f"decode draft={draft!s:5}: {len(out) / dt:6.1f} tok/s ({len(out)} tokens incl. prefill of {len(ids)}) "
              f"sha {sha}", flush=True)

    def report(p, what: str, per: int = 1) -> None:
        rows = sorted((e for e in p.key_averages() if e.device_time_total > 0), key=lambda e: -e.device_time_total)
        total = sum(e.device_time_total for e in rows)
        print(f"profile, {what}: GPU {total / 1e6 / per * 1e3:.1f} ms a unit over "
              f"{sum(e.count for e in rows) / per:.0f} launches")
        for e in rows[:20]:
            print(f"  {e.device_time_total / 1e3 / per:9.2f} ms {100 * e.device_time_total / total:5.1f}% "
                  f"{e.count / per:7.1f}x  {e.key[:100]}")

    if a.profile or a.profile_decode:
        from torch.profiler import ProfilerActivity, profile

    if a.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            eng.prefill_caches(filler[:lengths[-1]])
            sync()
        report(p, f"prefill {lengths[-1]} (unit: the prefill)")
    if a.profile_decode:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            eng.generate(ids, 33, None, lambda new: None, stop_eos=False, draft=False)
            sync()
        report(p, "decode (unit: one token, the prompt's prefill included)", per=33)


if __name__ == "__main__":
    main()
