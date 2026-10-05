"""Microbenchmark of the ROCm affine decode matmul (one row) at the 35B-A3B's shapes: best time, GB/s and a bits hash.

  python3 tools/bench_rocm_affine.py [--rows 1] [--reps 200]
"""

from __future__ import annotations

import argparse
import hashlib
import time

import torch

SHAPES = {                      # name: (N, K), 4-bit words, groups of 64
    "gdn_in 8192x2048": (8192, 2048),
    "out 2048x4096": (2048, 4096),
    "attn 4096x2048": (4096, 2048),
    "experts 9216x2048": (9216, 2048),
    "down 2048x512": (2048, 512),
    "head 248320x2048": (248320, 2048),
    "tiny 32x2048": (32, 2048),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    from tensorfold.rocm.kernels import affine

    g = torch.Generator(device="cuda").manual_seed(0)
    # The APU's GPU and memory clocks ramp under load: a second of work first, then the best of five trials a shape.
    warm = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")
    t = time.perf_counter()
    while time.perf_counter() - t < 1.0:
        warm.view(torch.int32).max()
    torch.cuda.synchronize()
    del warm
    digest = hashlib.sha256()
    total = 0.0
    for name, (n, k) in SHAPES.items():
        words = torch.randint(-2**31, 2**31 - 1, (n, k * 4 // 32), generator=g, device="cuda", dtype=torch.int32)
        scale = (torch.rand(n, k // 64, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
        bias = (torch.rand(n, k // 64, generator=g, device="cuda") * -0.1).to(torch.bfloat16)
        x = torch.randn(a.rows, k, generator=g, device="cuda").to(torch.bfloat16)
        f = lambda: affine.matmul(x, words, scale, bias, bits=4, group=64)     # noqa: E731
        out = f()
        torch.cuda.synchronize()
        reps = max(5, a.reps // 50) if n > 100000 else a.reps
        dt = float("inf")
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(reps):
                f()
            torch.cuda.synchronize()
            dt = min(dt, (time.perf_counter() - t) / reps)
        nbytes = words.numel() * 4 + scale.numel() * 4
        total += dt
        digest.update(out.float().cpu().numpy().tobytes())
        print(f"{name:20s} {dt * 1e6:8.1f} us {nbytes / dt / 1e9:6.1f} GB/s", flush=True)
    print(f"sum {total * 1e3:.3f} ms, bits {digest.hexdigest()[:12]}")


if __name__ == "__main__":
    main()
