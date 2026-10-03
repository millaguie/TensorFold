"""prompt_precision.py's score for Qwen3.8-27B on CUDA and ROCm: the engine's prompt rows against an fp32 forward.

The same eight sequences of 4096 tokens (wikitext-2, CPython, chats). The path prefills each sequence through the
engine's prompt chunks (bf16 or FP8 activations, bf16 or FP8 keys and values) and takes every row's logits; the
reference is ``reference.forward``: fp32 matmuls of the dequantized weights, stored activations rounded to bf16.
KL(reference || path) over the vocabulary at every row, top-1 agreement and perplexity, scored on the GPU.

  python3 tools/prompt_precision_cuda.py --model DIR [--prefill-fp8] [--kv-dtype fp8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_precision import LENGTH, counts, encode, sequences  # noqa: E402

SOURCES = ("wikitext", "code", "chat")


class Score:
    """``prompt_precision.Score`` on torch rows: the same sums, accumulated in float64."""

    def __init__(self) -> None:
        self.kl = 0.0
        self.rows = 0
        self.top = 0
        self.nll = {s: [0.0, 0] for s in SOURCES}
        self.nll_ref = {s: [0.0, 0] for s in SOURCES}

    def add(self, ref, path, targets, source: str) -> None:
        import torch

        r = torch.log_softmax(ref.double(), -1)
        p = torch.log_softmax(path.double(), -1)
        self.kl += float((r.exp() * (r - p)).sum())
        self.rows += int(ref.shape[0])
        self.top += int((r.argmax(-1) == p.argmax(-1)).sum())
        if targets is not None and targets.numel():
            n = targets.numel()
            pick = torch.arange(n, device=targets.device)
            self.nll[source][0] += float(-p[pick, targets].sum())
            self.nll[source][1] += n
            self.nll_ref[source][0] += float(-r[pick, targets].sum())
            self.nll_ref[source][1] += n

    def line(self) -> str:
        import math

        def ppl(bucket, source):
            total, n = bucket[source]
            return f"{math.exp(total / n):.4f}" if n else "n/a"

        def delta(source):
            (got, n), (ref, m) = self.nll[source], self.nll_ref[source]
            return f"{(math.exp(got / n) - math.exp(ref / m)) / math.exp(ref / m) * 100:+.3f}%" if n and m else "n/a"

        return (f"KL {self.kl / self.rows:.5f}  top-1 {100.0 * self.top / self.rows:.2f}%  rows {self.rows}  "
                f"ppl wikitext {ppl(self.nll, 'wikitext')} (ref {ppl(self.nll_ref, 'wikitext')}, {delta('wikitext')})  "
                f"ppl code {ppl(self.nll, 'code')} (ref {ppl(self.nll_ref, 'code')}, {delta('code')})")


def _dense(q):
    """A projection's (N, K) weight in fp32, whatever its stored form."""

    import torch

    from tensorfold.families.qwen3_5.cuda.qmm import dequantize
    from tensorfold.families.qwen3_5.cuda.qmm_fast import untile

    if hasattr(q, "dequantize"):                      # MXFP4 (mx4_load.Mx4)
        return q.dequantize().float()
    if getattr(q, "layout", None) == "b16":           # a bf16 table or head as stored
        return q.weight.float() if q.weight.is_cuda else q.weight.to("cuda", torch.float32)
    q = untile(q)
    return dequantize(q.weight, q.scales, q.biases, q.bits, q.gs).float()


def install_reference() -> None:
    """``reference._linear`` for every stored form: fp32 matmul of the dequantized weight, bf16 out."""

    import torch

    from tensorfold.families.qwen3_5.cuda import reference

    reference._linear = lambda x, q: (x.float() @ _dense(q).T).to(torch.bfloat16)


def reference_logits(w, ids, st):
    """``reference.forward`` with the embedding and head read through ``_dense`` (any stored form)."""

    import torch

    from tensorfold.families.qwen3_5.cuda import reference as ref

    c = w.config
    x = _dense(w.embed)[ids.long()].to(torch.bfloat16) if getattr(w.embed, "layout", None) == "b16" else None
    if x is None:
        from tensorfold.families.qwen3_5.cuda.qmm import dequantize

        e = w.embed
        x = dequantize(e.weight[ids.long()], e.scales[ids.long()] if e.scales is not None else None,
                       e.biases[ids.long()] if e.biases is not None else None, e.bits, e.gs).to(torch.bfloat16)
    for i, layer in enumerate(w.layers):
        h = ref._rms(x, layer.input_norm, c.eps).to(torch.bfloat16)
        r = ref._gdn(layer, h, st, i, c) if layer.linear else ref._attention(layer, h, st, i, c, w.inv_freq)
        x = (x.float() + r.float()).to(torch.bfloat16)
        h = ref._rms(x, layer.post_norm, c.eps).to(torch.bfloat16)
        act = (torch.nn.functional.silu(ref._linear(h, layer.gate).float())
               * ref._linear(h, layer.up).float()).to(torch.bfloat16)
        x = (x.float() + ref._linear(act, layer.down).float()).to(torch.bfloat16)
    st.pos += int(ids.shape[0])
    h = ref._rms(x, w.norm, c.eps).to(torch.bfloat16).float()
    return torch.cat([h @ part.T for part in _head_slices(w.head)], dim=1)


def _head_slices(head, rows: int = 32768):
    """The head's fp32 rows a slice at a time (all of them at once would take 5 GB, 10 dequantizing an MLX head)."""

    import torch

    from tensorfold.families.qwen3_5.cuda.qmm import dequantize
    from tensorfold.families.qwen3_5.cuda.qmm_fast import untile

    if getattr(head, "layout", None) == "b16" or hasattr(head, "dequantize"):
        weight = head.weight if getattr(head, "layout", None) == "b16" else None
        if weight is None:
            yield _dense(head)
            return
        for a in range(0, weight.shape[0], rows):
            yield weight[a:a + rows].to("cuda", torch.float32)
        return
    q = untile(head)
    for a in range(0, q.n, rows):
        b = min(q.n, a + rows)
        yield dequantize(q.weight[a:b], q.scales[a:b] if q.scales is not None else None,
                         q.biases[a:b] if q.biases is not None else None, q.bits, q.gs).float()


def path_logits(w, ids, st, chunk: int):
    """The engine's prompt rows: every row's final normed state from its chunks, then the head as decode runs it."""

    import torch

    from tensorfold.families.qwen3_5.cuda.forward import _mm
    from tensorfold.families.qwen3_5.cuda.prefill import prefill_chunk

    out = []
    for a in range(0, ids.shape[0], chunk):
        normed, _ = prefill_chunk(w, ids[a:a + chunk], st, every=True)
        out.append(torch.cat([_mm(normed[b:b + 16].contiguous(), w.head)          # bf16, as decode samples them
                              for b in range(0, normed.shape[0], 16)]))
    return torch.cat(out)


def run(args) -> int:
    import torch
    from tokenizers import Tokenizer

    from tensorfold.cuda import prompt_precision
    from tensorfold.families.qwen3_5 import cuda_engine
    from tensorfold.families.qwen3_5.cuda import reference
    from tensorfold.families.qwen3_5.cuda.forward import State

    prompt_precision.set_fp8(args.prefill_fp8)            # before any weight loads, as the server does
    engine = cuda_engine(args.model, no_drafts=True, kv_dtype=args.kv_dtype, context=args.length + 256)
    w = engine.w
    install_reference()
    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    rows = sequences(tok, args.wikitext, args.length, *counts(args.sequences))
    score = Score()
    for k, (source, ids) in enumerate(rows):
        t0 = time.perf_counter()
        x = torch.tensor(ids, dtype=torch.int32, device="cuda")
        with torch.no_grad():
            got = path_logits(w, x, State(w), args.chunk)
            st = reference.State(w)
            for a in range(0, len(ids), args.ref_chunk):
                want = reference_logits(w, x[a:a + args.ref_chunk], st)
                n = want.shape[0]
                targets = x[a + 1:a + n + 1].long()
                score.add(want[:targets.numel()] if targets.numel() < n else want,
                          got[a:a + n][:targets.numel()] if targets.numel() < n else got[a:a + n], targets, source)
                if targets.numel() < n:              # the last row has no next token: KL and top-1 only
                    score.add(want[-1:], got[a + n - 1:a + n], None, source)
                del want
        del got, st
        torch.cuda.empty_cache()
        print(f"[{k + 1}/{len(rows)}] {source}: {time.perf_counter() - t0:.0f} s  {score.line()}", flush=True)
    mode = f"prompts {'FP8' if args.prefill_fp8 else 'bf16'}, keys and values {args.kv_dtype}"
    print(f"{Path(args.model).name} ({mode}): {score.line()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the 27B's CUDA prompt rows against an fp32 forward.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wikitext", type=Path, default=Path.home() / "tf-data" / "wikitext-2-raw" / "wiki.test.raw")
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--prefill-fp8", action="store_true")
    parser.add_argument("--kv-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--chunk", type=int, default=2048, help="the engine's prompt chunk")
    parser.add_argument("--ref-chunk", type=int, default=512, help="rows a reference step scores")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
