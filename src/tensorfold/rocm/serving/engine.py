"""ROCm Qwen engine for the torch server: one request at a time, prefix cache, MTP drafts, tp ranks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose
from tensorfold.rocm.model.forward import _blank_caches, _project, forward_hidden
from tensorfold.rocm.model.mtp import MTPEngine
from tensorfold.rocm.serving.prefix import PrefixCache, entry_end, trim_bytes
from tensorfold.rocm.model.qwen import Engine as Kernels
from tensorfold.rocm.model.qwen import activation_dtype, load, slice_for_tp
from tensorfold.rocm.model.qwen_math import DevicePos
from tensorfold.rocm.model.qwen_tp import tp_forward_hidden, vocab_gather

_DEFAULT_EOS = (151645,)
_DEFAULT_MTP_DEPTH = 4
_LONG_PROMPT = 8192            # past this many prompt tokens a request hands its freed memory back


def _resolve_p2p(gfx: str, p2p: bool | None) -> bool | None:
    """``--p2p`` / ``--no-p2p`` as given; on for a multi-die APU; otherwise ``None``: RCCL decides."""

    if p2p is not None:
        return bool(p2p)
    try:
        multi = bool(torch.cuda.get_device_properties(0).multi_gpu_capable)
        integrated = bool(torch.cuda.get_device_properties(0).is_integrated)
    except (AttributeError, AssertionError, RuntimeError):
        return None
    return True if multi and integrated else None


def read_eos(model_dir: Path) -> tuple[int, ...]:
    """Every end id in config.json (top level and text_config) and generation_config.json."""

    found: list[int] = []

    def take(raw) -> None:
        for token in [raw] if isinstance(raw, int) else (raw or []):
            if int(token) not in found:
                found.append(int(token))

    root = Path(model_dir)
    for name in ("config.json", "generation_config.json"):
        path = root / name
        if path.is_file():
            config = json.loads(path.read_text())
            take(config.get("eos_token_id"))
            take(config.get("text_config", {}).get("eos_token_id"))
    return tuple(found) or _DEFAULT_EOS


def cache_bytes(caches: list[dict]) -> int:
    """Bytes of the tensors a prefix entry keeps."""

    total = 0
    for cache in caches:
        for key in ("k", "v", "conv", "state"):
            value = cache.get(key)
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
    return total


def clone_caches(caches: list[dict]) -> list[dict]:
    """A detached copy. Key and value rows past ``len`` are the next reply's room, not the prefix."""

    cloned = []
    for cache in caches:
        if "k" in cache:
            used = int(cache["len"]) if "len" in cache else int(cache["k"].shape[2])
            cloned.append({
                "k": cache["k"][:, :, :used].detach().clone(),
                "v": cache["v"][:, :, :used].detach().clone(),
                "len": used,
            })
        else:
            conv, state = cache.get("conv"), cache.get("state")
            cloned.append({
                "conv": None if conv is None else conv.detach().clone(),
                "state": None if state is None else state.detach().clone(),
            })
    return cloned


def _grow(caches: list[dict], total: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
    """Copy a prefix into buffers long enough for ``total`` positions."""

    grown = []
    for cache in caches:
        if "k" not in cache:
            grown.append(cache)
            continue
        batch, heads, _, dim = cache["k"].shape
        used = int(cache["len"]) if "len" in cache else int(cache["k"].shape[2])
        hold = max(total, used)
        key = torch.empty(batch, heads, hold, dim, dtype=dtype, device=device)
        value = torch.empty_like(key)
        if used:
            key[:, :, :used] = cache["k"][:, :, :used]
            value[:, :, :used] = cache["v"][:, :, :used]
        grown.append({"k": key, "v": value, "len": used})
    return grown


class _StepGraph:
    """One request's one-token forward: eager once, then captured and replayed with device token and position."""

    def __init__(self, caches: list[dict], device: torch.device) -> None:
        self.caches = caches
        self.ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.at = DevicePos(device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.hidden: torch.Tensor | None = None
        self.warm = False


def _caches_equal(left: list[dict], right: list[dict]) -> bool:
    for one, two in zip(left, right, strict=True):
        if "k" in one:
            used = int(one["len"]) if "len" in one else int(one["k"].shape[2])
            other = int(two["len"]) if "len" in two else int(two["k"].shape[2])
            if used != other:
                return False
            if used and (not torch.equal(one["k"][:, :, :used], two["k"][:, :, :used])
                         or not torch.equal(one["v"][:, :, :used], two["v"][:, :, :used])):
                return False
        else:
            for key in ("conv", "state"):
                first, second = one.get(key), two.get(key)
                if first is None or second is None:
                    if first is not None or second is not None:
                        return False
                elif not torch.equal(first, second):
                    return False
    return True


class QwenEngine:
    """One ROCm model. Sampling is the shared keyed draw. Tool calls are the server's gate over ``generate``."""

    exact_sampling = True
    concurrent = False
    call_gate = None

    def __init__(self, model, kernels: Kernels, eos: Sequence[int], *, keep: int = 8,
                 context: int | None = None, byte_budget: int | None = None,
                 points: Callable[[Sequence[int]], list[int]] | None = None,
                 tp: int = 1, rank: int = 0, rccl: Any = None, no_drafts: bool = False,
                 mtp_depth: int = _DEFAULT_MTP_DEPTH):
        self.model = model
        self.kernels = kernels
        self.eos = tuple(int(token) for token in eos)
        self.cache = PrefixCache(max(0, int(keep)))
        self.context_window = context
        self.byte_budget = byte_budget
        self.points = points
        self.tp, self.rank, self.rccl, self.no_drafts = int(tp), int(rank), rccl, bool(no_drafts)
        self._posted = 0          # requests rank 0 has published (tp > 1)
        head = getattr(model, "mtp", None)
        self.mtp = MTPEngine(model, head, linear=kernels.linear, rccl=rccl) if head is not None else None
        self.mtp_depth = int(mtp_depth)
        # One-token steps replay a captured graph; TENSORFOLD_GRAPH=0 keeps them eager. On gfx1150 (Radeon 890M, ROCm
        # 7.1-7.2) a replayed step aborts with a malformed AQL packet unless DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 is set
        # before HIP starts, and the graph decodes no faster there, so that part stays eager unless asked.
        from tensorfold.rocm.kernels.build import gfx_name

        self.graphs = os.environ.get("TENSORFOLD_GRAPH", "0" if gfx_name() == "gfx1150" else "1") != "0"
        self._step: _StepGraph | None = None

    @classmethod
    def load(cls, model_dir: Path | str, *, schedule: str | None = None, keep: int = 8,
             context: int | None = None, context_explicit: bool = False, byte_budget: int | None = None,
             tp: int = 1, rank: int = 0, master: str = "", master_port: int = 29551,
             p2p: bool | None = None, no_drafts: bool = False,
             mtp_depth: int = _DEFAULT_MTP_DEPTH) -> QwenEngine:
        from tensorfold.rocm.kernels.build import gfx_name

        # TENSORFOLD_ROCM_SCHEDULE=wmma runs every projection on the gfx11 WMMA tiles (opt-in; auto is dot2).
        schedule = schedule or os.environ.get("TENSORFOLD_ROCM_SCHEDULE", "auto")
        if schedule not in ("auto", "gemv", "wmma"):
            raise ValueError(f"TENSORFOLD_ROCM_SCHEDULE is auto, gemv or wmma, not {schedule!r}")
        path = Path(model_dir)
        if tp == 1:
            if rank != 0:
                raise ValueError("rank must be 0 when tp=1")
            model = load(path)
            from tensorfold.rocm.serving.prefix import message_points

            engine = cls(model, Kernels(model, schedule=schedule), read_eos(path), keep=keep,
                         points=message_points(path), tp=1, rank=0, no_drafts=no_drafts, mtp_depth=mtp_depth)
            return engine._planned(path, context, context_explicit, byte_budget)

        from tensorfold.rocm.serving.comm import RCCL

        # One GPU a rank: with every card visible, rank r takes card r; with one card per process, that card.
        torch.cuda.set_device(rank % torch.cuda.device_count())
        gfx = gfx_name()
        prefer_p2p = _resolve_p2p(gfx, p2p)
        rccl = RCCL(rank, tp, master, master_port, prefer_p2p=prefer_p2p)
        rccl.ready("startup")
        model = load(path)
        slice_for_tp(model, rank, tp)
        from tensorfold.rocm.serving.prefix import message_points

        engine = cls(model, Kernels(model, schedule=schedule), read_eos(path), keep=keep,
                     points=message_points(path), tp=tp, rank=rank, rccl=rccl, no_drafts=no_drafts,
                     mtp_depth=mtp_depth)
        return engine._planned(path, context, context_explicit, byte_budget)

    def _planned(self, path: Path, context: int | None, explicit: bool, byte_budget: int | None) -> QwenEngine:
        """Fit the context window and the prompt cache to this GPU's memory (every rank agrees)."""

        from tensorfold.rocm.serving import memory

        config = json.loads((path / "config.json").read_text())
        native = int((config.get("text_config") or config).get("max_position_embeddings") or 0)
        self.context_window, self.byte_budget = memory.plan(self, native, context, explicit, byte_budget, self.rccl)
        return self

    def warm(self) -> None:
        """One prefill span, nothing stored: every prefill kernel is built and its workspace allocated once."""

        from tensorfold.rocm.model.qwen_math import SPAN

        vocab = self.model.spec.vocab
        self._prefill([1 + index % (vocab - 1) for index in range(SPAN)], None, 0, SPAN + 1, store=False)

    def _dtype(self) -> torch.dtype:
        if self.kernels.dtype is None:
            from tensorfold.rocm.kernels.build import gfx_name

            self.kernels.dtype = activation_dtype(gfx_name())
        return self.kernels.dtype

    def _device(self) -> torch.device:
        return self.model.embed.words.device

    def _forward(self, tokens: Sequence[int], caches: list[dict] | None, pos0: int, *,
                 decode: bool = False) -> tuple[torch.Tensor, list[dict]]:
        """A prefill span (prefill rope and conv), or with ``decode`` one token (HIP rope and conv, graph replay)."""

        device, dtype = self._device(), self._dtype()
        if caches is None:
            caches = _blank_caches(self.model, 1, len(tokens), device, dtype)
        elif decode and self.graphs and len(tokens) == 1:
            return self._graph_forward(int(tokens[0]), caches, pos0)
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=device)
        with torch.inference_mode():
            if self.tp > 1:
                return tp_forward_hidden(self.model, ids, caches, self.kernels.linear, pos0, self.rccl,
                                         act_dtype=dtype, exact_short=not decode)
            return forward_hidden(self.model, ids, caches, self.kernels.linear, pos0, dtype, exact_short=not decode)

    def _graph_forward(self, token: int, caches: list[dict], pos0: int) -> tuple[torch.Tensor, list[dict]]:
        """The one-token forward at ``pos0`` with the eager step's bits; a failed capture turns graphs off."""

        step = self._step
        if step is None or step.caches is not caches:
            step = self._step = _StepGraph(caches, self._device())
        step.ids.fill_(token)
        step.at.set(pos0)

        def run(at: DevicePos | None = step.at) -> tuple[torch.Tensor, list[dict]]:
            if self.tp > 1:
                return tp_forward_hidden(self.model, step.ids, caches, self.kernels.linear, pos0, self.rccl,
                                         act_dtype=self._dtype(), at=at)
            return forward_hidden(self.model, step.ids, caches, self.kernels.linear, pos0, self._dtype(), at=at)

        with torch.inference_mode():
            if step.graph is not None:
                step.graph.replay()
                hidden = step.hidden
            elif not step.warm:
                hidden, fresh = run()
                for held, new in zip(caches, fresh):      # a state that was not yet an fp32 buffer now is one
                    held.update(new)
                step.warm = True
            else:
                graph = torch.cuda.CUDAGraph()
                failed = None
                try:
                    with torch.cuda.graph(graph):
                        step.hidden = run()[0]
                except Exception as exc:  # noqa: BLE001 - any capture failure: this engine stays eager
                    failed = exc
                if self._any_rank(failed is not None):   # the ranks replay together or not at all
                    print(f"[tensorfold] decode graph capture failed, decoding eagerly: {failed}", file=sys.stderr)
                    self.graphs, self._step = False, None
                    torch.cuda.synchronize()
                    return run(None)[0], caches              # the eager step advances each cache itself
                step.graph = graph
                graph.replay()
                hidden = step.hidden
            hidden = hidden.clone()               # the graph's output is overwritten by the next replay
        for cache in caches:
            if "len" in cache:
                cache["len"] = pos0 + 1
        return hidden, caches

    def _any_rank(self, flag: bool) -> bool:
        """True on every rank when ``flag`` is true on one (one rank: ``flag``)."""

        if self.tp <= 1:
            return flag
        value = torch.tensor([int(flag)], dtype=torch.int32, device=self._device())
        self.rccl.all_reduce(value, value, op="max")
        return bool(value.item())

    def _span(self, tokens: Sequence[int], caches: list[dict] | None, pos0: int, total: int,
              ) -> tuple[torch.Tensor, list[dict]]:
        """One forward of ``tokens`` starting at ``pos0``. Buffers grow to ``total`` positions."""

        device, dtype = self._device(), self._dtype()
        if caches is None:
            caches = _blank_caches(self.model, 1, total, device, dtype)
        else:
            caches = _grow(caches, total, dtype, device)
        return self._forward(tokens, caches, pos0)

    def _cuts(self, prompt: Sequence[int], cached: int) -> list[int]:
        """Positions past ``cached`` where this prefill keeps a state."""

        from tensorfold.cuda.markers import MIN_GAP

        stops = []
        if self.points is not None:
            stops.extend(int(point) for point in self.points(prompt) if cached < int(point) < len(prompt))
        stops.sort()
        end = entry_end(prompt)
        # A message start already next to the prompt end covers that boundary.
        near = bool(stops) and len(prompt) - stops[-1] < MIN_GAP
        if cached < end < len(prompt) and not near:
            stops.append(end)
        return sorted(set(stops))

    def _remember(self, ids: Sequence[int], caches: list[dict]) -> None:
        if self.cache.keep <= 0 or self.byte_budget == 0:
            return
        cloned = clone_caches(caches)
        self.cache.add(list(ids), cloned, cache_bytes(cloned))
        trim_bytes(self.cache, self.byte_budget)

    def _prefill(self, prompt: Sequence[int], caches: list[dict] | None, cached: int, total: int, *,
                 store: bool) -> tuple[torch.Tensor, list[dict]]:
        """Prefill ``prompt[cached:]``. Stored cuts split the forward so a resume repeats those launches."""

        cuts = self._cuts(prompt, cached) if store else []
        bounds = [cached, *cuts, len(prompt)]
        hidden = None
        for start, stop in zip(bounds, bounds[1:]):
            if start == stop:
                continue
            hidden, caches = self._span(list(prompt[start:stop]), caches, start, total)
            if store and stop in cuts:
                self._remember(prompt[:stop], caches)
        if hidden is None:
            raise RuntimeError("a prefill received no tokens")
        return hidden, caches

    def _sample(self, hidden: torch.Tensor, sampling: Sampling | None, position: int, constraint) -> int:
        local = _project(hidden[:, -1], self.model.output_head(), self.kernels.linear)
        logits = vocab_gather(self.rccl, local) if self.tp > 1 else local     # every rank joins the gather
        if self.rank != 0:
            return self._share([0])[0]
        if constraint is not None:
            constraint.mask(logits)
        row = logits.detach().float().reshape(-1)
        if sampling is None or float(sampling.temperature) <= 0.0:
            token = int(torch.argmax(row).item())
        else:
            width = int(row.shape[0])
            k = int(sampling.top_k)
            if k:
                count = min(width, k + MARGIN)
                values, index = torch.topk(row, count, sorted=False)
                token = choose(values.cpu().numpy(), index.cpu().numpy().astype(np.int64), position, sampling)
            else:
                token = choose(row.cpu().numpy(), np.arange(width, dtype=np.int64), position, sampling)
        if constraint is not None:
            constraint.advance([token])
        return self._share([token])[0]

    def _share(self, values: list[int]) -> list[int]:
        """Rank 0's ``values`` on every rank. Other ranks pass placeholders of the same length."""

        if self.tp <= 1:
            return values
        buffer = torch.tensor(values, dtype=torch.int64, device=self._device())
        self.rccl.broadcast(buffer, buffer, root=0)
        return [int(value) for value in buffer.tolist()]

    def generate(self, prompt: list[int], max_tokens: int, sampling: Sampling | None,
                 on_tokens: Callable[[list[int]], bool | None], *, stop_eos: bool = True, draft: bool = True,
                 constraint=None, vision=None, background: bool = False) -> dict[str, int]:
        """One prompt; ``draft=False`` skips the prefix cache. Returns ``{'cached': reused length}``."""

        del background
        if vision is not None:
            raise ValueError("image inputs are not served on ROCm")
        if not prompt:
            raise ValueError("prompt is empty")
        if self.context_window is not None and len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token window")
        room = int(max_tokens)
        if self.context_window is not None:
            room = min(room, self.context_window - len(prompt))
        if room < 1:
            raise ValueError("max_tokens must leave room for one token")
        depth = self.mtp_depth if draft and self.mtp is not None and not self.no_drafts else 0
        if self.tp > 1:
            self._post(list(prompt), room, draft, depth)
        return self._run(list(prompt), room, sampling, on_tokens, stop_eos, draft, constraint, depth)

    def _run(self, prompt: list[int], room: int, sampling: Sampling | None, on_tokens, stop_eos: bool,
             draft: bool, constraint, depth: int) -> dict[str, int]:
        hit = self.cache.longest(prompt) if draft else None
        cached = len(hit[0]) if hit is not None else 0
        held = clone_caches(hit[1]) if hit is not None else None
        # The reply ends at ``room`` tokens and a round forwards at most ``depth + 1``: the cache never holds more.
        total = len(prompt) + room + depth + 1
        hidden, caches = self._prefill(prompt, held, cached, total, store=draft)
        ends = set(self.eos)
        position = len(prompt)
        nxt = self._sample(hidden, sampling, position, constraint)
        emitted = 0

        def emit(token: int) -> bool:
            """Count one token against ``room`` and hand it to the client on rank 0. True ends the reply."""

            nonlocal emitted
            emitted += 1
            if self.rank != 0:
                return emitted >= room
            stop = bool(on_tokens([token])) if on_tokens is not None else False
            ended = stop_eos and token in ends
            grammar_done = constraint is not None and bool(getattr(constraint, "finished", False))
            return emitted >= room or stop or ended or grammar_done

        while True:
            done = emit(nxt)
            if self.tp > 1:
                done = bool(self._share([int(done)])[0])
            if done:
                break
            extra, hidden, caches, position, nxt = self._decode_step(
                hidden, caches, position, nxt,
                sampling=sampling, constraint=constraint, depth=depth,
            )
            for tok in extra:
                done = emit(tok)
                if done:
                    break
            # Every rank joins one vote a round, so a stop inside the drafts ends the reply on every rank.
            if self.tp > 1:
                done = bool(self._share([int(done)])[0])
            if done:
                break
        self._step = None                         # the reply's graph holds its caches
        if len(prompt) > _LONG_PROMPT:
            torch.cuda.empty_cache()              # a long prefill's freed blocks go back to the runtime's scratch
        return {"cached": cached}

    def _decode_step(self, hidden: torch.Tensor, caches: list[dict], position: int, last_token: int, *,
                     sampling: Sampling | None, constraint,
                     depth: int) -> tuple[list[int], torch.Tensor, list[dict], int, int]:
        """One round: returns accepted drafts, hidden, caches, the next token's slot and the next token."""
        if self.mtp is None or depth <= 0:
            hidden, caches = self._forward([last_token], caches, position, decode=True)
            position += 1
            nxt = self._sample(hidden, sampling, position, constraint)
            return [], hidden, caches, position, nxt

        dtype = self._dtype()
        hidden, caches = self._forward([last_token], caches, position, decode=True)
        position += 1
        mtp_state = self.mtp.fresh_cache(batch=1,
                                         total=position + max(0, depth) + 1,
                                         device=self._device(), dtype=dtype)
        # A greedy request and a follower draft greedily; verification keeps the request's own sampling.
        drafts = self.mtp.draft_chain(hidden[:, -1:], last_token, position - 1, depth, mtp_state,
                                      sampling=sampling or Sampling(seed=0, temperature=0.0), dtype=dtype)
        if self.tp > 1:
            # A rank's cache advances by how many drafts matched, so every rank takes rank 0's chain.
            drafts = self._share(drafts)
        extra: list[int] = []
        cur_hidden = hidden
        for i, d in enumerate(drafts):
            # The serial path keys a token by its own slot; the verifier has to draw the same way.
            nxt = self._sample(cur_hidden, sampling, position + i, constraint)
            if nxt != d:
                return extra, cur_hidden, caches, position + i, nxt
            cur_hidden, caches = self._forward([d], caches, position + i, decode=True)
            extra.append(d)
        nxt = self._sample(cur_hidden, sampling, position + depth, constraint)
        return extra, cur_hidden, caches, position + depth, nxt

    def _post(self, prompt: list[int], room: int, draft: bool, depth: int) -> None:
        """Rank 0 publishes a request with its draft depth and graph choice, so every rank runs the same calls."""

        store = self.rccl.store
        store.set(f"tf_request/{self._posted}", json.dumps([prompt, room, bool(draft), int(depth), bool(self.graphs)]))
        if self._posted:
            store.delete_key(f"tf_request/{self._posted - 1}")
        self._posted += 1

    def close(self) -> None:
        """Rank 0 tells the other ranks there are no more requests."""

        if self.tp > 1 and self.rank == 0:
            store = self.rccl.store
            store.set(f"tf_request/{self._posted}", json.dumps(None))
            self._posted += 1

    def follow(self) -> None:
        """Ranks above 0: run each of rank 0's requests in step with it. Returns once rank 0's store closes."""

        if self.rank == 0:
            raise RuntimeError("rank 0 serves requests; follow() is for the other ranks")
        while True:
            key = f"tf_request/{self._posted}"
            try:
                self.rccl.store.wait([key])
            except Exception as exc:  # noqa: BLE001 - the store's wait timeout: rank 0 is idle
                text = str(exc).lower()
                if "timeout" in text:
                    continue
                if "recv" in text or "connection" in text or "broken pipe" in text:
                    return                                  # rank 0 closed the store: the server stopped
                raise
            request = json.loads(self.rccl.store.get(key))
            if request is None:                             # rank 0 closed: no more requests
                return
            prompt, room, draft, depth, self.graphs = request
            self._posted += 1
            self._run(prompt, room, None, None, False, draft, None, depth if self.mtp is not None else 0)

    def prefill_caches(self, prompt: Sequence[int]) -> list[dict]:
        """Caches after one forward of ``prompt``. A stored prefix of that length matches this."""

        _, caches = self._span(list(prompt), None, 0, len(prompt))
        return caches


def caches_equal(left: list[dict], right: list[dict]) -> bool:
    return _caches_equal(left, right)
