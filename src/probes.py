"""Episode specifications (which query, which memories in the context, whose canary counts) and the record-once
episode store. Every episode is keyed by what the reader sees, so an identical episode is never run twice and a run
interrupted by a disconnect resumes where it stopped.

Activation of a unit u = (record i, category c) (plan v1, Equation 2): J queries of category c from the record's
own user, ranked by (retrieved by the context retriever, names the record's target, similarity); the context is the
top-k for that query with record i placed last if the retriever did not return it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from agent import run_episode
from memory import EmbeddingCache, Retrieval, topk_with_override


def episode_key(reader: str, query_id: str, memory_texts: list[str], canaries: list[str], temperature: float,
                seed: int) -> str:
    blob = json.dumps([config.RUNTIME_VERSION, reader, query_id, memory_texts, sorted(canaries), temperature, seed])
    return hashlib.sha1(blob.encode()).hexdigest()


def _spec(kind, reader, ctx_retriever, store, q, probe, variant, context, texts, canaries, temperature=0.0, seed=0):
    mem = [texts[r] for r in context]
    return {"kind": kind, "reader": reader, "ctx_retriever": ctx_retriever, "store": store, "user_id": q["user_id"],
            "query_id": q["query_id"], "query_category": q["category"], "probe_record": probe, "variant": variant, "context": context,
            "memory_texts": mem, "canaries": canaries, "temperature": temperature, "seed": seed,
            "key": episode_key(reader, q["query_id"], mem, canaries, temperature, seed)}


def rank_queries(suite: dict, ret: Retrieval, rid: str, cat: str, override_text: str | None = None,
                 cache: EmbeddingCache | None = None) -> list[tuple[str, list[str]]]:
    """Category-c queries of the record's user, best probes first, each with its top-k context."""
    r = suite["records"][rid]
    uid, tgt = r["user_id"], r.get("target")
    rows = []
    for qid in ret.by_user[uid]["qids"]:
        q = suite["queries"][qid]
        if q["category"] != cat:
            continue
        if override_text is None:
            ctx, s = ret.topk(qid, uid), ret.sim(qid, rid, uid)
        else:
            ctx, s = topk_with_override(suite, ret, cache, uid, qid, rid, override_text)
        rows.append((rid in ctx, bool(tgt and tgt in q["text"]), s, qid, ctx))
    rows.sort(key=lambda x: (not x[0], not x[1], -x[2], x[3]))
    return [(qid, ctx) for _, _, _, qid, ctx in rows]


def force_in(ctx: list[str], rid: str, k: int = config.TOP_K) -> list[str]:
    return list(ctx) if rid in ctx else list(ctx[:k - 1]) + [rid]


def activation_specs(suite, ret, texts, units, reader, ctx_retriever, store, J=config.J_PROBES):
    specs = []
    for rid, cat in units:
        canary = suite["records"][rid]["canary"]
        for j, (qid, ctx) in enumerate(rank_queries(suite, ret, rid, cat)[:J]):
            s = _spec("activation", reader, ctx_retriever, store, suite["queries"][qid], rid, None,
                      force_in(ctx, rid), texts, [canary])
            s["probe_index"] = j
            specs.append(s)
    return specs


def variant_specs(suite, ret, cache, base_texts, rid, reader, J):
    """Every phrasing of one A4 record, probed in its own category with the phrasing in the store."""
    r = suite["records"][rid]
    specs = []
    for v, vt in enumerate(r["variants"]):
        texts = dict(base_texts)
        texts[rid] = vt
        for qid, ctx in rank_queries(suite, ret, rid, r["category"], override_text=vt, cache=cache)[:J]:
            specs.append(_spec("variant", reader, config.REF_RETRIEVER, "variant", suite["queries"][qid], rid, v,
                               force_in(ctx, rid), texts, [r["canary"]]))
    return specs


def variant_retrieved(suite, ret, cache, rid, v) -> bool:
    """Whether phrasing v of record rid is retrieved by at least one query of its category (reference retriever)."""
    r = suite["records"][rid]
    vt = r["variants"][v]
    return any(rid in ctx for _, ctx in rank_queries(suite, ret, rid, r["category"], override_text=vt, cache=cache))


def clean_sample(suite, n: int, seed: int = config.SEED) -> list[str]:
    qids = sorted(suite["queries"])
    return random.Random(seed).sample(qids, n)


def clean_specs(suite, ret_clean, texts_clean, qids, reader):
    return [_spec("clean", reader, config.REF_RETRIEVER, "clean", suite["queries"][q], None, None,
                  ret_clean.topk(q, suite["queries"][q]["user_id"]), texts_clean, []) for q in qids]


def stability_specs(suite, ret, texts, units, reader):
    specs = []
    for rid, cat in units:
        qid, ctx = rank_queries(suite, ret, rid, cat)[0]
        for s in range(1, config.STABILITY_SAMPLES + 1):
            specs.append(_spec("stability", reader, config.REF_RETRIEVER, "main", suite["queries"][qid], rid, None,
                               force_in(ctx, rid), texts, [suite["records"][rid]["canary"]],
                               temperature=config.STABILITY_TEMPERATURE, seed=s))
    return specs


OUTCOME_KEYS = ["calls", "final", "end_reason", "n_model_calls", "parse_failures", "prompt_tokens", "completion_tokens",
                "attack_hit", "task_success", "n_consequential", "outputs", "seconds", "runtime_version", "model_id"]


class EpisodeStore:
    """Recorded episodes of one reader: reports/01_records/<reader>/<kind>_<stamp>_<n>.jsonl.gz."""

    def __init__(self, reader: str, root: Path | None = None):
        self.reader = reader
        self.dir = Path(root or config.RECORDS) / reader
        self.dir.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        for f in sorted(self.dir.glob("*.jsonl.gz")):
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    e = json.loads(line)
                    if e.get("runtime_version") == config.RUNTIME_VERSION:
                        self.done[e["key"]] = e

    def _flush(self, kind: str, buf: list[dict]):
        if not buf:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        n = len(list(self.dir.glob(f"{kind}_*.jsonl.gz")))
        with gzip.open(self.dir / f"{kind}_{stamp}_{n:04d}.jsonl.gz", "wt") as fh:
            for e in buf:
                fh.write(json.dumps(e) + "\n")

    def run(self, reader_obj, specs: list[dict], suite: dict, workers: int = 32, flush_every: int = 200,
            log=print) -> list[dict]:
        """Runs the specs not yet recorded; returns every spec's episode (recorded or new), in spec order."""
        todo, seen = [], set()
        for s in specs:
            if s["key"] not in self.done and s["key"] not in seen:
                todo.append(s); seen.add(s["key"])
        kind = specs[0]["kind"] if specs else "none"
        log(f"[{self.reader}] {kind}: {len(specs)} specs, {len(specs) - len(todo)} recorded, {len(todo)} to run")
        lock, buf, t0 = threading.Lock(), [], time.time()

        def one(s):
            user = suite["users"][s["user_id"]]
            ep = run_episode(reader_obj, user, suite["queries"][s["query_id"]], s["memory_texts"], s["canaries"],
                             temperature=s["temperature"], seed=s["seed"])
            return {**{k: v for k, v in s.items() if k != "memory_texts"}, **ep,
                    "runtime_version": config.RUNTIME_VERSION, "model_id": config.MODELS.get(self.reader, self.reader)}

        done_n = 0
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(one, s) for s in todo]
                for f in as_completed(futs):
                    e = f.result()
                    with lock:
                        self.done[e["key"]] = e
                        buf.append(e); done_n += 1
                        if len(buf) >= flush_every:
                            self._flush(kind, buf); buf = []
                            rate = done_n / max(1e-9, time.time() - t0)
                            log(f"  {done_n}/{len(todo)} episodes ({rate:.1f}/s)")
        finally:
            self._flush(kind, buf)      # episodes finished before an error or a disconnect are kept
        # metadata from the requesting spec, outcome from the recorded episode: an identical episode recorded for
        # another purpose (a phrasing check that equals an activation probe) is reused, not re-run
        return [{**{k: v for k, v in s.items() if k != "memory_texts"},
                 **{k: self.done[s["key"]][k] for k in OUTCOME_KEYS}} for s in specs]
