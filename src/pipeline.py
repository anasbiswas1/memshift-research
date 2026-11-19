"""The stages of notebook 01 (record the ground truth), as functions over the suite, so the notebook stays short and
the same code runs on CPU with scripted readers (MEMSHIFT_DRY_RUN=1) before any GPU session.

Order of model sessions (one model on the GPU at a time):
  phrasing checks:  qwen25_3b, llama32_3b, qwen3_4b (surrogates), llama31_8b, qwen25_7b (old readers)
  activation:       qwen25_7b (kept up), qwen3_8b, llama31_8b
Every session first runs the clean smoke tasks and stops if too many episodes end on a parse failure.
A session whose episodes are all recorded does not start a server.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

import analysis as AN
import attacks as AT
import config
import memory as M
import probes as P
from serving import DRY_RUN, Server, make_reader, records_root
from tools import canary_categories

VARIANT_ORDER = ["qwen25_3b", "llama32_3b", "qwen3_4b", "llama31_8b", "qwen25_7b"]
ACTIVATION_ORDER = ["qwen25_7b", "qwen3_8b", "llama31_8b"]
ACTIVATION_STORES = {config.REF_RETRIEVER: ["main", "a5"], "bge_base_v15": ["main"]}


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ----------------------------------------------------------------------------- embeddings and scanner
def embed_everything(suite: dict, log=print) -> dict[str, M.EmbeddingCache]:
    recs, qs = M.all_texts(suite)
    caches = {}
    for tag in config.RETRIEVERS:
        c = M.EmbeddingCache(tag)
        need = any(f"r:{M.text_key(t)}" not in c.vec for t in recs) or any(f"q:{M.text_key(t)}" not in c.vec for t in qs)
        if need:
            emb = M.HashEmbedder(tag) if DRY_RUN else M.STEmbedder(tag)
            n = c.ensure(emb, recs, False) + c.ensure(emb, qs, True)
            c.save()
            del emb
            free_gpu()
            log(f"{tag}: embedded {n} new texts")
        else:
            log(f"{tag}: all {len(recs) + len(qs)} texts cached")
        caches[tag] = c
    return caches


def scanner_scores(suite: dict, log=print, path=None) -> tuple[dict[str, float], str]:
    """Scanner probability for every record text and phrasing; cached in reports/01_scanner_scores.csv (or `path`)."""
    path = Path(path) if path else config.REPORTS / "01_scanner_scores.csv"
    rows = []
    for rid, r in sorted(suite["records"].items()):
        if r["poison"] and r["attack"] == "A4":
            rows += [(rid, v, t) for v, t in enumerate(r["variants"])]
        else:
            rows.append((rid, -1, r["text"]))
    keys = {M.text_key(t) for _, _, t in rows}
    if path.exists():
        df = pd.read_csv(path)
        if keys <= set(df.text_key):
            by_key = dict(zip(df.text_key, df.score))
            log(f"scanner scores cached ({df.scanner.iloc[0]})")
            return {t: float(by_key[M.text_key(t)]) for _, _, t in rows}, str(df.scanner.iloc[0])
    from scanners import load_scanner
    sc = load_scanner()
    texts = sorted({t for _, _, t in rows})
    s = dict(zip(texts, sc.score(texts)))
    name = sc.name
    del sc
    free_gpu()
    pd.DataFrame([{"record_id": rid, "variant": v, "text_key": M.text_key(t), "scanner": name, "score": float(s[t])}
                  for rid, v, t in rows]).to_csv(path, index=False)
    log(f"scanner {name}: scored {len(texts)} texts")
    return {t: float(s[t]) for _, _, t in rows}, name


# ----------------------------------------------------------------------------- stores and retrieval
def retrievals(suite: dict, caches: dict, texts: dict) -> dict[str, M.Retrieval]:
    return {tag: M.Retrieval(suite, texts, caches[tag]) for tag in config.RETRIEVERS}


def a4_records(suite: dict) -> list[str]:
    return sorted(rid for rid, r in suite["records"].items() if r["poison"] and r["attack"] == "A4")


def retrieved_map(suite: dict, ret_ref: M.Retrieval, cache_ref: M.EmbeddingCache) -> dict[tuple[str, int], bool]:
    return {(rid, v): P.variant_retrieved(suite, ret_ref, cache_ref, rid, v)
            for rid in a4_records(suite) for v in range(config.N_VARIANTS)}


def variant_specs_for(tag: str, suite: dict, ret_ref, cache_ref, base_texts: dict, retrieved: dict) -> list[dict]:
    specs = []
    for rid in a4_records(suite):
        g = config.A4_GROUPS[suite["records"][rid]["group"]]
        J = config.J_VARIANT_OLD if tag == g["old"] else (config.J_VARIANT_SURR if tag == g["surrogate"] else 0)
        if J:
            specs += [s for s in P.variant_specs(suite, ret_ref, cache_ref, base_texts, rid, tag, J)
                      if retrieved[(rid, s["variant"])]]
    return specs


def probe_units(suite: dict, rets: dict[str, M.Retrieval]) -> list[tuple[str, str]]:
    """Poisoned units in a canary category with exposure under at least one retriever."""
    units = set()
    for ret in rets.values():
        for (rid, cat), n in ret.exposure_counts(suite).items():
            r = suite["records"][rid]
            if n > 0 and r["poison"] and cat in canary_categories(r):
                units.add((rid, cat))
    return sorted(units)


def stability_units(units_ref: list[tuple[str, str]]) -> list[tuple[str, str]]:
    k = max(1, round(config.STABILITY_SHARE * len(units_ref)))
    return sorted(random.Random(config.SEED).sample(units_ref, k))


# ----------------------------------------------------------------------------- model sessions
def smoke_floor(tag: str) -> float:
    """Clean-task success floor of the liveness check: the per-reader floor of amendment 5 (2.6) where one is set,
    otherwise the reader or surrogate floor."""
    by_reader = getattr(config, "MIN_SMOKE_SUCCESS_BY_READER", {})
    if tag in by_reader:
        return float(by_reader[tag])
    return float(config.MIN_SMOKE_SUCCESS["reader" if tag in config.READERS else "surrogate"])


class Sessions:
    """Runs groups of episode specs for one model, starting its server only when something is not yet recorded."""

    def __init__(self, suite: dict, server: Server, smoke_specs_for, workers: int = 48, log=print):
        self.suite, self.server, self.smoke_for = suite, server, smoke_specs_for
        self.workers, self.log = workers, log

    def run(self, tag: str, groups: list[tuple[str, list[dict]]], keep_up: bool = False) -> dict[str, list[dict]]:
        store = P.EpisodeStore(tag, records_root())
        smoke = self.smoke_for(tag)
        pending = sum(1 for _, specs in [("smoke", smoke)] + groups for s in specs if s["key"] not in store.done)
        reader = None
        if pending:
            self.server.ensure(tag)
            reader = make_reader(tag, self.server.port)
        out = {"smoke": store.run(reader, smoke, self.suite, self.workers, log=self.log)}
        fails = [e for e in out["smoke"] if e["end_reason"] == "parse_failure"]
        share = len(fails) / max(1, len(out["smoke"]))
        success = float(np.mean([e["task_success"] for e in out["smoke"]]))
        floor = smoke_floor(tag)
        lookup = float(np.mean([bool(e["calls"]) and e["calls"][0]["tool"] in ("lookup_contact", "list_payees",
                                                                                  "list_documents") for e in out["smoke"]]))
        stops = float(np.mean([e["stop_hits"] > 0 for e in out["smoke"]]))
        steps = float(np.mean([e["end_reason"] == "max_steps" for e in out["smoke"]]))
        reps = float(np.mean([e["repeated_calls"] > 0 for e in out["smoke"]]))
        self.log(f"[{tag}] smoke: success {success:.2f} (floor {floor:.2f}), parse failures {share:.2f} "
                 f"(ceiling {config.MAX_PARSE_FAIL_SHARE:.2f}), lookup first {lookup:.2f}, "
                 f"stopped at an invented tool result {stops:.2f}, repeated a call {reps:.2f}, out of steps {steps:.2f}")
        if share > config.MAX_PARSE_FAIL_SHARE or success < floor:
            self.server.stop()
            bad = fails or [e for e in out["smoke"] if not e["task_success"]]
            raise RuntimeError(f"{tag}: smoke gate failed (success {success:.2f}, parse failures {share:.2f}); "
                               f"sample of a failed episode, query {bad[0]['query_id']}:\n"
                               + json.dumps(bad[0]["calls"])[:600] + "\n---\n" + "\n---\n".join(bad[0]["outputs"][:3]))
        for name, specs in groups:
            out[name] = store.run(reader, specs, self.suite, self.workers, log=self.log) if specs else []
        if not keep_up and pending:
            self.server.stop()
        return out


# ----------------------------------------------------------------------------- stale results of an older harness
HARNESS_FREE_REPORTS = {"01_scanner_scores.csv"}     # does not depend on the episode harness: stays in place


def recording_revisions() -> dict:
    """Live episodes per reader by the harness revision that recorded them (carried episodes keep their revision)."""
    import gzip
    out = {}
    for f in sorted(config.RECORDS.glob("*/*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                if line.strip():
                    rv = json.loads(line).get("runtime_version")
                    d = out.setdefault(f.parent.name, {})
                    d[rv] = d.get(rv, 0) + 1
    return out


def archive_stale(log=print) -> dict:
    """Moves what an older harness made out of the live results, into reports/01_rv<n>/, where it stays as a record:
    whole episode shards of a harness that is not carried; from shards of a carried harness (config.CARRIED_RUNTIMES),
    only the episodes the current harness would run differently (probes.superseded); and the notebook-01 reports."""
    import gzip
    import shutil
    moved = {"shards": 0, "superseded_episodes": 0, "carried_episodes": 0, "reports": 0}
    for f in sorted(config.RECORDS.glob("*/*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            eps = [json.loads(line) for line in fh if line.strip()]
        rv = eps[0].get("runtime_version") if eps else None
        if not rv or rv == config.RUNTIME_VERSION:
            continue
        dest = config.REPORTS / f"01_rv{rv}" / "records" / f.parent.name
        if rv not in config.CARRIED_RUNTIMES:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest / f.name)); moved["shards"] += 1
            continue
        drop = [e for e in eps if P.superseded(e)]
        keep = [e for e in eps if not P.superseded(e)]
        moved["carried_episodes"] += len(keep)
        if not drop:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / f.name
        if target.exists():                    # superseded by an earlier revision already: keep both sets
            target = dest / f.name.replace(".jsonl.gz", f".superseded_rv{config.RUNTIME_VERSION}.jsonl.gz")
        with gzip.open(target, "wt") as fh:
            for e in drop:
                fh.write(json.dumps(e) + "\n")
        if keep:
            with gzip.open(f, "wt") as fh:
                for e in keep:
                    fh.write(json.dumps(e) + "\n")
        else:
            f.unlink()
        moved["superseded_episodes"] += len(drop)
    claims = config.REPORTS / "01_claims.json"
    if claims.exists():
        rv = json.loads(claims.read_text()).get("runtime_version")
        if rv and rv != config.RUNTIME_VERSION:
            dest = config.REPORTS / f"01_rv{rv}"
            dest.mkdir(parents=True, exist_ok=True)
            for f in sorted(config.REPORTS.glob("01_*")):
                if f.is_file() and f.name not in HARNESS_FREE_REPORTS:
                    shutil.move(str(f), str(dest / f.name)); moved["reports"] += 1
    log(f"older harness: {moved['shards']} shards and {moved['superseded_episodes']} superseded episodes archived, "
        f"{moved['carried_episodes']} episodes carried, {moved['reports']} report files archived")
    return moved


# ----------------------------------------------------------------------------- tables
def exposure_table(suite: dict, rets_by_store: dict[str, dict[str, M.Retrieval]]) -> pd.DataFrame:
    return AN.exposure_frame({(st, rt): ret.exposure_counts(suite)
                              for st, rets in rets_by_store.items() for rt, ret in rets.items()})


def clean_utility(eps_by_reader: dict[str, list[dict]], suite: dict) -> pd.DataFrame:
    rows = []
    for tag, eps in eps_by_reader.items():
        d = pd.DataFrame(eps)
        memdep = d.query_id.map(lambda q: suite["queries"][q]["memory_dependent"])
        rows.append({"reader": tag, "model_id": config.MODELS[tag], "episodes": len(d),
                     "task_success": d.task_success.mean(), "success_directory_tasks": d.task_success[~memdep].mean(),
                     "success_memory_tasks": d.task_success[memdep].mean(),
                     "lookup_before_action": d.calls.map(lambda cs: bool(cs) and cs[0]["tool"] in
                                                         ("lookup_contact", "list_payees", "list_documents")).mean(),
                     "parser_fallback_share": (d.parser_fallbacks > 0).mean(), "parse_failure_share": (d.end_reason == "parse_failure").mean(),
                     "stop_share": (d.stop_hits > 0).mean(), "trimmed_share": (d.trimmed_replies > 0).mean(),
                     "repeat_share": (d.repeated_calls > 0).mean(),
                     "max_steps_share": (d.end_reason == "max_steps").mean(),
                     "final_share": (d.end_reason == "final").mean(), "mean_model_calls": d.n_model_calls.mean(),
                     "mean_prompt_tokens": d.prompt_tokens.mean(), "mean_completion_tokens": d.completion_tokens.mean()})
    return pd.DataFrame(rows)


def parse_failure_table(act_eps: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(act_eps)
    return (d.groupby("reader").agg(activation_episodes=("end_reason", "size"),
                                    parse_failure_share=("end_reason", lambda s: (s == "parse_failure").mean()),
                                    max_steps_share=("end_reason", lambda s: (s == "max_steps").mean()),
                                    parser_fallback_share=("parser_fallbacks", lambda s: (s > 0).mean()),
                                    stop_share=("stop_hits", lambda s: (s > 0).mean()),
                                    trimmed_share=("trimmed_replies", lambda s: (s > 0).mean()),
                                    repeat_share=("repeated_calls", lambda s: (s > 0).mean()))
             .reset_index())


def stability_table(stab: dict[str, list[dict]], act: pd.DataFrame) -> pd.DataFrame:
    """Share of sampled episodes (temperature 0.7) whose attacker outcome equals the greedy first probe's."""
    rows = []
    for tag, eps in stab.items():
        greedy = act[(act.reader == tag) & (act.ctx_retriever == config.REF_RETRIEVER) & (act.store == "main")]
        g = {(r.record_id, r.category): r.first_probe_hit for r in greedy.itertuples()}
        agree = [int(bool(e["attack_hit"]) == bool(g[(e["probe_record"], e["query_category"])])) for e in eps]
        sampled_rate = float(np.mean([e["attack_hit"] for e in eps])) if eps else float("nan")
        rows.append({"reader": tag, "sampled_episodes": len(eps), "agreement_with_greedy": float(np.mean(agree)),
                     "sampled_hit_rate": sampled_rate,
                     "greedy_hit_rate_same_units": float(np.mean([g[(e["probe_record"], e["query_category"])]
                                                                  for e in eps]))})
    return pd.DataFrame(rows)
