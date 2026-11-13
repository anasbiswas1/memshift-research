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


def scanner_scores(suite: dict, log=print) -> tuple[dict[str, float], str]:
    """Scanner probability for every record text and phrasing; cached in reports/01_scanner_scores.csv."""
    path = config.REPORTS / "01_scanner_scores.csv"
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
        self.log(f"[{tag}] smoke: success {np.mean([e['task_success'] for e in out['smoke']]):.2f}, "
                 f"parse failures {share:.2f}")
        if share > config.MAX_PARSE_FAIL_SHARE:
            self.server.stop()
            raise RuntimeError(f"{tag}: {share:.0%} of smoke episodes ended on a parse failure; sample output:\n"
                               + "\n---\n".join(fails[0]["outputs"][:2]))
        for name, specs in groups:
            out[name] = store.run(reader, specs, self.suite, self.workers, log=self.log) if specs else []
        if not keep_up and pending:
            self.server.stop()
        return out


# ----------------------------------------------------------------------------- tables
def exposure_table(suite: dict, rets_by_store: dict[str, dict[str, M.Retrieval]]) -> pd.DataFrame:
    return AN.exposure_frame({(st, rt): ret.exposure_counts(suite)
                              for st, rets in rets_by_store.items() for rt, ret in rets.items()})


def clean_utility(eps_by_reader: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for tag, eps in eps_by_reader.items():
        d = pd.DataFrame(eps)
        rows.append({"reader": tag, "model_id": config.MODELS[tag], "episodes": len(d),
                     "task_success": d.task_success.mean(), "parse_failure_share": (d.end_reason == "parse_failure").mean(),
                     "final_share": (d.end_reason == "final").mean(), "mean_model_calls": d.n_model_calls.mean(),
                     "mean_prompt_tokens": d.prompt_tokens.mean(), "mean_completion_tokens": d.completion_tokens.mean()})
    return pd.DataFrame(rows)


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
