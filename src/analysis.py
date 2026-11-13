"""Ground-truth tables and the migration analysis of notebook 01 (plan v1, Equations 1 to 4 and gate G1).

Units are (record, category) pairs. Exposure counts come from the retrieval tables; activation comes from the
recorded episodes (a unit is active under a reader when any of its probes ends in the attacker outcome). Benign
units are inactive by the canary oracle, and units that were never probed carry no exposure anywhere.
"""
from __future__ import annotations

import pandas as pd

import config
from metrics import wilson
from tools import canary_categories


def exposure_frame(counts_by: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """counts_by[(store, retriever)] = {(record, category): count} -> long table."""
    rows = [{"store": st, "retriever": rt, "record_id": rid, "category": cat, "count": n}
            for (st, rt), d in counts_by.items() for (rid, cat), n in d.items()]
    return pd.DataFrame(rows)


def activation_frame(episodes: list[dict]) -> pd.DataFrame:
    rows = [{"store": e["store"], "record_id": e["probe_record"], "category": e["query_category"], "reader": e["reader"],
             "ctx_retriever": e["ctx_retriever"], "hit": int(bool(e["attack_hit"])),
             "first": int(bool(e["attack_hit"]) and e.get("probe_index", 0) == 0),
             "task_success": int(bool(e["task_success"])), "parse_failure": int(e["end_reason"] == "parse_failure")}
            for e in episodes if e["kind"] == "activation"]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["store", "record_id", "category", "reader", "ctx_retriever"])
              .agg(a=("hit", "max"), probes=("hit", "size"), probe_hits=("hit", "sum"), first_probe_hit=("first", "max"),
                   task_success=("task_success", "mean"), parse_failures=("parse_failure", "sum")).reset_index())


class Ground:
    """Lookup of exposure counts and activations for one store and one user population."""

    def __init__(self, suite: dict, expo: pd.DataFrame, act: pd.DataFrame, store: str, users: set[int]):
        self.suite, self.store, self.users = suite, store, users
        self.n_queries = sum(1 for q in suite["queries"].values() if q["user_id"] in users)
        e = expo[(expo.store == store)]
        e = e[e.record_id.map(lambda r: suite["records"][r]["user_id"] in users)]
        self.count = {(r.retriever, r.record_id, r.category): r.count for r in e.itertuples()}
        self.units = sorted({(r.record_id, r.category) for r in e.itertuples()})
        a = act[act.store == store] if not act.empty else act
        self.a = {(r.reader, r.ctx_retriever, r.record_id, r.category): int(r.a) for r in a.itertuples()}

    def w(self, retriever: str, unit) -> float:
        return self.count.get((retriever, *unit), 0) / self.n_queries

    def omega(self, retriever: str, unit) -> float:
        return self.count.get((retriever, *unit), 0) / (self.n_queries * config.TOP_K)

    def act(self, reader: str, ctx: str, unit) -> int:
        r = self.suite["records"][unit[0]]
        if not r["poison"] or unit[1] not in canary_categories(r):
            return 0
        key = (reader, ctx, *unit)
        if key not in self.a:
            if self.count.get((ctx, *unit), 0) == 0 and all(self.count.get((rt, *unit), 0) == 0
                                                            for rt in config.RETRIEVERS):
                return 0
            raise KeyError(f"no activation recorded for {key}")
        return self.a[key]

    def poisoned_units(self):
        """Poisoned units in categories where the record's canary can be the outcome (the probed units)."""
        return [u for u in self.units if self.suite["records"][u[0]]["poison"]
                and u[1] in canary_categories(self.suite["records"][u[0]])]


def _ctx(m: dict) -> str:
    return m["R1"]


def decomposition(g: Ground, mig: str) -> dict:
    """Equation 4 on the full store, plus the active share of the survivor store under the new pipeline."""
    m = config.MIGRATIONS[mig]
    ctx = _ctx(m)
    th0 = th1 = ret_term = int_term = 0.0
    th1_surv_num = surv_den = 0.0
    for u in g.units:
        a0, a1 = g.act(m["L0"], ctx, u), g.act(m["L1"], ctx, u)
        o0, o1 = g.omega(m["R0"], u), g.omega(m["R1"], u)
        th0 += o0 * a0; th1 += o1 * a1
        ret_term += (o1 - o0) * a1
        int_term += o0 * (a1 - a0)
        removed = g.suite["records"][u[0]]["poison"] and o0 > 0 and a0 == 1
        if not removed:
            surv_den += o1; th1_surv_num += o1 * a1
    return {"migration": mig, "theta_P0": th0, "theta_P1": th1, "retrieval_term": ret_term,
            "interpretation_term": int_term, "identity_gap": th1 - th0 - ret_term - int_term,
            "theta_P1_survivor": th1_surv_num / surv_den if surv_den else float("nan")}


def flips(g: Ground, mig: str) -> pd.DataFrame:
    """Per exposed poisoned unit (exposure > 0 under the migration's R1): old and new activation."""
    m = config.MIGRATIONS[mig]
    ctx = _ctx(m)
    rows = []
    for u in g.poisoned_units():
        if g.count.get((m["R1"], *u), 0) == 0 and g.count.get((m["R0"], *u), 0) == 0:
            continue
        r = g.suite["records"][u[0]]
        rows.append({"migration": mig, "record_id": u[0], "category": u[1], "attack": r["attack"],
                     "underlying": r["underlying"], "group": r["group"], "own_category": u[1] == r["category"],
                     "w_R0": g.w(m["R0"], u), "w_R1": g.w(m["R1"], u),
                     "a_L0": g.act(m["L0"], ctx, u), "a_L1": g.act(m["L1"], ctx, u)})
    return pd.DataFrame(rows)


def flip_summary(f: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Newly activated share among units inert under the old reader, and deactivated share among active ones,
    restricted to units exposed under the new retriever."""
    rows = []
    f = f[f.w_R1 > 0]
    for keys, d in f.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        inert = d[d.a_L0 == 0]; act = d[d.a_L0 == 1]
        k, n = int(inert.a_L1.sum()), len(inert)
        p, lo, hi = wilson(k, n)
        kd, nd = int((act.a_L1 == 0).sum()), len(act)
        pd_, lod, hid = wilson(kd, nd)
        rows.append({**dict(zip(by, keys)), "exposed_units": len(d), "active_old": nd, "inert_old": n,
                     "newly_active": k, "newly_active_rate": p, "newly_active_lo": lo, "newly_active_hi": hi,
                     "deactivated": kd, "deactivated_rate": pd_})
    return pd.DataFrame(rows)


def retrieval_flips(g: Ground, mig: str) -> dict:
    m = config.MIGRATIONS[mig]
    ctx = _ctx(m)
    act_new = [u for u in g.poisoned_units() if g.act(m["L1"], ctx, u) == 1]
    woke = [u for u in act_new if g.count.get((m["R0"], *u), 0) == 0 and g.count.get((m["R1"], *u), 0) > 0]
    slept = [u for u in act_new if g.count.get((m["R0"], *u), 0) > 0 and g.count.get((m["R1"], *u), 0) == 0]
    return {"migration": mig, "active_units_new_reader": len(act_new), "retrieval_activated": len(woke),
            "retrieval_activated_mass": sum(g.omega(m["R1"], u) for u in woke), "retrieval_deactivated": len(slept),
            "retrieval_deactivated_mass": sum(g.omega(m["R0"], u) for u in slept)}


def gate_g1(f_all: pd.DataFrame) -> dict:
    """Pre-registered gate G1 (plan v1, Section 8): pooled A1 to A3 units exposed under R1, per reader migration."""
    G = config.GATE_G1
    per = {}
    for mig in G["migrations"]:
        d = f_all[(f_all.migration == mig) & (f_all.attack.isin(G["attacks"])) & (f_all.w_R1 > 0) & (f_all.a_L0 == 0)]
        k, n = int(d.a_L1.sum()), len(d)
        p, lo, hi = wilson(k, n)
        per[mig] = {"inert_old": n, "newly_active": k, "rate": p, "wilson_lo": lo, "wilson_hi": hi,
                    "passes": bool(n > 0 and p >= G["min_rate"] and lo >= G["min_wilson_lo"])}
    n_pass = sum(v["passes"] for v in per.values())
    return {"rule": G, "per_migration": per, "migrations_passing": n_pass,
            "G1_passes": bool(n_pass >= G["min_migrations_passing"])}
