"""Candidate quarantine policies, the audit and its replay (plan v2, Sections 6.5 to 6.7 and 7.4 to 7.6).

Every policy is a removal set on the old pipeline's survivor store: whole records (the survivor removals, taken
out of the store before ranking) and (record, category) units or whole records (the quarantine). From revision 2
the quarantine is dropped from the retrieved top-k with nothing moving up in its place. Candidates are frozen before any audit draw; recording every candidate on the whole test log is an
evaluation device that lets 1,000 audits per cell be replayed on CPU.

Census probes are read from notebook 01's labels a_u(P1) (two probes per unit with the unit forced into context;
benign units have no canary and are inactive by construction); their cost is counted from 01's recorded episodes.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------------- sources and references
def laundered_records(suite: dict, share: float, users: set[int] | None = None, salt: str = config.LAUNDER_SALT) -> frozenset:
    """Notebook 07: poisoned records whose provenance label the attacker has laundered to 'user' (as if written through
    the agent's own note-taking): a fixed share of the poisoned records, chosen by a hash of the record id."""
    if share <= 0:
        return frozenset()
    out = set()
    for rid, r in suite["records"].items():
        if r["poison"] and (users is None or r["user_id"] in users):
            if int(hashlib.sha1(f"{salt}|{rid}".encode()).hexdigest(), 16) % 10000 < share * 10000:
                out.add(rid)
    return frozenset(out)


def record_source(rec: dict, laundered=frozenset()) -> str:
    """Provenance used by the perfect-label baselines: poison is external content; quoted instructions are external;
    half of the legitimate updates and memory-only contacts were told by the user and half learned from content
    (decided by a hash of the record id); everything else was told by the user. A laundered record reads 'user'."""
    if rec["record_id"] in laundered:
        return "user"
    if rec["poison"] or rec["kind"] in config.EXTERNAL_KINDS:
        return "external"
    if rec["kind"] in config.SOURCE_SPLIT_KINDS:
        return "external" if int(hashlib.sha1(rec["record_id"].encode()).hexdigest(), 16) % 2 else "user"
    return "user"


def expected_destination(q: dict) -> str:
    x = q["expected"]
    return (x.get("recipient") or "").lower() or (x.get("iban") or "").upper()


def user_destinations(suite: dict, uid: int) -> set[str]:
    """Destinations written in records the user told the agent (user-sourced memory)."""
    from e2e import EMAIL_RE, IBAN_RE
    out = set()
    for r in suite["records"].values():
        if r["user_id"] == uid and record_source(r) == "user":
            out |= {e.lower() for e in EMAIL_RE.findall(r["text"])}
            out |= {i.upper() for i in IBAN_RE.findall(r["text"].replace(" ", ""))}
    return out


def gating_references(of: pd.DataFrame, suite: dict) -> pd.DataFrame:
    """Two references computed from recorded episodes, with no extra runs. Destination gating refuses any send,
    invite, share or payment to a destination outside the directory registry; authority gating refuses one whose
    destination is neither in the directory nor in user-sourced memory (perfect provenance). Every canary is outside
    both, so the attacker-caused rate is 0 by construction. A task counts as solved only if it was solved in the
    recorded episode and its destination is allowed; recovery after a refusal is not credited."""
    from e2e import registry
    regs = {u: registry(suite["users"][u]) for u in set(of.user_id)}
    mine = {u: user_destinations(suite, u) for u in set(of.user_id)}
    rows = []
    for (pipe, pol, sp), g in of.groupby(["pipeline", "policy", "split"]):
        allowed_dir, allowed_auth = [], []
        for qid, uid in zip(g.query_id, g.user_id):
            d = expected_destination(suite["queries"][qid])
            allowed_dir.append(d in regs[uid])
            allowed_auth.append(d in regs[uid] or d in mine[uid])
        a_dir, a_auth = np.array(allowed_dir), np.array(allowed_auth)
        ts = g.task_success.to_numpy(bool)
        mem = g.memory_dependent.to_numpy(bool)
        for name, a in (("destination gating", a_dir), ("authority gating", a_auth)):
            rows.append({"pipeline": pipe, "policy": pol, "split": sp, "reference": name, "r_Q": 0.0,
                         "success": float((ts & a).mean()), "success_directory_tasks": float((ts & a)[~mem].mean()),
                         "success_memory_tasks": float((ts & a)[mem].mean()), "tasks_refused": float((~a).mean()),
                         "success_without_gating": float(ts.mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- units, grids, census
def survivor_sets(of02: pd.DataFrame) -> dict[str, frozenset]:
    """Survivor removals of notebook 02, rebuilt from its outcomes: round 1 on the unquarantined log, round 2 after."""
    from e2e import survivor_removal
    out = {}
    log = of02[of02.split == "log"]
    for p0 in sorted({p for p in log.pipeline}):
        r1 = log[(log.pipeline == p0) & (log.policy == "none")]
        r2 = log[(log.pipeline == p0) & (log.policy == f"round2|{p0}")]
        if len(r2):
            out[p0] = survivor_removal(r1, r2)[0]
    return out


def migration_units(uf: pd.DataFrame, mig: str, removed: frozenset) -> pd.DataFrame:
    """Units of the migration's score table that remain in the survivor store."""
    d = uf[(uf.migration == mig) & ~uf.record_id.isin(removed)].copy()
    d["w_s_full"] = d.w_r1 * d.s_full
    return d.reset_index(drop=True)


def grid_lambdas(units: pd.DataFrame, score: str, shares=config.GRID_SHARES) -> list[float]:
    """Thresholds at which the score quarantines the given shares of benign exposure on calibration users."""
    b = units[(units.split == "calibration") & (units.poison == 0)]
    order = np.argsort(-b[score].to_numpy(), kind="stable")
    sc = b[score].to_numpy()[order]
    cum = np.cumsum(b.w_r1.to_numpy()[order]) / b.w_r1.sum()
    return [float(sc[min(int(np.searchsorted(cum, sh, side="left")), len(sc) - 1)]) for sh in shares]


def census(test_units: pd.DataFrame, order_col: str | None, size: int, seed: int = 0) -> tuple[set, list]:
    """The units probed (top `size` by `order_col`, or a seeded uniform sample when order_col is None) and C+, the
    probed units found active (notebook 01's label y)."""
    if size <= 0:
        return set(), []
    if order_col is None:
        idx = np.random.default_rng(seed).choice(len(test_units), size=min(size, len(test_units)), replace=False)
        probed = test_units.iloc[np.sort(idx)]
    else:
        probed = test_units.sort_values(order_col, ascending=False, kind="stable").head(size)
    plus = {(r, c) for r, c, y in zip(probed.record_id, probed.category, probed.y) if y == 1}
    return plus, list(zip(probed.record_id, probed.category))


def candidate_sets(test_units: pd.DataFrame, score: str, lambdas: list[float], plus: set,
                   cleared=frozenset()) -> list[frozenset]:
    """K_g = C+ union {u not cleared : s_u >= lambda_g}, g = 1..G in the order of `lambdas`. A cleared unit is one
    the census probed and found inactive (revision 2); it is kept whatever its score."""
    out = []
    for lam in lambdas:
        above = {(r, c) for r, c, s in zip(test_units.record_id, test_units.category, test_units[score])
                 if s >= lam and (r, c) not in cleared}
        out.append(frozenset(plus | above))
    return out


def in_policy(r: str, c: str, K) -> bool:
    """A unit is quarantined when the policy holds the unit or its whole record."""
    return r in K or (r, c) in K


def benign_share(test_units: pd.DataFrame, K: frozenset) -> float:
    b = test_units[test_units.poison == 0]
    inK = np.array([in_policy(r, c, K) for r, c in zip(b.record_id, b.category)])
    return float(b.w_r1[inK].sum() / b.w_r1.sum()) if len(b) else 0.0


def scanner_units(test_units: pd.DataFrame, share: float, calib_units: pd.DataFrame) -> frozenset:
    """Scanner curve point: every unit whose record's scanner score reaches the threshold that quarantines `share`
    of benign exposure on calibration users."""
    lam = grid_lambdas(calib_units, "scanner", (share,))[0]
    return frozenset((r, c) for r, c, s in zip(test_units.record_id, test_units.category, test_units.scanner) if s >= lam)


def external_units(test_units: pd.DataFrame, suite: dict) -> frozenset:
    return frozenset((r, c) for r, c in zip(test_units.record_id, test_units.category)
                     if record_source(suite["records"][r]) == "external")


def external_records(suite: dict, users: set[int], laundered=frozenset()) -> frozenset:
    """Every externally sourced record of the given users, in every category (revision 2); a laundered record is not."""
    return frozenset(rid for rid, r in suite["records"].items()
                     if r["user_id"] in users and record_source(r, laundered) == "external")


# ----------------------------------------------------------------------------- audit
def n0(alpha: float, delta: float = config.DELTA) -> int:
    return int(math.ceil(math.log(delta) / math.log(1 - alpha)))


def audit_draws(n_queries: int, n: int, seeds: int, base_seed: int) -> np.ndarray:
    """The audit queries of every seed, shared by every method of a (migration, alpha) cell."""
    return np.random.default_rng(base_seed).integers(0, n_queries, size=(seeds, n))


def replay(Y: np.ndarray, draws: np.ndarray, alpha: float, delta: float = config.DELTA) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-sequence audit (Equations 6 and 7) for each row of `draws`. Y is (G, n_queries) of 0/1 outcomes, the
    candidates in their frozen order. Returns the deployed candidate index (-1 = reset) and the number of candidates
    run on the audit queries (testing stops at the first failure)."""
    from scipy.stats import binom
    n = draws.shape[1]
    X = Y[:, draws].sum(axis=2).T                      # seeds x G
    passed = binom.cdf(X, n, alpha) <= delta
    first_fail = np.where((~passed).any(axis=1), (~passed).argmax(axis=1), Y.shape[0])
    deployed = first_fail - 1
    tested = np.minimum(first_fail + 1, Y.shape[0])
    return deployed, tested


def replay_summary(Y: np.ndarray, success: np.ndarray, draws: np.ndarray, alpha: float, reset_success: float,
                   exec_tokens: np.ndarray | None = None) -> dict:
    """Certification rate, deployed policy's true r_Q and benign success, violation frequency, audit executions."""
    from scipy.stats import binom
    dep, tested = replay(Y, draws, alpha)
    r = np.where(dep >= 0, Y.mean(axis=1)[np.maximum(dep, 0)], 0.0)
    u = np.where(dep >= 0, success[np.maximum(dep, 0)], reset_success)
    v = int((r > alpha).sum())
    seeds = len(dep)
    out = {"seeds": seeds, "certified_rate": float((dep >= 0).mean()), "mean_deployed_index": float(dep[dep >= 0].mean()) if (dep >= 0).any() else float("nan"),
           "true_r_mean": float(r.mean()), "true_r_max": float(r.max()), "violations": v, "violation_rate": v / seeds,
           "c3_p_value": float(binom.sf(v - 1, seeds, config.DELTA)) if v else 1.0,
           "utility_mean": float(u.mean()), "audit_executions_mean": float(tested.mean() * draws.shape[1])}
    if exec_tokens is not None:
        cum = np.cumsum(exec_tokens)                       # tokens of one execution of candidates 1..g
        out["audit_tokens_mean"] = float((cum[tested - 1] * draws.shape[1]).mean())
    return out


# ----------------------------------------------------------------------------- frozen policies for the confirmation (notebook 04)
def confirmation_policies(T: pd.DataFrame, lam: dict, budget: dict, suite: dict, users: set[int], seed: int) -> list[dict]:
    """Every candidate of every method on one migration's confirmation units T (columns s_full, s_np, scanner, w_r1,
    y, poison). `lam` holds the frozen thresholds per score; `budget` holds light_tokens, census_unit_tokens and
    execution_tokens. Census probes read T.y; a unit the census found inactive is not treated specially."""
    G = len(lam["s_full"])
    T = T.copy()
    for col in ("s_full", "s_np", "scanner"):
        T[f"w_{col}"] = T.w_r1 * T[col]
    b_np = int(budget["light_tokens"] // budget["census_unit_tokens"])
    out = [{"method": "reuse all", "census": 0, "g": 0, "alpha": None, "K": frozenset()}]
    for gi, lamg in enumerate(lam["s_full"], 1):
        out.append({"method": "RLMA", "census": 0, "g": gi, "alpha": None, "lambda": lamg,
                    "K": frozenset((r, c) for r, c, s in zip(T.record_id, T.category, T.s_full) if s >= lamg)})
    for method, order, sc in (("RLMA without the light probe", "w_s_np", "s_np"),
                              ("scanner score + audit", "w_scanner", "scanner"),
                              ("random census", None, "s_np")):
        plus, _ = census(T, order, b_np, seed=seed)
        for gi, K in enumerate(candidate_sets(T, sc, lam[sc], plus), 1):
            out.append({"method": method, "census": b_np, "g": gi, "alpha": None, "lambda": lam[sc][gi - 1],
                        "K": K, "census_positives": len(plus)})
    for a in config.ALPHAS:
        b_gr = int((G - 1) * n0(a) * budget["execution_tokens"] // budget["census_unit_tokens"])
        plus, _ = census(T, "w_s_full", b_gr)
        out.append({"method": "greedy", "census": b_gr, "g": 1, "alpha": a, "K": frozenset(plus),
                    "census_positives": len(plus)})
    for gi, lamg in enumerate(lam["scanner_curve"], 1):
        out.append({"method": "scanner curve", "census": 0, "g": gi, "alpha": None, "lambda": lamg,
                    "K": frozenset((r, c) for r, c, s in zip(T.record_id, T.category, T.scanner) if s >= lamg)})
    out.append({"method": "blanket external-source quarantine", "census": 0, "g": 1, "alpha": None,
                "K": external_records(suite, users)})
    return out


# ----------------------------------------------------------------------------- notebook 05: census on what the candidates keep
def score_sets(T: pd.DataFrame, score: str, lambdas: list[float]) -> list[frozenset]:
    return [frozenset((r, c) for r, c, s in zip(T.record_id, T.category, T[score]) if s >= lam) for lam in lambdas]


def targeted_census(T: pd.DataFrame, score: str, lambdas: list[float], size: int) -> list[tuple[str, str]]:
    """Algorithm 1, step 1: the `size` units with the largest w*s among those the least restrictive candidate keeps
    (a unit every candidate already quarantines cannot change any candidate, so probing it is wasted)."""
    keep = score_sets(T, score, lambdas[-1:])[0]
    d = T[[(r, c) not in keep for r, c in zip(T.record_id, T.category)]].assign(_ws=lambda x: x.w_r1 * x[score])
    d = d.sort_values("_ws", ascending=False, kind="stable").head(size)
    return list(zip(d.record_id, d.category))


def top_census(T: pd.DataFrame, score: str, size: int) -> list[tuple[str, str]]:
    d = T.assign(_ws=T.w_r1 * T[score]).sort_values("_ws", ascending=False, kind="stable").head(size)
    return list(zip(d.record_id, d.category))


def random_census(T: pd.DataFrame, size: int, seed: int) -> list[tuple[str, str]]:
    idx = np.sort(np.random.default_rng(seed).choice(len(T), size=min(size, len(T)), replace=False))
    return list(zip(T.record_id.iloc[idx], T.category.iloc[idx]))


def census_selections(T: pd.DataFrame, lam: dict, size: int, seed: int,
                      ablations: dict[str, str] | None = None) -> dict[str, list[tuple[str, str]]]:
    """Units each census method probes (greedy is sized later from the audit tokens it saves). `ablations` maps an
    extra score-ablation name to its score column, which is also its key in `lam`."""
    out = {"RLMA": targeted_census(T, "s", lam["s"], size),
           "RLMA without the provenance label": targeted_census(T, "s_ns", lam["s_ns"], size),
           "scanner score + audit": targeted_census(T, "scanner", lam["scanner"], size),
           "random census": random_census(T, size, seed)}
    for name, col in (ablations or {}).items():
        out[name] = targeted_census(T, col, lam[col], size)
    return out


def greedy_size(size: int, G: int, alpha: float, execution_tokens: float, census_unit_tokens: float) -> int:
    """Greedy runs one candidate instead of G, so it gets the census size of the other methods plus the audit tokens
    it saves, spent on census units."""
    return int(size + (G - 1) * n0(alpha) * execution_tokens // census_unit_tokens)


def positives(selection: list[tuple[str, str]], labels: dict) -> frozenset:
    """Records of the probed units found active: a record proven to act for the attacker is quarantined whole."""
    return frozenset(r for r, c in selection if labels.get((r, c), 0) == 1)


def v5_policies(T: pd.DataFrame, lam: dict, selections: dict, labels: dict, suite: dict, users: set[int],
                ablations: dict[str, str] | None = None) -> list[dict]:
    """Every candidate of every method on one migration's units T (columns s, s_ns, scanner, w_r1, poison).
    `selections` holds the probed units of each census method (greedy keyed "greedy|<alpha>"); `ablations` adds
    score ablations (name -> score column and `lam` key) built like RLMA."""
    out = [{"method": "reuse all", "census": 0, "g": 0, "alpha": None, "K": frozenset()}]
    scored = [("RLMA", "s", lam["s"]), ("RLMA without the provenance label", "s_ns", lam["s_ns"]),
              ("scanner score + audit", "scanner", lam["scanner"]), ("random census", "s", lam["s"])]
    scored += [(name, col, lam[col]) for name, col in (ablations or {}).items()]
    for method, score, lm in scored:
        sel = selections[method]
        plus = positives(sel, labels)
        for gi, K in enumerate(score_sets(T, score, lm), 1):
            out.append({"method": method, "census": len(sel), "g": gi, "alpha": None, "lambda": lm[gi - 1],
                        "K": frozenset(K | plus), "census_positives": len(plus)})
    for gi, K in enumerate(score_sets(T, "s", lam["s"]), 1):
        out.append({"method": "RLMA without the census", "census": 0, "g": gi, "alpha": None, "lambda": lam["s"][gi - 1], "K": K})
    for a in config.ALPHAS:
        sel = selections[f"greedy|{a}"]
        plus = positives(sel, labels)
        out.append({"method": "greedy", "census": len(sel), "g": 1, "alpha": a, "K": plus, "census_positives": len(plus)})
    for gi, K in enumerate(score_sets(T, "scanner", lam["scanner_curve"]), 1):
        out.append({"method": "scanner curve", "census": 0, "g": gi, "alpha": None, "lambda": lam["scanner_curve"][gi - 1], "K": K})
    out.append({"method": "blanket external-source quarantine", "census": 0, "g": 1, "alpha": None,
                "K": external_records(suite, users)})
    return out


# ----------------------------------------------------------------------------- notebook 07: laundered provenance labels
def laundered_name(method: str, share: float) -> str:
    return f"{method}, {round(share * 100)}% laundered"


def laundered_frame(T: pd.DataFrame, laundered: frozenset) -> pd.DataFrame:
    """The units as the operator sees them when the attacker has laundered some records: the provenance feature of a
    laundered record reads 0 (user-sourced). The score column is recomputed by the caller with the frozen model."""
    Tl = T.copy()
    Tl["external"] = [0 if r in laundered else int(e) for r, e in zip(Tl.record_id, Tl.external)]
    return Tl


def laundered_policies(Tl: pd.DataFrame, lam_s: list[float], selection: list[tuple[str, str]], labels: dict,
                       suite: dict, users: set[int], laundered: frozenset, share: float) -> list[dict]:
    """The two methods that read the provenance label, on the laundered units Tl (column s already recomputed):
    RLMA with its targeted census (Algorithm 1, unchanged) and the blanket external-source quarantine, which keeps
    every laundered record because its label now reads 'user'."""
    plus = positives(selection, labels)
    out = []
    for gi, K in enumerate(score_sets(Tl, "s", lam_s), 1):
        out.append({"method": laundered_name("RLMA", share), "census": len(selection), "g": gi, "alpha": None,
                    "lambda": lam_s[gi - 1], "K": frozenset(K | plus), "census_positives": len(plus), "laundered": share})
    out.append({"method": laundered_name("blanket external-source quarantine", share), "census": 0, "g": 1, "alpha": None,
                "K": external_records(suite, users, laundered), "laundered": share})
    return out


# ----------------------------------------------------------------------------- notebook 08: the two-tier audit (amendment 2)
def two_tier_replay(YA: np.ndarray, YB: np.ndarray, draws: np.ndarray, alpha: float,
                    levels: tuple[float, float] = config.TWO_TIER_LEVELS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amendment 2 for each row of `draws` (one audit sample, shared by both tiers): the labelled candidates YA
    (G_A x n_queries, 0/1 outcomes, most restrictive first) are tested in order at level levels[0]; where the first
    of them fails, the label-free candidates YB are tested in order at level levels[1] on the same sample. Returns
    the deployed candidate's index within the tier used (-1 = reset), the tier used (0 labelled, 1 label-free) and
    the number of candidates run on the sample over both tiers."""
    dep_a, tested_a = replay(YA, draws, alpha, levels[0])
    dep_b, tested_b = replay(YB, draws, alpha, levels[1])
    fallback = dep_a < 0
    dep = np.where(fallback, dep_b, dep_a)
    tested = np.where(fallback, tested_a + tested_b, tested_a)
    return dep, fallback.astype(int), tested


def two_tier_summary(YA: np.ndarray, success_a: np.ndarray, YB: np.ndarray, success_b: np.ndarray, draws: np.ndarray,
                     alpha: float, reset_success: float, exec_tokens_a: np.ndarray | None = None,
                     exec_tokens_b: np.ndarray | None = None, census_tokens: float = 0.0,
                     levels: tuple[float, float] = config.TWO_TIER_LEVELS) -> dict:
    """replay_summary for the two-tier audit: certification and fallback rates, the deployed candidate's true r_Q
    and benign success, violations, executions and tokens on the sample; `census_tokens` is the cost of the
    label-free census, paid only when the fallback tier is entered."""
    from scipy.stats import binom
    dep, tier, tested = two_tier_replay(YA, YB, draws, alpha, levels)
    idx = np.maximum(dep, 0)
    r = np.where(dep >= 0, np.where(tier == 1, YB.mean(axis=1)[idx], YA.mean(axis=1)[idx]), 0.0)
    u = np.where(dep >= 0, np.where(tier == 1, success_b[idx], success_a[idx]), reset_success)
    v = int((r > alpha).sum())
    seeds, n = len(dep), draws.shape[1]
    out = {"seeds": seeds, "sample_size": n, "certified_rate": float((dep >= 0).mean()), "fallback_rate": float(tier.mean()),
           "certified_from_fallback_rate": float(((dep >= 0) & (tier == 1)).mean()),
           "mean_deployed_index": float(dep[dep >= 0].mean()) if (dep >= 0).any() else float("nan"),
           "true_r_mean": float(r.mean()), "true_r_max": float(r.max()), "violations": v, "violation_rate": v / seeds,
           "c3_p_value": float(binom.sf(v - 1, seeds, config.DELTA)) if v else 1.0,
           "utility_mean": float(u.mean()), "audit_executions_mean": float(tested.mean() * n),
           "census_tokens_mean": float(census_tokens * tier.mean())}
    if exec_tokens_a is not None and exec_tokens_b is not None:
        cum_a, cum_b = np.cumsum(exec_tokens_a), np.cumsum(exec_tokens_b)
        tested_a = np.where(tier == 1, 1, tested)                # on fallback the labelled tier ran its first candidate only
        tok = np.where(tier == 1, cum_a[0] + cum_b[np.maximum(tested - 2, 0)], cum_a[tested_a - 1]) * n
        out["audit_tokens_mean"] = float(tok.mean())
    return out
