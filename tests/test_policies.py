"""CPU tests of notebook 03's module: sources, gating references, grids, census, candidates and the audit replay."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMSHIFT_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import config  # noqa: E402
import policies as PO  # noqa: E402
import suite as S  # noqa: E402

SUITE = S.build()


def test_sources():
    recs = list(SUITE["records"].values())
    assert all(PO.record_source(r) == "external" for r in recs if r["poison"])
    split = [PO.record_source(r) for r in recs if r["kind"] in config.SOURCE_SPLIT_KINDS]
    assert 0.4 < np.mean([s == "external" for s in split]) < 0.6
    assert all(PO.record_source(r) == "user" for r in recs if r["kind"] == "note")


def test_n0_matches_the_plan():
    assert [PO.n0(a) for a in (0.01, 0.02, 0.05)] == [299, 149, 59]


def test_replay_fixed_sequence():
    Y = np.zeros((3, 100), int)
    Y[2, :50] = 1                                       # the last candidate is unsafe
    draws = PO.audit_draws(100, 299, 200, 0)
    dep, tested = PO.replay(Y, draws, 0.01)
    assert (dep == 1).all() and (tested == 3).all()
    Y[0, 0] = 1                                         # the first candidate fails whenever query 0 is drawn
    dep, _ = PO.replay(Y, draws, 0.01)
    hit = (draws == 0).any(axis=1)
    assert (dep[hit] == -1).all() and (dep[~hit] == 1).all()


def test_replay_violation_frequency_stays_below_delta():
    rng = np.random.default_rng(3)
    Y = (rng.random((1, 2000)) < 0.012).astype(int)     # just above alpha = 0.01
    s = PO.replay_summary(Y, np.array([0.6]), PO.audit_draws(2000, 299, 1000, 7), 0.01, 0.4)
    assert s["violation_rate"] <= 0.06 and s["c3_p_value"] > 0.01


def _units():
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame({"record_id": [f"r{i}" for i in range(n)], "category": "email",
                      "split": ["calibration"] * 100 + ["test"] * 200, "w_r1": rng.random(n) + 0.1,
                      "s_full": rng.random(n), "scanner": rng.random(n)})
    d["poison"] = (d.s_full > 0.9).astype(int)
    d["y"] = ((d.poison == 1) & (rng.random(n) < 0.7)).astype(int)
    return d


def test_grid_census_candidates_and_shares():
    d = _units()
    lam = PO.grid_lambdas(d, "s_full")
    assert lam == sorted(lam)                           # most quarantine first
    test = d[d.split == "test"]
    plus, probed = PO.census(test, "s_full", 20)
    assert len(probed) == 20 and plus <= set(probed)
    K = PO.candidate_sets(test, "s_full", lam, plus)
    assert all(K[i] >= K[i + 1] for i in range(len(K) - 1)) and all(plus <= k for k in K)
    shares = [PO.benign_share(test, k) for k in K]
    assert shares == sorted(shares, reverse=True) and 0 <= shares[-1] <= shares[0] <= 1
    rp, rprobed = PO.census(test, None, 20, seed=1)
    assert len(rprobed) == 20 and PO.census(test, None, 20, seed=1)[1] == rprobed


def test_gating_references_block_attacks_and_cost_memory_destinations():
    uid = 0
    qs = [q for q in SUITE["queries"].values() if q["user_id"] == uid]
    of = pd.DataFrame({"pipeline": "p", "policy": "none", "split": "log", "query_id": [q["query_id"] for q in qs],
                       "user_id": uid, "memory_dependent": [q["memory_dependent"] for q in qs], "task_success": 1})
    ref = PO.gating_references(of, SUITE).set_index("reference")
    assert (ref.r_Q == 0).all() and ref.loc["destination gating", "success"] <= ref.loc["authority gating", "success"] <= 1
    assert ref.loc["destination gating", "tasks_refused"] > 0


def test_survivor_sets_read_from_csv(tmp_path):
    of = pd.DataFrame({"pipeline": "a|b", "policy": ["none", "none", "round2|a|b"], "split": "log",
                       "hit_records": ["r1|r2", "", "r3"]})
    of.to_csv(tmp_path / "o.csv.gz", index=False)
    back = pd.read_csv(tmp_path / "o.csv.gz")
    assert PO.survivor_sets(back) == {"a|b": frozenset({"r1", "r2", "r3"})}
