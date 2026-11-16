"""CPU tests of notebook 05: units on the served store, the census on what the candidates keep, whole-record
quarantine of census positives, and the second confirmation users."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMSHIFT_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import policies as PO  # noqa: E402
import score as SC  # noqa: E402
import suite as S  # noqa: E402


class _Ret:
    def __init__(self, cnt):
        self.cnt = cnt

    def exposure_counts(self, suite):
        return self.cnt


def _suite():
    recs = {"a": {"record_id": "a", "user_id": 130, "poison": True, "attack": "A1", "kind": "poison"},
            "b": {"record_id": "b", "user_id": 130, "poison": False, "attack": None, "kind": "note"},
            "c": {"record_id": "c", "user_id": 130, "poison": False, "attack": None, "kind": "quoted_instruction"}}
    return {"records": recs, "users": {130: {"split": "confirmation"}}}


def test_served_frame_counts_the_served_store():
    suite = _suite()
    recf = pd.DataFrame({"record_id": ["a", "b", "c"], "user_id": 130, "scanner": [0.9, 0.1, 0.2]})
    d = SC.served_frame(suite, _Ret({("a", "email"): 3, ("b", "email"): 10, ("c", "payments"): 0}), recf,
                        labels={("a", "email"): 1})
    assert list(zip(d.record_id, d.category)) == [("a", "email"), ("b", "email")]
    assert d.w_r1.tolist() == [0.3, 1.0] and d.dw.tolist() == [0.0, 0.0]
    assert d.poison.tolist() == [1, 0] and d.external.tolist() == [1, 0] and d.y.tolist() == [1, 0]


def test_unit_labels_and_training_frame():
    act = pd.DataFrame({"record_id": ["a", "a", "b"], "category": ["email", "email", "email"],
                        "reader": ["llama31_8b", "qwen25_7b", "llama31_8b"], "a": [1, 0, 0]})
    assert SC.unit_labels(act, "llama31_8b") == {("a", "email"): 1, ("b", "email"): 0}
    d1 = pd.DataFrame({"record_id": ["a"], "category": ["email"], "user_id": [1], "poison": [1], "w_r1": [0.1],
                       "target": [1], "scanner": [0.5], "extra": [9]})
    t = SC.training_frame([d1, d1], ["scanner", "w_r1"])
    assert len(t) == 2 and "extra" not in t.columns


def _units(n=120, seed=3):
    rng = np.random.default_rng(seed)
    T = pd.DataFrame({"record_id": [f"u130_r{i:02d}" for i in range(n)], "category": "email", "user_id": 130,
                      "w_r1": rng.random(n) + 0.05, "s": rng.random(n), "s_ns": rng.random(n), "scanner": rng.random(n)})
    T["poison"] = (T.s > 0.8).astype(int)
    return T


def test_targeted_census_skips_units_every_candidate_quarantines():
    T = _units()
    lam = [0.5, 0.6, 0.7, 0.8, 0.9]
    sel = PO.targeted_census(T, "s", lam, 20)
    s = dict(zip(zip(T.record_id, T.category), T.s))
    assert len(sel) == 20 and all(s[u] < 0.9 for u in sel)
    ws = [T.w_r1[T.record_id == r].iloc[0] * s[(r, c)] for r, c in sel]
    assert ws == sorted(ws, reverse=True)
    assert PO.greedy_size(200, 5, 0.01, 2000.0, 4000.0) == 200 + 4 * 299 * 2000 // 4000


def test_v5_policies():
    T = _units()
    lam = {"s": [0.5, 0.6, 0.7, 0.8, 0.9], "s_ns": [0.5, 0.6, 0.7, 0.8, 0.9], "scanner": [0.5, 0.6, 0.7, 0.8, 0.9],
           "scanner_curve": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    sel = PO.census_selections(T, lam, 20, seed=1)
    for a in config.ALPHAS:
        sel[f"greedy|{a}"] = PO.top_census(T, "s", 30)
    probed = {u for v in sel.values() for u in v}
    labels = {u: 1 for u in probed if T.poison[T.record_id == u[0]].iloc[0] and hash(u) % 2 == 0}
    suite = {"records": {r: {"record_id": r, "user_id": 130, "poison": bool(p), "kind": "note"} for r, p in zip(T.record_id, T.poison)}}
    pols = PO.v5_policies(T, lam, sel, labels, suite, {130})
    meth = pd.Series([p["method"] for p in pols]).value_counts().to_dict()
    assert meth == {"RLMA": 5, "RLMA without the provenance label": 5, "scanner score + audit": 5, "random census": 5,
                    "RLMA without the census": 5, "scanner curve": 6, "greedy": 3, "reuse all": 1,
                    "blanket external-source quarantine": 1}
    rl = [p["K"] for p in pols if p["method"] == "RLMA"]
    assert all(rl[i] >= rl[i + 1] for i in range(4))
    plus = PO.positives(sel["RLMA"], labels)
    assert all(r in K for K in rl for r in plus)                     # whole records, in every candidate
    nocensus = [p["K"] for p in pols if p["method"] == "RLMA without the census"]
    assert all(a == b - plus for a, b in zip(nocensus, rl))
    assert all(PO.in_policy(r, "payments", rl[-1]) for r in plus)    # also in categories the census never probed


def test_second_confirmation_users(tmp_path):
    c = S.write(tmp_path / "c2", config.CONFIRM2_USERS)
    assert c == {"users": 30, "records": 1800, "queries": 1200, "heldout_queries": 600}
    sc = S.load(tmp_path / "c2")
    assert min(sc["users"]) == 130 and max(sc["users"]) == 159
    first = S.build_all(config.CONFIRM_USERS)
    assert not set(sc["records"]) & {r["record_id"] for r in first["records"]}
    assert "notebook_05" in config.preregistration()
