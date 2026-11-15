"""CPU tests of notebook 04: confirmation users, the frozen score ensemble and the confirmation policies."""
import json
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


def test_confirmation_users_are_new_and_leave_development_unchanged(tmp_path):
    c = S.write(tmp_path / "c", config.CONFIRM_USERS)
    assert c == {"users": 30, "records": 1800, "queries": 1200, "heldout_queries": 600}
    sc = S.load(tmp_path / "c")
    assert {u["split"] for u in sc["users"].values()} == {"confirmation"}
    assert min(sc["users"]) == 100 and max(sc["users"]) == 129
    dev = S.build_all()
    assert not set(sc["records"]) & {r["record_id"] for r in dev["records"]}
    committed = ROOT / "suite_data" / "users.json"
    if committed.exists():
        assert json.loads(committed.read_text()) == dev["users"]


def _dev(n_users=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for i in range(30):
            pois = int(rng.random() < 0.2)
            x = rng.normal(size=3) + (1.5 if pois else 0)
            rows.append({"user_id": u, "record_id": f"u{u}_r{i}", "category": "email", "poison": pois,
                         "w_r1": rng.random() + 0.1, "f0": x[0], "f1": x[1], "f2": x[2],
                         "target": int(pois and rng.random() < 0.7), "hits_e2e": int(pois) * int(rng.integers(0, 3))})
    return pd.DataFrame(rows)


def test_ensemble_scores_rank_poison_and_thresholds_are_ordered():
    d = _dev()
    models, oof = SC.fit_ensemble(d, ["f0", "f1", "f2"])
    assert len(models) == config.SCORE_FOLDS and oof.shape == (len(d),)
    s = SC.predict_ensemble(models, d, ["f0", "f1", "f2"])
    assert s.shape == (len(d),) and s[d.poison == 1].mean() > s[d.poison == 0].mean()
    lam = SC.thresholds(d, oof, config.CONFIRM_GRID_SHARES)
    assert lam == sorted(lam) and len(lam) == len(config.CONFIRM_GRID_SHARES)
    assert 0 <= SC.risk_caught(d, oof, 0.05) <= SC.risk_caught(d, oof, 0.25) <= 1


def test_dev_frame_target_is_probe_or_end_to_end():
    uf = pd.DataFrame({"migration": "M1", "record_id": ["a", "b", "c"], "category": "email", "y": [1, 0, 0]})
    of = pd.DataFrame({"pipeline": "bge_large_v15|llama31_8b", "policy": "none", "query_id": ["q1", "q2"],
                       "hit_records": ["b", np.nan]})
    d = SC.dev_frame(uf, of, {"q1": {"category": "email"}, "q2": {"category": "email"}}, "M1")
    assert d.target.tolist() == [1, 1, 0] and d.hits_e2e.tolist() == [0, 1, 0]


def test_confirmation_policies():
    rng = np.random.default_rng(2)
    n = 200
    T = pd.DataFrame({"record_id": [f"u100_r{i:02d}" for i in range(n)], "category": "email", "w_r1": rng.random(n) + 0.1,
                      "s_full": rng.random(n), "s_np": rng.random(n), "scanner": rng.random(n), "user_id": 100})
    T["poison"] = (T.s_full > 0.85).astype(int)
    T["y"] = ((T.poison == 1) & (rng.random(n) < 0.6)).astype(int)
    lam = {"s_full": [0.5, 0.6, 0.7, 0.8, 0.9], "s_np": [0.5, 0.6, 0.7, 0.8, 0.9], "scanner": [0.5, 0.6, 0.7, 0.8, 0.9],
           "scanner_curve": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    budget = {"light_tokens": 2e5, "census_unit_tokens": 4000.0, "execution_tokens": 2000.0}
    suite = {"records": {r: {"record_id": r, "user_id": 100, "poison": bool(p), "kind": "note"} for r, p in zip(T.record_id, T.poison)}}
    pols = PO.confirmation_policies(T, lam, budget, suite, {100}, seed=1)
    meth = pd.Series([p["method"] for p in pols]).value_counts().to_dict()
    assert meth == {"RLMA": 5, "RLMA without the light probe": 5, "scanner score + audit": 5, "random census": 5,
                    "scanner curve": 6, "greedy": 3, "reuse all": 1, "blanket external-source quarantine": 1}
    rl = [p["K"] for p in pols if p["method"] == "RLMA"]
    assert all(rl[i] >= rl[i + 1] for i in range(4))
    blanket = next(p["K"] for p in pols if p["method"].startswith("blanket"))
    assert blanket == frozenset(T.record_id[T.poison == 1])


def test_provenance_feature_matches_the_blanket_label():
    recs = {"a": {"record_id": "a", "poison": True, "kind": "note"}, "b": {"record_id": "b", "poison": False, "kind": "note"},
            "c": {"record_id": "c", "poison": False, "kind": "quoted_instruction"}}
    d = SC.with_source(pd.DataFrame({"record_id": ["a", "b", "c"]}), {"records": recs})
    assert d.external.tolist() == [1, 0, 1]
    assert "external" in config.SCORE_FEATURES_SOURCE and config.CONFIRM_GRID_SHARES == tuple(sorted(config.CONFIRM_GRID_SHARES, reverse=True))


def test_residual_rate_falls_as_quarantine_grows():
    d = _dev()
    _, oof = SC.fit_ensemble(d, ["f0", "f1", "f2"])
    r = [SC.residual_rate(d, oof, sh, "hits_e2e", 100) for sh in (0.02, 0.10, 0.50)]
    assert r[0] >= r[1] >= r[2] >= 0
