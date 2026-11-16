"""CPU tests of notebook 06: exposure change on the served store and the extra score ablation."""
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


class _Ret:
    def __init__(self, cnt):
        self.cnt = cnt

    def exposure_counts(self, suite):
        return self.cnt


def test_exposure_change_on_the_served_store():
    suite = {"records": {"a": {"record_id": "a", "user_id": 130, "poison": True, "attack": "A2", "kind": "poison"},
                         "b": {"record_id": "b", "user_id": 130, "poison": False, "attack": None, "kind": "note"}},
             "users": {130: {"split": "confirmation"}}}
    recf = pd.DataFrame({"record_id": ["a", "b"], "user_id": 130, "scanner": [0.2, 0.1]})
    d = SC.served_frame(suite, _Ret({("a", "email"): 4, ("b", "email"): 5}), recf,
                        ret_old=_Ret({("b", "email"): 7}))
    assert d.dw.tolist() == [0.4, 0.2] and d.w_r1.tolist() == [0.4, 0.5]
    assert "dw" in config.METHOD_FEATURES and "dw" not in config.NO_EXPOSURE_CHANGE_FEATURES
    assert set(config.METHOD_FEATURES) - set(config.NO_EXPOSURE_CHANGE_FEATURES) == {"dw"}
    assert config.RETRIEVER_MIGRATIONS == ("M4", "M6") and "notebook_06" in config.preregistration()
    assert all(config.MIGRATIONS[m]["R0"] != config.MIGRATIONS[m]["R1"] for m in config.RETRIEVER_MIGRATIONS)


def test_extra_ablation_is_built_like_rlma():
    rng = np.random.default_rng(5)
    n = 100
    T = pd.DataFrame({"record_id": [f"u130_r{i:02d}" for i in range(n)], "category": "payments", "user_id": 130,
                      "w_r1": rng.random(n) + 0.05, "s": rng.random(n), "s_ns": rng.random(n), "s_nd": rng.random(n),
                      "scanner": rng.random(n)})
    T["poison"] = (T.s > 0.85).astype(int)
    grid = [0.5, 0.6, 0.7, 0.8, 0.9]
    lam = {"s": grid, "s_ns": grid, "s_nd": grid, "scanner": grid, "scanner_curve": grid + [0.95]}
    abl = {"RLMA without the exposure change": "s_nd"}
    sel = PO.census_selections(T, lam, 15, seed=2, ablations=abl)
    assert set(sel) >= {"RLMA", "RLMA without the exposure change"}
    for a in config.ALPHAS:
        sel[f"greedy|{a}"] = PO.top_census(T, "s", 20)
    pols = PO.v5_policies(T, lam, sel, {}, {"records": {r: {"record_id": r, "user_id": 130, "poison": False, "kind": "note"}
                                                       for r in T.record_id}}, {130}, ablations=abl)
    nd = [p["K"] for p in pols if p["method"] == "RLMA without the exposure change"]
    assert len(nd) == 5 and nd == PO.score_sets(T, "s_nd", grid)
    assert len([p for p in pols if p["method"] == "RLMA"]) == 5
