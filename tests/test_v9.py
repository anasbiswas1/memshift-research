"""CPU tests of notebook 09: the zero-exposure extension and the transferred score of amendment 3."""
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


def test_extension_adds_only_zero_exposure_units_of_quarantined_records():
    exposed = {("r1", "email"), ("r1", "payments"), ("r2", "calendar"), ("r3", "email")}
    K = frozenset({("r1", "email"), "r9"})
    ext = PO.extend_zero_exposure(K, exposed)
    assert ("r1", "calendar") in ext and ("r1", "documents") in ext        # zero exposure, record quarantined elsewhere
    assert ("r1", "payments") not in ext                                     # exposed and kept: untouched
    assert not any(isinstance(u, tuple) and u[0] in ("r2", "r3") for u in ext)   # records K does not quarantine
    assert "r9" in ext and not any(isinstance(u, tuple) and u[0] == "r9" for u in ext)   # whole records already cover every category
    for r, c in exposed:
        assert PO.in_policy(r, c, ext) == PO.in_policy(r, c, K)             # nothing changes on the exposed units


def test_extension_is_idempotent_and_keeps_log_contexts():
    exposed = {("r1", "email")}
    K = frozenset({("r1", "email")})
    ext = PO.extend_zero_exposure(K, exposed)
    assert PO.extend_zero_exposure(ext, exposed) == ext
    assert len(ext) == 4


def test_tier_policies_join_census_positives():
    T = pd.DataFrame({"record_id": ["a", "a", "b", "c"], "category": ["email", "payments", "email", "email"],
                      "s": [0.9, 0.2, 0.5, 0.1], "w_r1": [0.5, 0.1, 0.3, 0.2], "poison": [1, 1, 0, 1]})
    pols = PO.tier_policies(T, "RLMA", "s", [0.15, 0.4, 0.8], [("c", "email"), ("b", "email")], {("c", "email"): 1})
    assert [p["g"] for p in pols] == [1, 2, 3] and all(p["census"] == 2 and p["census_positives"] == 1 for p in pols)
    assert "c" in pols[2]["K"] and ("a", "email") in pols[2]["K"] and ("b", "email") not in pols[2]["K"]
    assert ("a", "payments") in pols[0]["K"] and ("a", "payments") not in pols[1]["K"]


def test_transfer_training_excludes_the_target_reader():
    train = {m: pd.DataFrame({"migration": [m], "x": [1]}) for m in ("M1", "M2", "M3", "M7")}
    for target, srcs in config.TRANSFER_SOURCES.items():
        d = SC.transfer_training(train, target)
        assert set(d.migration) == set(srcs) and target not in set(d.migration)
    assert config.preregistration()["notebook_09"]["claims"]["C8"].startswith("extension")


def test_risk_breakdown_counts():
    o = pd.DataFrame({"migration": ["M1"] * 4, "name": ["p"] * 4, "split": ["log"] * 4,
                      "category": ["email", "email", "payments", "payments"], "y": [1, 0, 0, 0]})
    b = PO.risk_breakdown(o, "category").set_index("category")
    assert b.loc["email", "attacked"] == 1 and b.loc["email", "tasks"] == 2 and b.loc["payments", "rate"] == 0
