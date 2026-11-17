"""CPU tests of notebook 07: the fourth reader migration M7, laundered provenance labels, saved score models."""
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


def _suite(n_users=4, per_user=20, poison_every=4):
    recs = {}
    for u in range(130, 130 + n_users):
        for i in range(per_user):
            rid = f"u{u}_r{i:02d}"
            poison = i % poison_every == 0
            recs[rid] = {"record_id": rid, "user_id": u, "poison": poison, "attack": "A2" if poison else None,
                         "kind": "poison" if poison else "note"}
    return {"records": recs, "users": {u: {"split": "confirmation"} for u in range(130, 130 + n_users)}}


def test_m7_is_a_reader_migration_outside_the_plan_table():
    g = config.MIGRATIONS["M7"]
    assert g["L0"] == "qwen25_7b" and g["L1"] == "mistral_7b" and g["R0"] == g["R1"] == config.REF_RETRIEVER
    assert "M7" not in config.READER_MIGRATIONS and "M7" not in config.POLICY_MIGRATIONS
    assert config.EXTRA_READER_MIGRATIONS == ("M7",) and "mistral_7b" in config.READERS
    assert "M7" not in config.A4_GROUPS
    pre = config.preregistration()["notebook_07"]
    assert pre["laundering"]["shares"] == config.LAUNDER_SHARES and pre["m7"] == g


def test_laundering_is_a_fixed_share_of_poisoned_records_by_hash():
    suite = _suite()
    pois = {r for r, x in suite["records"].items() if x["poison"]}
    assert PO.laundered_records(suite, 0.0) == frozenset()
    l25, l50 = PO.laundered_records(suite, 0.25), PO.laundered_records(suite, 0.50)
    assert l25 <= pois and l50 <= pois and l25 <= l50            # nested: a record laundered at 25% is laundered at 50%
    assert l25 == PO.laundered_records(suite, 0.25)                # deterministic
    assert 0 < len(l25) < len(l50) < len(pois)
    assert PO.laundered_records(suite, 0.5, users={130}) == {r for r in l50 if suite["records"][r]["user_id"] == 130}
    assert PO.laundered_records(suite, 0.5, salt="other") != l50
    big = _suite(n_users=40)
    share = len(PO.laundered_records(big, 0.25)) / sum(x["poison"] for x in big["records"].values())
    assert 0.18 < share < 0.32


def test_laundered_record_reads_user_everywhere_the_label_is_read():
    suite = _suite()
    L = PO.laundered_records(suite, 0.5)
    rid = sorted(L)[0]
    assert PO.record_source(suite["records"][rid]) == "external"
    assert PO.record_source(suite["records"][rid], L) == "user"
    users = set(suite["users"])
    ext, ext_l = PO.external_records(suite, users), PO.external_records(suite, users, L)
    assert ext_l == ext - L and L <= ext
    d = pd.DataFrame({"record_id": sorted(suite["records"])})
    clean, laund = SC.with_source(d, suite), SC.with_source(d, suite, L)
    assert clean.external.sum() == len(ext) and laund.external.sum() == len(ext) - len(L)
    Tl = PO.laundered_frame(clean, L)
    assert Tl.external.tolist() == laund.external.tolist() and "external" in clean  # the original frame is untouched
    assert clean.external.sum() == len(ext)


def test_laundered_policies_are_built_like_rlma_and_blanket():
    suite = _suite()
    users = set(suite["users"])
    rng = np.random.default_rng(7)
    T = pd.DataFrame({"record_id": sorted(suite["records"]), "category": "email"})
    T["user_id"] = T.record_id.map(lambda r: suite["records"][r]["user_id"])
    T["poison"] = T.record_id.map(lambda r: int(suite["records"][r]["poison"]))
    T["w_r1"] = rng.random(len(T)) + 0.05
    T = SC.with_source(T, suite)
    L = PO.laundered_records(suite, 0.5)
    Tl = PO.laundered_frame(T, L)
    Tl["s"] = 0.2 + 0.6 * Tl.external + 0.1 * rng.random(len(Tl))       # a score that leans on the label
    grid = [0.75, 0.7, 0.65, 0.6, 0.55]
    sel = PO.targeted_census(Tl, "s", grid, 5)
    labels = {(r, "email"): 1 for r in sorted(L)[:2]}                   # two laundered records found active by the census
    pols = PO.laundered_policies(Tl, grid, sel, labels, suite, users, L, 0.5)
    rl = [p for p in pols if p["method"] == "RLMA, 50% laundered"]
    bl = [p for p in pols if p["method"] == "blanket external-source quarantine, 50% laundered"]
    assert len(rl) == 5 and len(bl) == 1 and all(p["laundered"] == 0.5 for p in pols)
    plus = PO.positives(sel, labels)
    assert [p["K"] for p in rl] == [frozenset(K | plus) for K in PO.score_sets(Tl, "s", grid)]
    assert bl[0]["K"] == PO.external_records(suite, users, L) and not (bl[0]["K"] & L)
    # a laundered record scores low, so it is kept unless the census found it active
    low = [r for r in L if r not in plus]
    assert low and all(not PO.in_policy(r, "email", rl[-1]["K"]) for r in low)
    assert PO.laundered_name("RLMA", 0.25) == "RLMA, 25% laundered"


def test_reader_frame_relabels_the_base_units():
    uf = pd.DataFrame({"migration": ["M1", "M1", "M2"], "record_id": ["a", "b", "a"], "category": ["email"] * 3,
                       "w_r1": [0.3, 0.2, 0.3], "y": [1, 0, 0]})
    d = SC.reader_frame(uf, "M1", {("b", "email"): 1}, {("a", "email"): 2}, "M7")
    assert d.migration.tolist() == ["M7", "M7"] and d.record_id.tolist() == ["a", "b"]
    assert d.y.tolist() == [0, 1] and d.hits_e2e.tolist() == [2, 0] and d.target.tolist() == [1, 1]


def test_saved_ensemble_reproduces_its_scores(tmp_path):
    rng = np.random.default_rng(3)
    n = 400
    d = pd.DataFrame({"user_id": rng.integers(0, 20, n), "w_r1": rng.random(n), "scanner": rng.random(n),
                      "external": rng.integers(0, 2, n), "poison": 0})
    d["target"] = ((d.scanner + 0.5 * d.external + 0.2 * rng.random(n)) > 0.9).astype(int)
    feats = ["w_r1", "scanner", "external"]
    models, _ = SC.fit_ensemble(d, feats)
    d["s"] = SC.predict_ensemble(models, d, feats)
    SC.save_ensemble(models, tmp_path / "m.pkl.gz")
    back = SC.load_ensemble(tmp_path / "m.pkl.gz")
    assert SC.refit_gap(back, d, feats) == 0.0
    assert SC.refit_gap(SC.fit_ensemble(d, feats)[0], d, feats) < config.REFIT_TOLERANCE   # the recipe is deterministic here
