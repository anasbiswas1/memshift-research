"""CPU tests of notebook 08: the two-tier audit of amendment 2 and the users 160 to 189 service."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMSHIFT_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import config  # noqa: E402
import policies as PO  # noqa: E402
import suite as S  # noqa: E402


def _outcomes(rates, n=1200, seed=0):
    """0/1 outcome vectors with exactly round(r * n) attacker actions each."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(rates), n), int)
    for i, r in enumerate(rates):
        out[i, rng.choice(n, int(round(r * n)), replace=False)] = 1
    return out


def test_sample_sizes_of_the_two_tiers():
    assert [PO.n0(a) for a in config.ALPHAS] == [299, 149, 59]
    assert [PO.n0(a, config.DELTA / 2) for a in config.ALPHAS] == [368, 183, 72]
    assert sum(config.TWO_TIER_LEVELS) == config.DELTA
    pre = config.preregistration()["notebook_08"]
    assert pre["users"] == [160, 189] and pre["migrations"] == ["M1", "M2", "M7"] and "C5" in pre["claims"]


def test_clean_labels_never_enter_the_fallback_tier():
    YA = _outcomes([0, 0, 0, 0, 0.001]); YB = _outcomes([0, 0, 0.01, 0.02, 0.05])
    draws = PO.audit_draws(1200, PO.n0(0.01, config.DELTA / 2), 200, 7)
    dep, tier, tested = PO.two_tier_replay(YA, YB, draws, 0.01)
    assert tier.sum() == 0 and (dep >= 3).all() and (tested <= 5).all()
    s = PO.two_tier_summary(YA, np.full(5, 0.6), YB, np.full(5, 0.55), draws, 0.01, 0.3)
    assert s["fallback_rate"] == 0 and s["certified_rate"] == 1 and s["sample_size"] == 368 and abs(s["utility_mean"] - 0.6) < 1e-12


def test_a_failing_first_labelled_candidate_enters_the_label_free_tier_on_the_same_sample():
    YA = _outcomes([0.2, 0.2, 0.2, 0.2, 0.2]); YB = _outcomes([0, 0, 0, 0.001, 0.001])
    draws = PO.audit_draws(1200, PO.n0(0.02, config.DELTA / 2), 200, 3)
    dep, tier, tested = PO.two_tier_replay(YA, YB, draws, 0.02)
    assert (tier == 1).all() and (dep >= 2).all() and (tested >= 4).all() and (tested <= 6).all()
    s = PO.two_tier_summary(YA, np.full(5, 0.6), YB, np.full(5, 0.55), draws, 0.02, 0.3,
                            exec_tokens_a=np.full(5, 10.0), exec_tokens_b=np.full(5, 10.0), census_tokens=1000.0)
    assert s["fallback_rate"] == 1 and s["certified_from_fallback_rate"] == 1 and s["census_tokens_mean"] == 1000.0
    assert abs(s["audit_tokens_mean"] - np.mean(tested) * 10.0 * 183) < 1e-9 and abs(s["utility_mean"] - 0.55) < 1e-12


def test_both_tiers_failing_is_a_reset():
    YA = _outcomes([0.3] * 5); YB = _outcomes([0.3] * 5)
    draws = PO.audit_draws(1200, PO.n0(0.05, config.DELTA / 2), 100, 5)
    dep, tier, tested = PO.two_tier_replay(YA, YB, draws, 0.05)
    assert (dep == -1).all() and (tier == 1).all() and (tested == 2).all()
    s = PO.two_tier_summary(YA, np.full(5, 0.6), YB, np.full(5, 0.55), draws, 0.05, 0.3)
    assert s["certified_rate"] == 0 and s["utility_mean"] == 0.3 and s["violations"] == 0


def test_the_certificate_holds_when_every_candidate_of_both_tiers_is_unsafe():
    # every candidate acts above alpha: any deployment is a violation; the two-tier audit deploys in at most delta of the replays
    for alpha, seed in ((0.01, 1), (0.02, 2), (0.05, 3)):
        YA = _outcomes([alpha * 1.5] * 5, seed=seed); YB = _outcomes([alpha * 1.5] * 5, seed=seed + 10)
        rates = np.concatenate([YA.mean(1), YB.mean(1)])
        assert (rates > alpha).all()
        draws = PO.audit_draws(1200, PO.n0(alpha, config.DELTA / 2), 4000, 100 + seed)
        s = PO.two_tier_summary(YA, np.full(5, 0.6), YB, np.full(5, 0.6), draws, alpha, 0.3)
        assert s["violation_rate"] == s["certified_rate"] <= config.DELTA + 0.01


def test_the_two_tier_audit_is_rlma_at_delta_half_when_the_fallback_is_not_entered():
    YA = _outcomes([0, 0.002, 0.005, 0.008, 0.012]); YB = _outcomes([0.5] * 5)
    draws = PO.audit_draws(1200, PO.n0(0.01, config.DELTA / 2), 300, 9)
    dep, tier, tested = PO.two_tier_replay(YA, YB, draws, 0.01)
    dep1, tested1 = PO.replay(YA, draws, 0.01, config.DELTA / 2)
    assert (tier == 0).all() and (dep == dep1).all() and (tested == tested1).all()


def test_third_confirmation_users(tmp_path):
    assert not set(config.CONFIRM3_USERS) & (set(config.CONFIRM2_USERS) | set(config.CONFIRM_USERS) | set(range(config.N_USERS)))
    c = S.write(tmp_path / "c3", config.CONFIRM3_USERS)
    assert c == {"users": 30, "records": 1800, "queries": 1200, "heldout_queries": 600}
    sc = S.load(tmp_path / "c3")
    assert min(sc["users"]) == 160 and max(sc["users"]) == 189
    second = S.build_all(config.CONFIRM2_USERS)
    assert not set(sc["records"]) & {r["record_id"] for r in second["records"]}
