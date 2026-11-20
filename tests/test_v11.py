"""CPU tests of notebooks 11 and 12: amendment 5 (pass rule, evidence-informed fallback, joint pool, smoke floor)."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMSHIFT_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import pipeline as PL  # noqa: E402
import policies as PO  # noqa: E402


def test_n_k_is_the_smallest_valid_sample_and_extends_n0():
    from scipy.stats import binom
    for a in config.ALPHAS:
        for level in (config.DELTA, config.DELTA / 2, config.DELTA / 4):
            assert PO.n_k(a, level, 0) == PO.n0(a, level)
            for k in (1, 2):
                n = PO.n_k(a, level, k)
                assert binom.cdf(k, n, a) <= level and binom.cdf(k, n - 1, a) > level
                assert n > PO.n_k(a, level, k - 1)
    assert PO.n_k(0.01, config.DELTA / 2, 1) == 555 and PO.n_k(0.01, config.DELTA / 4, 0) == 437


def test_replay_with_k_passes_at_most_k_actions():
    rng = np.random.default_rng(0)
    Y = np.zeros((3, 500), int); Y[1, :3] = 1; Y[2, :40] = 1          # K1 clean, K2 three attacked tasks, K3 forty
    draws = np.stack([np.arange(500)] * 2)                            # the whole log as the sample
    dep0, _ = PO.replay(Y, draws, 0.01, config.DELTA / 2, k=0)
    dep1, _ = PO.replay(Y, draws, 0.01, config.DELTA / 2, k=1)
    dep3, _ = PO.replay(Y, draws, 0.01, config.DELTA / 2, k=3)
    assert list(dep0) == [0, 0] and list(dep1) == [0, 0] and list(dep3) == [1, 1]
    s = PO.replay_summary(Y, np.array([0.5, 0.6, 0.7]), draws, 0.01, 0.4, k=3)
    assert s["certified_rate"] == 1.0 and abs(s["utility_mean"] - 0.6) < 1e-12
    _ = rng


def test_repaired_ladder_outcomes_follow_the_sampled_records():
    H = ["ra", "rb"]
    base_y = np.array([[0, 1, 1, 0], [0, 1, 1, 1]]); base_s = np.array([[1, 0, 0, 1], [1, 0, 0, 0]], float)
    # task 1 holds ra, task 2 holds ra and rb; quarantining them repairs the hit
    var = [[(1, 0b01, {0b01: (0, 1)}), (2, 0b11, {0b01: (1, 0), 0b10: (0, 1), 0b11: (0, 1)})],
           [(1, 0b01, {0b01: (0, 1)}), (2, 0b11, {0b01: (1, 0), 0b10: (0, 1), 0b11: (0, 1)})]]
    L = PO.RepairedLadder(base_y, base_s, var)
    Y, S = L.outcomes(0)
    assert (Y == base_y).all()
    Y, S = L.outcomes(0b01)
    assert list(Y[0]) == [0, 0, 1, 0] and S[0, 1] == 1
    Y, S = L.outcomes(0b11)
    assert list(Y[0]) == [0, 0, 0, 0] and list(Y[1]) == [0, 0, 0, 1]
    assert L.subsets_needed() == 8
    assert list(PO.hit_mask([["ra"], [], ["rb", "ra"], ["rz"]], H)) == [1, 0, 3, 0]


def test_three_ladder_replay_repairs_what_the_first_audit_saw_and_resets_otherwise():
    n = 400
    Y1 = np.zeros((2, n), int); Y1[:, :8] = 1                        # the labelled ladder leaves eight attacked tasks, all from record ra
    S1 = np.full((2, n), 0.6)
    hits1 = PO.hit_mask([["ra"] if i < 8 else [] for i in range(n)], ["ra"])
    var = [[(i, 1, {1: (0, 1)}) for i in range(8)] for _ in range(2)]
    L2 = PO.RepairedLadder(Y1, S1, var)                               # with ra quarantined every task is clean
    L3 = PO.RepairedLadder(np.ones((2, n), int), np.full((2, n), 0.3), [[], []])   # the label-free ladder is useless here
    rng = np.random.default_rng(1)
    draws = tuple(rng.integers(0, n, size=(200, m)) for m in (PO.n_k(0.01, config.DELTA / 2), PO.n_k(0.01, config.DELTA / 4), PO.n_k(0.01, config.DELTA / 4)))
    rep = PO.three_ladder_replay(Y1, S1, hits1, L2, L3, draws, 0.01)
    summ = PO.three_ladder_summary(rep, draws, 0.01, 0.2)
    # eight of 400 tasks attacked: ladder 1 fails in nearly every draw; whenever the sample held a hit, R = {ra} and ladder 2 certifies
    assert summ["ladder1_rate"] < 0.05 and summ["ladder2_rate"] > 0.95 and summ["ladder3_rate"] == 0.0
    assert summ["violations"] == 0 and abs(summ["utility_mean"] - 0.6) < 0.02
    # a ladder 1 failure whose sample saw no hit cannot happen here (a failure is a hit), so R is never empty on entry
    assert summ["mean_R_size"] == 1.0
    # without any repairable record the procedure ends in the label-free ladder or the reset
    L2b = PO.RepairedLadder(Y1, S1, [[], []])
    rep_b = PO.three_ladder_replay(Y1, S1, hits1, L2b, L3, draws, 0.01)
    assert (rep_b["ladder"] == 0).mean() > 0.9


def test_three_ladder_certificate_holds_under_the_union_bound():
    # a store whose candidates all carry risk above alpha: the deployed policy must exceed alpha in at most delta of the seeds (here far fewer)
    rng = np.random.default_rng(7)
    n, G, alpha = 2000, 3, 0.01
    Y1 = (rng.random((G, n)) < 0.02).astype(int); S1 = np.full((G, n), 0.5)
    hits1 = PO.hit_mask([["r0"] if Y1[0, i] else [] for i in range(n)], ["r0"])
    # the repair removes nothing (the record r0 is a stand-in for many): repaired outcomes equal the originals
    var = [[(i, 1, {1: (int(Y1[g, i]), 0.5)}) for i in range(n) if Y1[0, i]] for g in range(G)]
    L2 = PO.RepairedLadder(Y1, S1, var); L3 = PO.RepairedLadder(Y1, S1, var)
    draws = tuple(rng.integers(0, n, size=(500, m)) for m in (PO.n_k(alpha, config.DELTA / 2), PO.n_k(alpha, config.DELTA / 4), PO.n_k(alpha, config.DELTA / 4)))
    summ = PO.three_ladder_summary(PO.three_ladder_replay(Y1, S1, hits1, L2, L3, draws, alpha), draws, alpha, 0.2)
    assert summ["violation_rate"] <= config.DELTA


def test_smoke_floor_is_per_reader_where_set():
    assert PL.smoke_floor("mistral_7b") == config.MIN_SMOKE_SUCCESS_BY_READER["mistral_7b"] == 0.40
    assert PL.smoke_floor("qwen25_7b") == config.MIN_SMOKE_SUCCESS["reader"]
    assert PL.smoke_floor("llama32_3b") == config.MIN_SMOKE_SUCCESS["surrogate"]


def test_preregistration_names_amendment_5():
    p = config.preregistration()
    assert p["notebook_11"]["amendment"] == "docs/plan_v2_amendment_5.md" and p["notebook_12"]["method"]["levels"] == config.THREE_LADDER_LEVELS
    assert abs(sum(config.THREE_LADDER_LEVELS) - config.DELTA) < 1e-12


def test_ladder_order_follows_the_labels_of_the_records_behind_the_hits():
    n = 400
    Y1 = np.zeros((2, n), int); Y1[:, :8] = 1; S1 = np.full((2, n), 0.6)
    H = ["ra"]; hits1 = PO.hit_mask([["ra"] if i < 8 else [] for i in range(n)], H)
    var = [[(i, 1, {1: (0, 1)}) for i in range(8)] for _ in range(2)]
    L2 = PO.RepairedLadder(Y1, S1, var)                               # repaired labelled: clean once ra is quarantined
    L3 = PO.RepairedLadder(Y1, np.full((2, n), 0.5), var)             # repaired label-free: also clean, lower success
    rng = np.random.default_rng(3)
    draws = tuple(rng.integers(0, n, size=(200, m)) for m in (PO.n_k(0.01, config.DELTA / 2), PO.n_k(0.01, config.DELTA / 4), PO.n_k(0.01, config.DELTA / 4)))
    # ra honestly labelled external (mask 0): labelled ladder first, certified there
    rep = PO.three_ladder_replay(Y1, S1, hits1, L2, L3, draws, 0.01, user_label_mask=0)
    s = PO.three_ladder_summary(rep, draws, 0.01, 0.2)
    assert s["ladder2_rate"] > 0.95 and s["ladder3_rate"] == 0.0 and s["label_free_first_rate"] == 0.0 and abs(s["utility_mean"] - 0.6) < 0.02
    # ra carries the user label (mask bit set): label-free ladder first, certified there, the labelled ladder never tested
    rep = PO.three_ladder_replay(Y1, S1, hits1, L2, L3, draws, 0.01, user_label_mask=1)
    s = PO.three_ladder_summary(rep, draws, 0.01, 0.2)
    assert s["ladder3_rate"] > 0.95 and s["ladder2_rate"] == 0.0 and s["label_free_first_rate"] > 0.95 and abs(s["utility_mean"] - 0.5) < 0.02
    assert (rep["tested2"][rep["ladder"] == 3] == 0).all()
    assert abs(s["mean_R_size"] - 1.0) < 1e-9 and s["entered_ladder3_rate"] > 0.95   # R reported whichever ladder went first
    # episodes: K_1 on the first sample, then both repaired label-free candidates on one fallback sample; the other ladder costs nothing
    assert abs(s["audit_executions_mean"] - (draws[0].shape[1] + 2 * draws[1].shape[1])) < 1e-9
