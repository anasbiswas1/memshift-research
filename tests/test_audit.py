"""Tests of Algorithm 1 (src/audit.py): Equation 8 against brute force, census feasibility, and the violation rate
under scores that are random, inverted or adversarial (Proposition 1)."""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMSHIFT_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import binom  # noqa: E402

import audit as AU  # noqa: E402


@pytest.mark.parametrize("a,d,h", [(0.01, 0.05, 0), (0.02, 0.05, 1), (0.05, 0.1, 2), (0.3, 0.05, 0)])
def test_n_h_is_the_smallest_sample_size(a, d, h):
    n = AU.n_h(a, d, h)
    assert binom.cdf(h, n, a) <= d < binom.cdf(h, n - 1, a)
    if h == 0:
        assert n == math.ceil(math.log(d) / math.log(1 - a))


def test_census_leaves_enough_audit_draws():
    rng = np.random.default_rng(0)
    w, s = rng.lognormal(0, 1, 2000), rng.random(2000)
    b, order, ok = AU.choose_census(w, s, 400, 0.02, 0.05, 1, 25)
    assert ok and b % 25 == 0
    W_U, W_V = w.sum(), w.sum() - w[order[:b]].sum()
    assert b + AU.n_h(0.02 * W_U / W_V, 0.05, 1) <= 400


def test_census_actives_are_never_kept():
    pop = AU.synthetic_population(1500, 0.08, "informative", np.random.default_rng(3))
    lam = AU.lambda_grid(pop["s"], 20)
    r = AU.run_audit(pop["w"], pop["s"], pop["a"], 400, 0.02, 0.05, 1, lam, np.random.default_rng(1))
    assert r["census"] + r["audit_draws"] == 400 and r["theta"] >= 0


@pytest.mark.parametrize("score", ["random", "inverted", "adversarial"])
def test_violation_rate_within_delta(score):
    pop = AU.synthetic_population(2000, 0.06, score, np.random.default_rng(11))
    lam = AU.lambda_grid(AU.synthetic_population(2000, 0.06, score, np.random.default_rng(12))["s"], 20)
    r = AU.simulate(pop, 200, 0.05, 0.05, 1, lam, seeds=300)
    assert r["violation_rate"] <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 300)
