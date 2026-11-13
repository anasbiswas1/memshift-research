"""Algorithm 1 of plan v1: risk-limiting memory audit at migration (working name RLMA).

Inputs per unit u of the survivor store: exposure w_u under the new retriever, a cheap score s_u in [0, 1] fixed
before the audit, and the check outcome a_u in {0, 1} (revealed only for checked units). Census: the units with the
largest w*s are checked one by one. Audit: n = B - B_c draws from the unchecked set with probability w/W_V, with
replacement. Thresholds are tested from most quarantine to least with the lower-tail binomial p-value at
alpha' = alpha * W_U / W_V (Equations 5 to 8); the certificate is theta(K) <= alpha at confidence 1 - delta for any
score (Proposition 1).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def n_h(alpha_p: float, delta: float, h: int) -> int:
    """Smallest n with P[Binomial(n, alpha') <= h] <= delta (Equation 8). 0 when alpha' >= 1."""
    if alpha_p >= 1:
        return 0
    if alpha_p <= 0:
        return math.inf
    lo, hi = h, max(h + 1, int(math.ceil((h + 1) / alpha_p)))
    while binom.cdf(h, hi, alpha_p) > delta:
        hi *= 2
    while hi - lo > 1:                     # cdf(h, n, a) decreases in n
        mid = (lo + hi) // 2
        if binom.cdf(h, mid, alpha_p) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def choose_census(w, s, B: int, alpha: float, delta: float, h: int, step: int) -> tuple[int, np.ndarray, bool]:
    """Largest census size b on the grid with b + n_h(alpha'(b)) <= B. Returns (b, ranking, feasible)."""
    order = np.argsort(-(w * s), kind="stable")
    W_U = float(w.sum())
    csum = np.concatenate([[0.0], np.cumsum(w[order])])
    top = min(B, len(w))
    for b in range(top - top % step, -1, -step):
        W_V = W_U - csum[b]
        a_p = math.inf if W_V <= 0 else alpha * W_U / W_V
        if b + n_h(a_p, delta, h) <= B:
            return b, order, True
    return 0, order, False


def lambda_grid(calibration_scores, n: int) -> np.ndarray:
    """n quantiles of the calibration scores (ascending) plus a final threshold that keeps every unit."""
    qs = np.quantile(np.asarray(calibration_scores, float), np.linspace(0, 1, n + 1)[1:])
    return np.unique(np.concatenate([qs, [np.inf]]))


def run_audit(w, s, a, B: int, alpha: float, delta: float, h: int, lambdas, rng, step: int = 25) -> dict:
    w, s, a = np.asarray(w, float), np.asarray(s, float), np.asarray(a, int)
    N = len(w)
    W_U = float(w.sum())
    b, order, feasible = choose_census(w, s, B, alpha, delta, h, step)
    census = order[:b]
    in_census = np.zeros(N, bool); in_census[census] = True
    V = np.flatnonzero(~in_census)
    W_V = float(w[V].sum())
    n = B - b
    a_p = math.inf if W_V <= 0 else alpha * W_U / W_V
    draws = rng.choice(V, size=n, replace=True, p=w[V] / W_V) if (n > 0 and W_V > 0) else np.array([], int)
    lam_star, p_vals = None, []
    for lam in lambdas:
        X = int(np.sum(a[draws] * (s[draws] < lam))) if len(draws) else 0
        p = 0.0 if a_p >= 1 else (float(binom.cdf(X, n, a_p)) if n > 0 else 1.0)
        p_vals.append(p)
        if p <= delta:
            lam_star = lam
        else:
            break
    keep = np.zeros(N, bool)
    keep[census[a[census] == 0]] = True                      # verified inactive census units
    if lam_star is not None:
        keep[V[s[V] < lam_star]] = True
    theta = float(np.sum(w[keep] * a[keep]) / W_U) if W_U > 0 else 0.0
    benign = a == 0
    return {"B": B, "alpha": alpha, "delta": delta, "h": h, "census": b, "audit_draws": int(n),
            "unique_checks": int(b + len(np.unique(draws))), "census_feasible": feasible,
            "lambda_star": None if lam_star is None else float(lam_star), "certified": lam_star is not None,
            "theta": theta, "violation": theta > alpha,
            "kept_benign_exposure": float(w[keep & benign].sum() / max(w[benign].sum(), 1e-12)),
            "quarantined_units": int(N - keep.sum())}


def simulate(population: dict, B: int, alpha: float, delta: float, h: int, lambdas, seeds: int,
             step: int = 25) -> dict:
    """Repeats the audit on one fixed population with fresh audit randomness; returns violation rate and means."""
    out = [run_audit(population["w"], population["s"], population["a"], B, alpha, delta, h, lambdas,
                     np.random.default_rng(i), step) for i in range(seeds)]
    v = np.array([o["violation"] for o in out])
    return {"B": B, "alpha": alpha, "violation_rate": float(v.mean()),
            "violation_se": float(np.sqrt(max(v.mean() * (1 - v.mean()), 1e-12) / seeds)),
            "certified_rate": float(np.mean([o["certified"] for o in out])),
            "mean_kept_benign_exposure": float(np.mean([o["kept_benign_exposure"] for o in out])),
            "mean_census": float(np.mean([o["census"] for o in out]))}


def synthetic_population(n: int, theta_total: float, score: str, rng) -> dict:
    """Exposures from a log-normal, actives at a share that makes theta_total of the exposure mass active, and a
    score that is informative, random, inverted or adversarial (actives get the lowest scores)."""
    w = rng.lognormal(0, 1, n)
    order = rng.permutation(n)
    a = np.zeros(n, int)
    mass, target = 0.0, theta_total * w.sum()
    for i in order:
        if mass >= target:
            break
        a[i] = 1; mass += w[i]
    noise = rng.normal(0, 1, n)
    informative = 1 / (1 + np.exp(-(2.5 * a - 1.5 + noise)))
    s = {"informative": informative, "random": rng.random(n), "inverted": 1 - informative,
         "adversarial": np.where(a == 1, rng.random(n) * 0.05, 0.05 + 0.95 * rng.random(n))}[score]
    return {"w": w, "a": a, "s": s, "theta_total": float((w * a).sum() / w.sum())}
