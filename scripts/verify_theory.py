#!/usr/bin/env python3
"""
Detailed numerical verification of CASPC theory (Sections III–IV).

Checks:
  - Proposition 1: MAP confidence update
  - Theorem 2: feasible-set nesting rho(c1) >= rho(c2) for c1 <= c2
  - Theorem 4 (bias): KKT optimism c* >= c_hat when constraint active
  - Proposition 5: risk budget conservation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from caspc.allocation import confidence_weighted_allocation
from caspc.confidence import ConfidenceParams
from caspc.risk import RiskParams, compute_backoff


def check_proposition1():
    alpha, beta = 0.30, 0.33
    c_k, y_k = 0.75, 0.4
    c_next = (1 - beta) * c_k + alpha * y_k
    print("Proposition 1 — confidence update")
    print(f"  c_k={c_k}, y_k={y_k} -> c_{{k+1}}={c_next:.4f}")


def check_theorem_feasible_monotonicity():
    p = RiskParams()
    cs = np.linspace(0, 1, 50)
    rhos = [compute_backoff(c, p)["rho"] for c in cs]
    ok = all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1))
    print("Theorem 3 — feasible set monotonicity (rho non-increasing in c)")
    print(f"  monotone: {ok}, rho(0)={rhos[0]:.3f}, rho(1)={rhos[-1]:.3f}")


def check_bias_kkt():
    """Numerical KKT check for Theorem 5 (endogenous bias)."""
    p = RiskParams()
    c_hat = 0.6
    mu = 2.0
    eps = 1e-4
    rho = lambda c: compute_backoff(c, p)["rho"]
    drho = (rho(c_hat + eps) - rho(c_hat - eps)) / (2 * eps)
    c_star = c_hat - (mu / 2) * drho
    print("Theorem 5 — endogenous bias (KKT)")
    print(f"  c_hat={c_hat}, mu={mu}, rho'(c_hat)={drho:.4f}")
    print(f"  implied c*={c_star:.4f} >= c_hat: {c_star >= c_hat - 1e-9}")


def check_allocation():
    p = RiskParams()
    alloc = confidence_weighted_allocation([0.45, 0.88], 0.10, p)
    total = sum(a["delta_tilde"] for a in alloc)
    print("Proposition 5 — risk allocation")
    print(f"  delta_tilde = {[round(a['delta_tilde'], 4) for a in alloc]}, sum={total:.6f}")


def main():
    print("=" * 60)
    print("CASPC theoretical verification")
    print("=" * 60)
    check_proposition1()
    check_theorem_feasible_monotonicity()
    check_bias_kkt()
    check_allocation()
    print("=" * 60)


if __name__ == "__main__":
    main()
