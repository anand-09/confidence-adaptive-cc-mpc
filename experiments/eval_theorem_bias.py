#!/usr/bin/env python3
"""Numerical evaluation of Theorem 3 (endogenous confidence bias).

Minimizes f(c) = (c - c_hat)^2 + mu * rho(c) on a fine grid and compares the
minimizer to the KKT stationarity condition
    c* = c_hat - (mu/2) * rho'(c*).
This is the envelope-theorem reduction of (P_naive) when the safety constraint
is active with multiplier mu.

Also reports the trust-region patch of Corollary 2 on the operating interval
observed in Scenario A, c in [0.49, 0.87].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from caspc.literature import GMMTightening, LPESTightening
from caspc.risk import RiskParams, compute_backoff, delta_of_c, sigma_eff


def rho_and_deriv(c: float, p: RiskParams, eps: float = 1e-5) -> tuple[float, float]:
    rho = compute_backoff(c, p)["rho"]
    c_lo = max(0.0, c - eps)
    c_hi = min(1.0, c + eps)
    r_lo = compute_backoff(c_lo, p)["rho"]
    r_hi = compute_backoff(c_hi, p)["rho"]
    drho = (r_hi - r_lo) / (c_hi - c_lo)
    return float(rho), float(drho)


def kkt_c_star(c_hat: float, mu: float, p: RiskParams, n: int = 2001) -> dict[str, float]:
    """Solve stationarity on a grid; also return the direct grid minimizer of f."""
    cs = np.linspace(0.0, 1.0, n)
    rhos = np.array([compute_backoff(float(c), p)["rho"] for c in cs])
    f = (cs - c_hat) ** 2 + mu * rhos
    i = int(np.argmin(f))
    c_grid = float(cs[i])
    rho_grid = float(rhos[i])
    _, drho = rho_and_deriv(c_grid, p)
    c_kkt = float(np.clip(c_hat - 0.5 * mu * drho, 0.0, 1.0))
    rho_hat = compute_backoff(c_hat, p)["rho"]
    return {
        "mu": mu,
        "c_hat": c_hat,
        "c_grid": c_grid,
        "c_kkt": c_kkt,
        "bias": c_grid - c_hat,
        "rho_hat": rho_hat,
        "rho_star": rho_grid,
        "drho": drho,
        "delta_hat": delta_of_c(c_hat, p),
        "delta_star": delta_of_c(c_grid, p),
    }


def operating_lipschitz(p: RiskParams, c_lo: float = 0.49, c_hi: float = 0.87) -> float:
    cs = np.linspace(c_lo, c_hi, 401)
    derivs = [abs(rho_and_deriv(float(c), p)[1]) for c in cs]
    return float(max(derivs))


def literature_on_scenario_a(p: RiskParams) -> dict[str, float]:
    """Same-trajectory tightening comparison on the reported Scenario A c-range."""
    gmm = GMMTightening(params=p, delta_fix=0.05)
    lpes = LPESTightening(delta_fix=0.05, sigma0=p.sigma0)
    cs = [0.49, 0.68, 0.87]
    gmm_rhos = [gmm.evaluate(c)["rho"] for c in cs]
    ours = [compute_backoff(c, p)["rho"] for c in cs]
    # LPES with outer-approximation floor is constantly the fixed-delta Cantelli value
    lpes.reset()
    lpes_rho = lpes.step(0.0)["rho"]
    return {
        "ours_lo": ours[0],
        "ours_mid": ours[1],
        "ours_hi": ours[2],
        "gmm_lo": gmm_rhos[0],
        "gmm_mid": gmm_rhos[1],
        "gmm_hi": gmm_rhos[2],
        "lpes": lpes_rho,
        "fixed": lpes_rho,
        "sigma_mid": sigma_eff(0.68, p),
    }


def latex_bias_table(rows: list[dict[str, float]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Numerical evaluation of Theorem~\ref{thm:bias} at $\hat c=0.60$ (mid-braking operating point of Scenario~A). Grid: $2001$ points on $[0,1]$.}",
        r"\label{tab:bias}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$\mu$ & $c^\star_{\mathrm{grid}}$ & $c^\star_{\mathrm{KKT}}$ & Bias $c^\star-\hat c$ & $\rho(\hat c)$ [m] & $\rho(c^\star)$ [m] \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"${r['mu']:.2f}$ & ${r['c_grid']:.3f}$ & ${r['c_kkt']:.3f}$ & "
            f"${r['bias']:+.3f}$ & ${r['rho_hat']:.2f}$ & ${r['rho_star']:.2f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    p = RiskParams()
    c_hat = 0.60
    mus = [0.0, 0.05, 0.10, 0.20, 0.50]
    rows = [kkt_c_star(c_hat, mu, p) for mu in mus]
    L = operating_lipschitz(p)
    lit = literature_on_scenario_a(p)

    print("=" * 64)
    print("Theorem 3 — endogenous bias (P_naive)")
    print("=" * 64)
    print(f"{'mu':>6}  {'c_grid':>8}  {'c_kkt':>8}  {'bias':>8}  {'rho_hat':>8}  {'rho_star':>8}")
    for r in rows:
        print(
            f"{r['mu']:6.2f}  {r['c_grid']:8.4f}  {r['c_kkt']:8.4f}  "
            f"{r['bias']:8.4f}  {r['rho_hat']:8.3f}  {r['rho_star']:8.3f}"
        )
        assert r["c_grid"] + 1e-9 >= r["c_hat"]
        assert abs(r["c_grid"] - r["c_kkt"]) < 0.02

    print(f"\nLipschitz |rho'| on c in [0.49, 0.87]: L = {L:.3f} m")
    for mu_max in (0.20, 0.50):
        delta = mu_max * L / 2.0
        print(f"  Corollary 2: mu_max={mu_max:.2f}  =>  Delta={delta:.3f}")

    print("\nLiterature tightening on Scenario A (c in {0.49, 0.68, 0.87})")
    print(f"  Ours  rho = [{lit['ours_hi']:.2f}, {lit['ours_mid']:.2f}, {lit['ours_lo']:.2f}] m  (high/mid/low c)")
    print(f"  GMM   rho = [{lit['gmm_hi']:.2f}, {lit['gmm_mid']:.2f}, {lit['gmm_lo']:.2f}] m")
    print(f"  LPES  rho = {lit['lpes']:.2f} m  (floor at sigma0, identical to fixed-delta)")
    print()
    print(latex_bias_table(rows))


if __name__ == "__main__":
    main()
