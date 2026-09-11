"""
Parameter sensitivity analysis (reviewer request: effect of alpha, beta, lambda, kappa).

Runs a single representative Scenario-A trial per parameter value using the
fast Monte Carlo simulator core.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.monte_carlo_fast import METHODS, TrialParams, run_trial
from plotting.style import IEEE_COL_IN, apply_ieee_style, save_fig

# Fixed trial conditions for fair comparison
REP_A_BRAKE = 3.0
REP_T_START = 4.0
REP_T_DUR = 2.0
REP_NOISE = 0.05
SEED = 42

SWEEPS = {
    "alpha": {"values": [0.10, 0.20, 0.30, 0.40, 0.50], "fixed": dict(beta=0.33, lam=3.0, kappa=0.6)},
    "beta": {"values": [0.20, 0.27, 0.33, 0.40, 0.50], "fixed": dict(alpha=0.30, lam=3.0, kappa=0.6)},
    "lambda": {"values": [1.0, 2.0, 3.0, 4.0, 5.0], "fixed": dict(alpha=0.30, beta=0.33, kappa=0.6)},
    "kappa": {"values": [0.0, 0.3, 0.6, 0.9, 1.2], "fixed": dict(alpha=0.30, beta=0.33, lam=3.0)},
}


def run_sensitivity() -> dict:
    out = {}
    rng = np.random.default_rng(SEED)
    for param, spec in SWEEPS.items():
        rows = []
        for val in spec["values"]:
            kw = dict(spec["fixed"])
            key = "lam" if param == "lambda" else param
            kw[key] = val
            p = TrialParams(**kw)
            res = run_trial("adaptive", rng, REP_A_BRAKE, REP_T_START, REP_T_DUR, REP_NOISE, p)
            rows.append({"value": val, "mean_backoff": res["mean_backoff"], "min_gap": res["min_gap"], "violation": bool(res["violation"])})
        out[param] = rows
    return out


def plot_sensitivity(data: dict, out_dir: Path) -> None:
    apply_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_COL_IN, 3.35), layout="constrained")
    axes = axes.ravel()
    titles = {"alpha": r"Evidence gain $\alpha$", "beta": r"Prior-anchor gain $\beta$", "lambda": r"Risk modulation $\lambda$", "kappa": r"Uncertainty scale $\kappa$"}
    for ax, (param, rows) in zip(axes, data.items()):
        xs = [r["value"] for r in rows]
        bo = [r["mean_backoff"] for r in rows]
        ax.plot(xs, bo, "o-", color="#1f4e79", lw=1.6, markersize=5)
        ax.set_xlabel(titles[param])
        ax.set_ylabel("Mean back-off [m]")
        ax.grid(True, alpha=0.3)
        # Mark nominal value
        nominal = {"alpha": 0.30, "beta": 0.33, "lambda": 3.0, "kappa": 0.6}[param]
        ax.axvline(nominal, color="gray", ls="--", lw=1, label="nominal")
        ax.legend()
    save_fig(fig, out_dir / "fig_sensitivity.pdf")

    # Violation check panel (not used in the paper)
    fig2, ax2 = plt.subplots(figsize=(IEEE_COL_IN, 2.1), layout="constrained")
    for param, rows in data.items():
        xs = [r["value"] for r in rows]
        viol = [100 * r["violation"] for r in rows]
        ax2.plot(xs, viol, "o-", label=param, lw=1.4)
    ax2.set_xlabel("Parameter value")
    ax2.set_ylabel("Violation (single trial) [%]")
    ax2.legend(ncol=2)
    ax2.set_ylim(-5, 105)
    save_fig(fig2, out_dir / "fig_sensitivity_violations.pdf")


def main(out_dir: str | Path = "outputs") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run_sensitivity()
    plot_sensitivity(data, out_dir)
    with open(out_dir / "sensitivity.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    main()
