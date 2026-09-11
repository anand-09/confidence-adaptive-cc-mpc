"""Export detailed simulation logs to CSV/JSON for analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios.monte_carlo import MonteCarloConfig, run_monte_carlo
from scenarios.scenario_a import run_scenario_a
from scenarios.scenario_b import run_scenario_b
from scenarios.scenario_c import ScenarioCConfig, run_scenario_c, window_backoff_stats


def _log_to_df(log, extra_cols: dict | None = None) -> pd.DataFrame:
    d = log.to_dict()
    df = pd.DataFrame(
        {
            "t": d["t"],
            "confidence": d["confidence"],
            "e_bar": d["e_bar"],
            "delta": d["delta"],
            "rho": d["rho"],
            "gap": d["gaps"],
            "violation": d["violations"],
            "solve_ms": d["solve_ms"],
        }
    )
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    return df


def export_all(output_dir: str | Path = "outputs/data") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for mode in ("adaptive", "fixed"):
        la = run_scenario_a(mode)
        _log_to_df(la, {"lateral": la.extra["lateral"]}).to_csv(out / f"scenario_a_{mode}.csv", index=False)

    for mode in ("adaptive", "fixed"):
        lb = run_scenario_b(mode)
        _log_to_df(lb, {"speed": lb.extra["speed"]}).to_csv(out / f"scenario_b_{mode}.csv", index=False)

    for mode in ("adaptive", "fixed_split"):
        lc = run_scenario_c(mode)
        _log_to_df(
            lc,
            {
                "c1": lc.extra["c1"],
                "c2": lc.extra["c2"],
                "delta1": lc.extra["delta1"],
                "delta2": lc.extra["delta2"],
                "rho1": lc.extra["rho1"],
                "rho2": lc.extra["rho2"],
                "gap1": lc.extra["gap1"],
                "gap2": lc.extra["gap2"],
            },
        ).to_csv(out / f"scenario_c_{mode}.csv", index=False)

    cfg_c = ScenarioCConfig()
    stats = {
        "scenario_c_windows": {
            "adaptive": window_backoff_stats(run_scenario_c("adaptive"), cfg_c),
            "fixed_split": window_backoff_stats(run_scenario_c("fixed_split"), cfg_c),
        }
    }
    for regime in ("moderate", "severe"):
        mc = run_monte_carlo(regime, mc_cfg=MonteCarloConfig(n_trials=100))
        stats[f"montecarlo_{regime}"] = {
            m: {
                "mean_backoff": s["mean_backoff"],
                "std_backoff": s["std_backoff"],
                "violation_rate": s["violation_rate"],
                "solve_ms_mean": s["solve_ms_mean"],
            }
            for m, s in mc.items()
        }

    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(f"Detailed CSV/JSON exported to {out.resolve()}")


if __name__ == "__main__":
    export_all()
