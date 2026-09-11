"""Monte Carlo evaluation for Scenario A (Sec. V-E)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenarios.scenario_a import LeaderTruth, ScenarioAConfig, run_scenario_a


@dataclass
class MonteCarloConfig:
    n_trials: int = 100
    seed: int = 42
    moderate: dict | None = None
    severe: dict | None = None

    def __post_init__(self) -> None:
        self.moderate = self.moderate or {
            "decel_range": (2.0, 3.5),
            "onset_range": (3.5, 4.5),
            "duration_range": (1.5, 2.5),
            "v_final_range": (14.0, 16.0),
        }
        self.severe = self.severe or {
            "decel_range": (2.0, 5.0),
            "onset_range": (3.0, 5.0),
            "duration_range": (1.0, 3.0),
            "v_final_range": (10.0, 16.0),
        }


def _sample_leader(rng: np.random.Generator, regime: dict) -> LeaderTruth:
    v0 = 22.0
    onset = rng.uniform(*regime["onset_range"])
    duration = rng.uniform(*regime["duration_range"])
    v_final = rng.uniform(*regime["v_final_range"])
    return LeaderTruth(x0=80.0, v0=v0, brake_start=onset, brake_end=onset + duration, v_final=v_final)


def run_monte_carlo(
    regime_name: str = "moderate",
    methods: list[str] | None = None,
    mc_cfg: MonteCarloConfig | None = None,
) -> dict:
    mc_cfg = mc_cfg or MonteCarloConfig()
    regime = mc_cfg.moderate if regime_name == "moderate" else mc_cfg.severe
    methods = methods or ["fixed", "worst_case", "naive_reactive", "adaptive"]
    rng = np.random.default_rng(mc_cfg.seed if regime_name == "moderate" else mc_cfg.seed + 1)

    results = {m: {"mean_rho": [], "violations": [], "solve_ms": []} for m in methods}

    for trial in range(mc_cfg.n_trials):
        leader = _sample_leader(rng, regime)
        noise = rng.normal(0.0, 0.05)
        for method in methods:
            cfg = ScenarioAConfig(leader=leader, noise_std=abs(noise), seed=mc_cfg.seed + trial)
            log = run_scenario_a(mode=method, cfg=cfg)
            results[method]["mean_rho"].append(float(np.mean(log.rho)))
            results[method]["violations"].append(any(log.violations))
            results[method]["solve_ms"].extend(log.solve_ms)

    summary = {}
    for method, data in results.items():
        summary[method] = {
            "mean_backoff": float(np.mean(data["mean_rho"])),
            "std_backoff": float(np.std(data["mean_rho"])),
            "violation_rate": float(np.mean(data["violations"])),
            "solve_ms_mean": float(np.mean(data["solve_ms"])),
            "solve_ms_median": float(np.median(data["solve_ms"])),
            "solve_ms_p95": float(np.percentile(data["solve_ms"], 95)),
            "solve_ms_p99": float(np.percentile(data["solve_ms"], 99)),
            "solve_ms_max": float(np.max(data["solve_ms"])),
            "backoffs": np.array(data["mean_rho"]),
        }
    return summary
