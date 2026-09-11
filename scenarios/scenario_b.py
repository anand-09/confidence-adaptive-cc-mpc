"""Scenario B: occluded pedestrian crosswalk approach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caspc.confidence import ConfidenceEstimator, ConfidenceParams
from caspc.mpc import LongitudinalIntegrator, MPCSolver, SimulationLog
from caspc.risk import RiskModulator, RiskParams


@dataclass
class ScenarioBConfig:
    ts: float = 0.2
    duration: float = 10.0
    v0: float = 12.0
    crosswalk_x: float = 88.0
    reveal_time: float = 4.0
    d_min: float = 2.0
    horizon: int = 10
    conf: ConfidenceParams = ConfidenceParams()
    risk: RiskParams = RiskParams()
    fixed_delta: float = 0.05
    seed: int = 0


def pedestrian_present(t: float, reveal_time: float) -> bool:
    return t >= reveal_time


def run_scenario_b(mode: str = "adaptive", cfg: ScenarioBConfig | None = None) -> SimulationLog:
    cfg = cfg or ScenarioBConfig()
    model = LongitudinalIntegrator(ts=cfg.ts)
    mpc = MPCSolver(model=model, horizon=cfg.horizon)
    risk_mod = RiskModulator(cfg.risk)
    conf_est = ConfidenceEstimator(cfg.conf, c=0.88)

    steps = int(cfg.duration / cfg.ts)
    x_ego = np.array([0.0, cfg.v0])
    x_ref = np.array([cfg.crosswalk_x - 5.0, cfg.v0 * 0.85])
    x_pred_stored = np.array([cfg.crosswalk_x, 0.0])

    log = SimulationLog()
    for k in range(steps):
        t = k * cfg.ts
        present = pedestrian_present(t + cfg.ts, cfg.reveal_time)
        if present:
            agent_meas = np.array([cfg.crosswalk_x, 0.0])
        else:
            agent_meas = np.array([cfg.crosswalk_x + 40.0, 0.0])

        metrics = conf_est.step(agent_meas, x_pred_stored)
        c_k = metrics["c"]
        rb = risk_mod.evaluate(c_k) if mode == "adaptive" else risk_mod.fixed_delta(cfg.fixed_delta)

        u0, stats = mpc.solve_longitudinal(
            x0=x_ego,
            x_ref=x_ref,
            rho=rb["rho"],
            d_min=cfg.d_min,
            obstacle_x=cfg.crosswalk_x,
        )

        x_ego = model.A @ x_ego + model.B @ u0
        gap = cfg.crosswalk_x - x_ego[0]

        log.t.append(t)
        log.ego_state.append(x_ego.copy())
        log.controls.append(u0.copy())
        log.confidence.append(c_k)
        log.e_bar.append(metrics["e_bar"])
        log.delta.append(rb["delta"])
        log.rho.append(rb["rho"])
        log.gaps.append(gap)
        log.solve_ms.append(stats.solve_time_ms)
        log.violations.append(gap < cfg.d_min)

        x_pred_stored = agent_meas.copy()

    log.extra["speed"] = np.array([s[1] for s in log.ego_state])
    return log
