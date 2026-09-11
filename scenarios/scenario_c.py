"""Scenario C: two-agent unsignalized intersection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caspc.allocation import confidence_weighted_allocation
from caspc.confidence import ConfidenceEstimator, ConfidenceParams
from caspc.mpc import LongitudinalIntegrator, MPCSolver, SimulationLog
from caspc.risk import RiskModulator, RiskParams


@dataclass
class ScenarioCConfig:
    ts: float = 0.2
    duration: float = 12.0
    v0: float = 12.0
    x_conflict_vehicle: float = 75.0
    x_conflict_ped: float = 105.0
    vehicle_no_yield_start: float = 4.0
    vehicle_no_yield_end: float = 7.0
    ped_reveal_time: float = 8.0
    d_min_vehicle: float = 3.0
    d_min_ped: float = 2.0
    delta_total: float = 0.10
    horizon: int = 10
    conf: ConfidenceParams = ConfidenceParams()
    risk: RiskParams = RiskParams()
    seed: int = 0


def run_scenario_c(mode: str = "adaptive", cfg: ScenarioCConfig | None = None) -> SimulationLog:
    cfg = cfg or ScenarioCConfig()
    model = LongitudinalIntegrator(ts=cfg.ts)
    mpc = MPCSolver(model=model, horizon=cfg.horizon)
    risk_mod = RiskModulator(cfg.risk)
    est1 = ConfidenceEstimator(cfg.conf, c=0.88)
    est2 = ConfidenceEstimator(cfg.conf, c=0.90)

    steps = int(cfg.duration / cfg.ts)
    x_ego = np.array([0.0, cfg.v0])
    x_ref = np.array([cfg.x_conflict_ped + 20.0, cfg.v0 * 0.9])

    pred1 = np.array([cfg.x_conflict_vehicle + 40.0])
    pred2 = np.array([cfg.x_conflict_ped + 50.0])

    log = SimulationLog()
    log.extra["c1"] = []
    log.extra["c2"] = []
    log.extra["delta1"] = []
    log.extra["delta2"] = []
    log.extra["rho1"] = []
    log.extra["rho2"] = []
    log.extra["gap1"] = []
    log.extra["gap2"] = []

    for k in range(steps):
        t = k * cfg.ts
        vehicle_conflict = cfg.vehicle_no_yield_start <= (t + cfg.ts) <= cfg.vehicle_no_yield_end
        ped_present = (t + cfg.ts) >= cfg.ped_reveal_time

        true1 = np.array([cfg.x_conflict_vehicle]) if vehicle_conflict else np.array([cfg.x_conflict_vehicle + 40.0])
        true2 = np.array([cfg.x_conflict_ped]) if ped_present else np.array([cfg.x_conflict_ped + 50.0])

        m1 = est1.step(true1, pred1[:1])
        m2 = est2.step(true2, pred2[:1])
        c1, c2 = m1["c"], m2["c"]

        if mode in ("adaptive",):
            alloc = confidence_weighted_allocation([c1, c2], cfg.delta_total, cfg.risk)
            rho1, rho2 = alloc[0]["rho"], alloc[1]["rho"]
            d1, d2 = alloc[0]["delta_tilde"], alloc[1]["delta_tilde"]
        else:  # fixed_split or fixed
            d_fix = cfg.delta_total / 2.0
            b1 = risk_mod.fixed_delta(d_fix)
            b2 = risk_mod.fixed_delta(d_fix)
            rho1, rho2 = b1["rho"], b2["rho"]
            d1 = d2 = d_fix

        agents = [
            {"obstacle_x": cfg.x_conflict_vehicle, "d_min": cfg.d_min_vehicle, "rho": rho1},
            {"obstacle_x": cfg.x_conflict_ped, "d_min": cfg.d_min_ped, "rho": rho2},
        ]
        u0, stats = mpc.solve_intersection_longitudinal(x0=x_ego, x_ref=x_ref, agents=agents)
        x_ego = model.A @ x_ego + model.B @ u0

        g1 = cfg.x_conflict_vehicle - x_ego[0]
        g2 = cfg.x_conflict_ped - x_ego[0]
        approaching1 = x_ego[0] < cfg.x_conflict_vehicle
        approaching2 = x_ego[0] < cfg.x_conflict_ped
        viol = (approaching1 and g1 < cfg.d_min_vehicle) or (approaching2 and g2 < cfg.d_min_ped)

        log.t.append(t)
        log.ego_state.append(x_ego.copy())
        log.controls.append(u0.copy())
        log.confidence.append(0.5 * (c1 + c2))
        log.e_bar.append(0.5 * (m1["e_bar"] + m2["e_bar"]))
        log.delta.append(d1 + d2)
        log.rho.append(0.5 * (rho1 + rho2))
        log.gaps.append(min(g1, g2))
        log.solve_ms.append(stats.solve_time_ms)
        log.violations.append(viol)

        log.extra["c1"].append(c1)
        log.extra["c2"].append(c2)
        log.extra["delta1"].append(d1)
        log.extra["delta2"].append(d2)
        log.extra["rho1"].append(rho1)
        log.extra["rho2"].append(rho2)
        log.extra["gap1"].append(g1)
        log.extra["gap2"].append(g2)

        if not vehicle_conflict:
            pred1 = true1.copy()
        else:
            pred1 = np.array([cfg.x_conflict_vehicle + 40.0])
        if ped_present:
            pred2 = true2.copy()
        else:
            pred2 = np.array([cfg.x_conflict_ped + 50.0])

    for key in ("c1", "c2", "delta1", "delta2", "rho1", "rho2", "gap1", "gap2"):
        log.extra[key] = np.array(log.extra[key])
    log.extra["speed"] = np.array([s[1] for s in log.ego_state])
    return log


def window_backoff_stats(log: SimulationLog, cfg: ScenarioCConfig) -> dict[str, dict[str, float]]:
    t = np.array(log.t)
    rho1 = log.extra["rho1"]
    rho2 = log.extra["rho2"]

    windows = {
        "calm": t < cfg.vehicle_no_yield_start,
        "agent1_stressed": (t >= cfg.vehicle_no_yield_start) & (t <= cfg.vehicle_no_yield_end),
        "agent2_spike": (t >= cfg.ped_reveal_time) & (t <= cfg.ped_reveal_time + 0.6),
    }
    out = {}
    for name, mask in windows.items():
        if mask.any():
            out[name] = {"rho1": float(np.mean(rho1[mask])), "rho2": float(np.mean(rho2[mask]))}
    return out
