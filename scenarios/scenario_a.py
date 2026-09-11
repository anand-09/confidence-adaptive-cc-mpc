"""Scenario A: lane change with braking cut-in vehicle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caspc.confidence import ConfidenceEstimator, ConfidenceParams, NaiveReactiveRisk
from caspc.mpc import DoubleIntegrator2D, MPCSolver, SimulationLog
from caspc.risk import RiskModulator, RiskParams


@dataclass
class LeaderTruth:
    x0: float = 12.0
    v0: float = 20.0
    brake_start: float = 4.0
    brake_end: float = 6.0
    v_final: float = 14.0

    def state(self, t: float) -> tuple[float, float]:
        if t < self.brake_start:
            v = self.v0
            x = self.x0 + self.v0 * t
        elif t <= self.brake_end:
            dt = t - self.brake_start
            T = self.brake_end - self.brake_start
            a = (self.v_final - self.v0) / T
            v = self.v0 + a * dt
            x = self.x0 + self.v0 * self.brake_start + self.v0 * dt + 0.5 * a * dt**2
        else:
            dt_brake = self.brake_end - self.brake_start
            a = (self.v_final - self.v0) / dt_brake
            x_brake_end = self.x0 + self.v0 * self.brake_start + self.v0 * dt_brake + 0.5 * a * dt_brake**2
            v = self.v_final
            x = x_brake_end + self.v_final * (t - self.brake_end)
        return x, v


@dataclass
class ScenarioAConfig:
    ts: float = 0.2
    duration: float = 10.0
    lane_offset: float = 3.6
    d_min: float = 5.0
    vx0: float = 20.0
    horizon: int = 10
    conf: ConfidenceParams = ConfidenceParams()
    risk: RiskParams = RiskParams()
    fixed_delta: float = 0.05
    leader: LeaderTruth = field(default_factory=LeaderTruth)
    noise_std: float = 0.0
    seed: int = 0


def run_scenario_a(mode: str = "adaptive", cfg: ScenarioAConfig | None = None) -> SimulationLog:
    cfg = cfg or ScenarioAConfig()
    rng = np.random.default_rng(cfg.seed)

    model = DoubleIntegrator2D(ts=cfg.ts)
    mpc = MPCSolver(model=model, horizon=cfg.horizon)
    risk_mod = RiskModulator(cfg.risk)
    conf_est = ConfidenceEstimator(cfg.conf, c=0.85)
    naive = NaiveReactiveRisk(cfg.conf.e_norm)

    steps = int(cfg.duration / cfg.ts)
    x_ego = np.array([0.0, cfg.vx0, 0.0, 0.0])
    x_lead0, v_lead0 = cfg.leader.state(cfg.ts)
    x_pred_stored = np.array([x_lead0, v_lead0])  # one-step prediction x_hat_{k|k-1}

    log = SimulationLog()
    x_ref = np.array([0.0, cfg.vx0, cfg.lane_offset, 0.0])

    for k in range(steps):
        t = k * cfg.ts
        x_lead, v_lead = cfg.leader.state(t + cfg.ts)
        x_lead_meas = x_lead + rng.normal(0.0, cfg.noise_std)
        agent_meas = np.array([x_lead_meas])
        pred_for_conf = np.array([x_pred_stored[0]])

        if mode == "naive_reactive":
            metrics = naive.step(agent_meas, pred_for_conf)
            c_k = metrics["c"]
            rb = risk_mod.evaluate(c_k)
        elif mode == "adaptive":
            metrics = conf_est.step(agent_meas, pred_for_conf)
            c_k = metrics["c"]
            rb = risk_mod.evaluate(c_k)
        elif mode == "worst_case":
            metrics = conf_est.step(agent_meas, pred_for_conf)
            c_k = metrics["c"]
            rb = risk_mod.worst_case()
        else:
            metrics = conf_est.step(agent_meas, pred_for_conf)
            c_k = metrics["c"]
            rb = risk_mod.fixed_delta(cfg.fixed_delta)

        # CV lag only during active braking window
        if cfg.leader.brake_start <= t <= cfg.leader.brake_end:
            v_pred = float(x_pred_stored[1])
        else:
            v_pred = float(v_lead)
        leader_pred = []
        x_p = float(x_lead_meas)
        for _ in range(cfg.horizon):
            x_p += v_pred * cfg.ts
            leader_pred.append(x_p)

        u0, stats = mpc.solve_lane_change(
            x0=x_ego,
            x_ref=x_ref,
            rho=rb["rho"],
            d_min=cfg.d_min,
            leader_pred=np.array(leader_pred),
        )

        x_ego = model.A @ x_ego + model.B @ u0
        gap = x_lead - x_ego[0]

        log.t.append(t)
        log.ego_state.append(x_ego.copy())
        log.controls.append(u0.copy())
        log.confidence.append(c_k)
        log.e_bar.append(metrics.get("e_bar", 0.0))
        log.delta.append(rb["delta"])
        log.rho.append(rb["rho"])
        log.gaps.append(gap)
        log.solve_ms.append(stats.solve_time_ms)
        log.violations.append(gap < cfg.d_min)

        # Store one-step CV prediction for next confidence update
        x_pred_stored = np.array([x_lead_meas + v_pred * cfg.ts, v_pred])

    log.extra["lateral"] = np.array([s[2] for s in log.ego_state])
    log.extra["leader_x"] = np.array([cfg.leader.state(t)[0] for t in log.t])
    return log
