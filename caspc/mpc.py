"""Discrete-time vehicle models and MPC solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class DoubleIntegrator2D:
    ts: float = 0.2
    ax_max: float = 2.5
    ay_max: float = 2.0
    vy_max: float = 3.0

    @property
    def n_state(self) -> int:
        return 4

    @property
    def n_input(self) -> int:
        return 2

    @property
    def A(self) -> np.ndarray:
        ts = self.ts
        return np.array([[1, ts, 0, 0], [0, 1, 0, 0], [0, 0, 1, ts], [0, 0, 0, 1]], dtype=float)

    @property
    def B(self) -> np.ndarray:
        ts = self.ts
        return np.array([[0, 0], [ts, 0], [0, 0], [0, ts]], dtype=float)


@dataclass(frozen=True)
class LongitudinalIntegrator:
    ts: float = 0.2
    a_max: float = 3.5

    @property
    def n_state(self) -> int:
        return 2

    @property
    def n_input(self) -> int:
        return 1

    @property
    def A(self) -> np.ndarray:
        ts = self.ts
        return np.array([[1.0, ts], [0.0, 1.0]], dtype=float)

    @property
    def B(self) -> np.ndarray:
        ts = self.ts
        return np.array([[0.5 * ts**2], [ts]], dtype=float)


@dataclass
class MPCWeights:
    q_diag: np.ndarray
    r_diag: np.ndarray
    qf_diag: np.ndarray | None = None


@dataclass
class SolveStats:
    status: str
    solve_time_ms: float
    used_fallback: bool = False


@dataclass
class MPCSolver:
    model: DoubleIntegrator2D | LongitudinalIntegrator
    horizon: int = 10
    weights: MPCWeights | None = None
    solver_primary: str = "OSQP"
    solver_fallback: str = "SCS"

    def __post_init__(self) -> None:
        n = self.model.n_state
        m = self.model.n_input
        if self.weights is None:
            if isinstance(self.model, DoubleIntegrator2D):
                self.weights = MPCWeights(
                    q_diag=np.diag([0.2, 0.5, 8.0, 1.0]),
                    r_diag=np.diag([0.5, 0.8]),
                    qf_diag=np.diag([1.0, 2.0, 20.0, 2.0]),
                )
            else:
                self.weights = MPCWeights(
                    q_diag=np.diag([0.05, 1.0]),
                    r_diag=np.diag([0.3]),
                    qf_diag=np.diag([0.2, 3.0]),
                )

    def _solve_problem(self, prob: cp.Problem) -> SolveStats:
        import time

        t0 = time.perf_counter()
        try:
            prob.solve(solver=self.solver_primary, warm_start=True, verbose=False)
            used_fallback = False
            if prob.status not in ("optimal", "optimal_inaccurate"):
                prob.solve(solver=self.solver_fallback, warm_start=True, verbose=False)
                used_fallback = True
        except Exception:
            prob.solve(solver=self.solver_fallback, warm_start=True, verbose=False)
            used_fallback = True
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return SolveStats(status=str(prob.status), solve_time_ms=dt_ms, used_fallback=used_fallback)

    def solve_longitudinal(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        rho: float,
        d_min: float,
        obstacle_x: float,
        v_min: float = 0.0,
        v_max: float = 40.0,
    ) -> tuple[np.ndarray, SolveStats]:
        assert isinstance(self.model, LongitudinalIntegrator)
        N = self.horizon
        A, B = self.model.A, self.model.B
        Q, R = self.weights.q_diag, self.weights.r_diag
        Qf = self.weights.qf_diag if self.weights.qf_diag is not None else 5.0 * Q

        x = cp.Variable((N + 1, 2))
        u = cp.Variable((N, 1))
        constraints = [x[0] == x0]
        cost = 0
        for k in range(N):
            constraints += [
                x[k + 1] == A @ x[k] + B @ u[k],
                u[k] <= self.model.a_max,
                u[k] >= -self.model.a_max,
                x[k + 1, 1] <= v_max,
                x[k + 1, 1] >= v_min,
                obstacle_x - x[k + 1, 0] >= d_min + rho,
            ]
            err = x[k] - x_ref
            cost += cp.quad_form(err, Q) + cp.quad_form(u[k], R)
        err_f = x[N] - x_ref
        cost += cp.quad_form(err_f, Qf)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        stats = self._solve_problem(prob)
        u0 = np.array(u.value[0]).reshape(-1) if u.value is not None else np.zeros(1)
        return u0, stats

    def solve_lane_change(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        rho: float,
        d_min: float,
        leader_pred: np.ndarray,
    ) -> tuple[np.ndarray, SolveStats]:
        """leader_pred[j] = predicted leader x at step j+1."""
        assert isinstance(self.model, DoubleIntegrator2D)
        N = self.horizon
        A, B = self.model.A, self.model.B
        Q, R = self.weights.q_diag, self.weights.r_diag
        Qf = self.weights.qf_diag if self.weights.qf_diag is not None else 5.0 * Q

        x = cp.Variable((N + 1, 4))
        u = cp.Variable((N, 2))
        constraints = [x[0] == x0]
        cost = 0
        for k in range(N):
            gap = leader_pred[k] - x[k + 1, 0]
            constraints += [
                x[k + 1] == A @ x[k] + B @ u[k],
                u[k, 0] <= self.model.ax_max,
                u[k, 0] >= -self.model.ax_max,
                u[k, 1] <= self.model.ay_max,
                u[k, 1] >= -self.model.ay_max,
                x[k + 1, 3] <= self.model.vy_max,
                x[k + 1, 3] >= -self.model.vy_max,
                gap >= d_min + rho,
            ]
            err = x[k] - x_ref
            cost += cp.quad_form(err, Q) + cp.quad_form(u[k], R)
        err_f = x[N] - x_ref
        cost += cp.quad_form(err_f, Qf)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        stats = self._solve_problem(prob)
        u0 = np.array(u.value[0]).reshape(-1) if u.value is not None else np.zeros(2)
        return u0, stats

    def solve_intersection_longitudinal(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        agents: list[dict],
        v_min: float = 0.0,
        v_max: float = 40.0,
    ) -> tuple[np.ndarray, SolveStats]:
        """agents: list of {obstacle_x, d_min, rho}."""
        assert isinstance(self.model, LongitudinalIntegrator)
        N = self.horizon
        A, B = self.model.A, self.model.B
        Q, R = self.weights.q_diag, self.weights.r_diag
        Qf = self.weights.qf_diag if self.weights.qf_diag is not None else 5.0 * Q

        x = cp.Variable((N + 1, 2))
        u = cp.Variable((N, 1))
        constraints = [x[0] == x0]
        cost = 0
        for k in range(N):
            constraints += [
                x[k + 1] == A @ x[k] + B @ u[k],
                u[k] <= self.model.a_max,
                u[k] >= -self.model.a_max,
                x[k + 1, 1] <= v_max,
                x[k + 1, 1] >= v_min,
            ]
            for ag in agents:
                constraints.append(ag["obstacle_x"] - x[k + 1, 0] >= ag["d_min"] + ag["rho"])
            err = x[k] - x_ref
            cost += cp.quad_form(err, Q) + cp.quad_form(u[k], R)
        err_f = x[N] - x_ref
        cost += cp.quad_form(err_f, Qf)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        stats = self._solve_problem(prob)
        u0 = np.array(u.value[0]).reshape(-1) if u.value is not None else np.zeros(1)
        return u0, stats


ControllerMode = Literal["adaptive", "fixed", "worst_case", "naive_reactive", "fixed_split"]


@dataclass
class SimulationLog:
    t: list[float] = field(default_factory=list)
    ego_state: list[np.ndarray] = field(default_factory=list)
    controls: list[np.ndarray] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    e_bar: list[float] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    solve_ms: list[float] = field(default_factory=list)
    violations: list[bool] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "t": np.array(self.t),
            "ego_state": np.vstack(self.ego_state),
            "controls": np.vstack(self.controls),
            "confidence": np.array(self.confidence),
            "e_bar": np.array(self.e_bar),
            "delta": np.array(self.delta),
            "rho": np.array(self.rho),
            "gaps": np.array(self.gaps),
            "solve_ms": np.array(self.solve_ms),
            "violations": np.array(self.violations),
            **self.extra,
        }
