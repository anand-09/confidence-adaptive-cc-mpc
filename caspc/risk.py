"""Confidence-modulated risk and Cantelli back-off (Eqs. 5-7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskParams:
    delta_min: float = 0.01
    delta_max: float = 0.20
    lam: float = 3.0
    sigma0: float = 1.0
    kappa: float = 0.6

    def validate(self) -> None:
        if not (0.0 < self.delta_min < self.delta_max < 1.0):
            raise ValueError("Require 0 < delta_min < delta_max < 1.")


def delta_of_c(c: float, p: RiskParams) -> float:
    return p.delta_min + (p.delta_max - p.delta_min) * (1.0 - np.exp(-p.lam * c))


def sigma_eff(c: float, p: RiskParams) -> float:
    return p.sigma0 * (1.0 + p.kappa * (1.0 - c))


def compute_backoff(c: float, p: RiskParams, delta: float | None = None) -> dict[str, float]:
    d = delta if delta is not None else delta_of_c(c, p)
    d = float(np.clip(d, 1e-6, 1.0 - 1e-6))
    sig = sigma_eff(c, p)
    rho = sig * np.sqrt((1.0 - d) / d)
    return {"delta": d, "sigma_eff": sig, "rho": float(rho)}


@dataclass
class RiskModulator:
    params: RiskParams = RiskParams()

    def __post_init__(self) -> None:
        self.params.validate()

    def evaluate(self, c: float, delta_override: float | None = None) -> dict[str, float]:
        return compute_backoff(c, self.params, delta=delta_override)

    def fixed_delta(self, delta_fix: float) -> dict[str, float]:
        d = float(np.clip(delta_fix, 1e-6, 1.0 - 1e-6))
        sig = self.params.sigma0
        rho = sig * np.sqrt((1.0 - d) / d)
        return {"delta": d, "sigma_eff": sig, "rho": float(rho)}

    def worst_case(self) -> dict[str, float]:
        return self.evaluate(c=0.0)
