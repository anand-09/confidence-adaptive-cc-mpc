"""Confidence estimation via regularized weighted least squares (Prop. 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def saturate01(x: float | np.ndarray) -> float | np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass(frozen=True)
class ConfidenceParams:
    alpha: float = 0.30
    beta: float = 0.33
    e_norm: float = 0.5

    def validate(self) -> None:
        if not (0.0 < self.beta < 2.0):
            raise ValueError("Require 0 < beta < 2.")
        if not (0.0 < self.alpha <= self.beta <= 1.0):
            raise ValueError("Require 0 < alpha <= beta <= 1.")


@dataclass
class ConfidenceEstimator:
    """MAP confidence filter with structural decay prior."""

    params: ConfidenceParams = ConfidenceParams()
    c: float = 0.8

    def __post_init__(self) -> None:
        self.params.validate()
        self.c = float(saturate01(self.c))

    @property
    def weights(self) -> tuple[float, float, float]:
        p = self.params
        w1 = 1.0 - p.beta
        w2 = p.alpha
        w3 = p.beta - p.alpha
        return w1, w2, w3

    def evidence_from_error(self, error: float) -> tuple[float, float]:
        e_bar = float(saturate01(error / self.params.e_norm))
        y = 1.0 - e_bar
        return e_bar, y

    def update(self, y: float) -> float:
        """Closed-form MAP update c_{k+1} = (1-beta)c_k + alpha y_k."""
        p = self.params
        c_next = (1.0 - p.beta) * self.c + p.alpha * float(y)
        self.c = float(saturate01(c_next))
        return self.c

    def step(self, measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        error = float(np.linalg.norm(measured - predicted))
        e_bar, y = self.evidence_from_error(error)
        c = self.update(y)
        return {"error": error, "e_bar": e_bar, "y": y, "c": c}

    def reset(self, c0: float = 0.8) -> None:
        self.c = float(saturate01(c0))


class NaiveReactiveRisk:
    """Baseline: map raw normalized error directly to confidence (no filtering)."""

    def __init__(self, e_norm: float = 0.5):
        self.e_norm = e_norm
        self.c = 0.8

    def step(self, measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        error = float(np.linalg.norm(measured - predicted))
        e_bar = float(saturate01(error / self.e_norm))
        self.c = 1.0 - e_bar
        return {"error": error, "e_bar": e_bar, "y": self.c, "c": self.c}

    def reset(self, c0: float = 0.8) -> None:
        self.c = c0
