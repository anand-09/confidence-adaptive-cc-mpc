"""Literature-faithful adaptive-tightening baselines (Ma LPES, Engelaar GMM).

Both keep the chance level delta fixed and adapt only the uncertainty geometry,
which is the distinction claimed in the paper: they do not set delta from a
derived per-agent reliability estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caspc.risk import RiskParams, sigma_eff


def cantelli_rho(sigma: float, delta: float) -> float:
    d = float(np.clip(delta, 1e-6, 1.0 - 1e-6))
    return float(sigma) * np.sqrt((1.0 - d) / d)


@dataclass
class LPESTightening:
    """Ma-style learned prediction-error set: EWMA residual radius, fixed delta.

    The set is an outer approximation, so the radius is floored at sigma0.
    Consequently rho >= sigma0 * sqrt((1-delta_fix)/delta_fix) at every step.
    """

    delta_fix: float = 0.05
    sigma0: float = 1.0
    gamma: float = 0.30
    sigma: float = 1.0

    def reset(self, sigma: float | None = None) -> None:
        self.sigma = self.sigma0 if sigma is None else float(sigma)

    def step(self, error: float) -> dict[str, float]:
        self.sigma = (1.0 - self.gamma) * self.sigma + self.gamma * abs(float(error))
        radius = max(self.sigma, self.sigma0)
        rho = cantelli_rho(radius, self.delta_fix)
        return {"sigma": radius, "delta": self.delta_fix, "rho": rho}


@dataclass
class GMMTightening:
    """Engelaar-style two-mode Gaussian mixture with similar-variance components.

    Mode weights (c, 1-c), component stds (sigma0, sigma0(1+kappa)), fixed delta.
    Mixture std coincides with sigma_eff(c) in the paper; only delta is not adapted.
    """

    params: RiskParams = RiskParams()
    delta_fix: float = 0.05

    def evaluate(self, c: float) -> dict[str, float]:
        sig = float(sigma_eff(c, self.params))
        rho = cantelli_rho(sig, self.delta_fix)
        return {"sigma_eff": sig, "delta": self.delta_fix, "rho": rho}
