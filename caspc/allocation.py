"""Multi-agent confidence-weighted risk allocation (Prop. 5, Thm. 4)."""

from __future__ import annotations

import numpy as np

from caspc.risk import RiskParams, compute_backoff, delta_of_c


def confidence_weighted_allocation(
    confidences: list[float],
    delta_total: float,
    risk_params: RiskParams,
) -> list[dict[str, float]]:
    nominal = [delta_of_c(c, risk_params) for c in confidences]
    denom = sum(nominal)
    if denom <= 0:
        share = delta_total / len(confidences)
        allocated = [share] * len(confidences)
    else:
        allocated = [delta_total * m / denom for m in nominal]

    out = []
    for c, d_tilde in zip(confidences, allocated):
        b = compute_backoff(c, risk_params, delta=d_tilde)
        out.append({"c": c, "delta_nominal": delta_of_c(c, risk_params), "delta_tilde": d_tilde, **b})
    return out
