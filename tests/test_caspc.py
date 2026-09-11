"""Unit tests for theoretical CASPC components."""

from __future__ import annotations

import unittest

import numpy as np

from caspc.allocation import confidence_weighted_allocation
from caspc.confidence import ConfidenceEstimator, ConfidenceParams
from caspc.literature import GMMTightening, LPESTightening
from caspc.risk import RiskParams, compute_backoff, delta_of_c


class TestConfidence(unittest.TestCase):
    def test_map_update_closed_form(self):
        p = ConfidenceParams(alpha=0.3, beta=0.33)
        est = ConfidenceEstimator(p, c=0.5)
        y = 0.8
        expected = (1 - p.beta) * 0.5 + p.alpha * y
        c = est.update(y)
        self.assertAlmostEqual(c, np.clip(expected, 0, 1))

    def test_weights_sum_to_one(self):
        p = ConfidenceParams(alpha=0.3, beta=0.33)
        w1, w2, w3 = ConfidenceEstimator(p).weights
        self.assertAlmostEqual(w1 + w2 + w3, 1.0)

    def test_convergence_constant_evidence(self):
        est = ConfidenceEstimator(ConfidenceParams(alpha=0.3, beta=0.33), c=0.0)
        y_inf = 0.9
        for _ in range(500):
            est.update(y_inf)
        c_star = 0.3 / 0.33 * y_inf
        self.assertAlmostEqual(est.c, c_star, places=3)


class TestRisk(unittest.TestCase):
    def test_rho_monotone_decreasing_in_c(self):
        p = RiskParams()
        rhos = [compute_backoff(c, p)["rho"] for c in np.linspace(0, 1, 20)]
        self.assertTrue(all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1)))

    def test_delta_bounds(self):
        p = RiskParams()
        self.assertAlmostEqual(delta_of_c(0, p), p.delta_min)
        self.assertLess(delta_of_c(1, p), p.delta_max + 0.01)


class TestAllocation(unittest.TestCase):
    def test_budget_exact(self):
        p = RiskParams()
        alloc = confidence_weighted_allocation([0.3, 0.9], 0.10, p)
        total = sum(a["delta_tilde"] for a in alloc)
        self.assertAlmostEqual(total, 0.10)


class TestTheoremBias(unittest.TestCase):
    def test_kkt_optimism_when_active(self):
        p = RiskParams()
        c_hat, mu = 0.60, 0.20
        cs = np.linspace(0.0, 1.0, 1001)
        rhos = np.array([compute_backoff(float(c), p)["rho"] for c in cs])
        f = (cs - c_hat) ** 2 + mu * rhos
        c_star = float(cs[int(np.argmin(f))])
        self.assertGreater(c_star, c_hat)
        self.assertLess(compute_backoff(c_star, p)["rho"], compute_backoff(c_hat, p)["rho"])

    def test_inactive_constraint_unbiased(self):
        p = RiskParams()
        c_hat = 0.60
        cs = np.linspace(0.0, 1.0, 1001)
        rhos = np.array([compute_backoff(float(c), p)["rho"] for c in cs])
        f = (cs - c_hat) ** 2 + 0.0 * rhos
        c_star = float(cs[int(np.argmin(f))])
        self.assertAlmostEqual(c_star, c_hat, places=2)


class TestLiteratureBaselines(unittest.TestCase):
    def test_lpes_never_below_fixed_delta(self):
        p = RiskParams()
        floor = compute_backoff(1.0, p, delta=0.05)["rho"]
        lpes = LPESTightening(delta_fix=0.05, sigma0=p.sigma0)
        lpes.reset()
        for e in (0.0, 0.1, 0.5, 1.2):
            rho = lpes.step(e)["rho"]
            self.assertGreaterEqual(rho, floor - 1e-9)

    def test_gmm_uses_fixed_delta(self):
        p = RiskParams()
        gmm = GMMTightening(params=p, delta_fix=0.05)
        out = gmm.evaluate(0.68)
        self.assertAlmostEqual(out["delta"], 0.05)
        self.assertGreater(out["rho"], compute_backoff(0.68, p)["rho"])


if __name__ == "__main__":
    unittest.main()
