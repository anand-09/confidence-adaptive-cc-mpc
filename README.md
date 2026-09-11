# Confidence-Adaptive Chance-Constrained MPC (CASPC)

Code and reported results for the paper

**Confidence-Adaptive Chance-Constrained Model Predictive Control for Autonomous Driving**

Anand Singh (Institute of Science Tokyo), Johannes Betz and Mattia Piccinini (Technical University of Munich)

Contact: `as7376835@gmail.com`

The planner keeps a per-agent **confidence state** from a regularized MAP estimator and uses it online to set the chance-constraint risk $\delta(c)$ and Cantelli back-off $\rho(c)$. Confidence is estimated **outside** the QP (estimate-then-plan). Jointly optimizing confidence with control is shown to be optimistically biased (Theorem 3).

## Reported results (paper)

All numbers below match the manuscript. Lower mean back-off at a matched violation rate is better.

### Scenarios A–B (single closed-loop rollouts, 10 s)

| Scenario | Controller | Min. gap [m] | Mean back-off [m] | Violations |
|---|---|---:|---:|---:|
| A: braking cut-in | Fixed $\delta=0.05$ | 8.03 | 4.36 | 0 |
| A: braking cut-in | **Adaptive (ours)** | 6.87 | **2.56 (−41%)** | 0 |
| B: occluded pedestrian | Fixed $\delta=0.05$ | 7.20 | 4.36 | 0 |
| B: occluded pedestrian | **Adaptive (ours)** | 5.26 | **2.29 (−48%)** | 0 |

During Scenario A, $c_k\in[0.49,0.87]$, $\delta(c_k)\in[0.157,0.186]$, $\rho(c_k)\in[2.26,3.02]$ m.

### Scenario C (two-agent intersection, $\delta_{\mathrm{total}}=0.10$)

Per-agent back-off $\rho^{(i)}$ [m] by window. Adaptive reallocates a **conserved** risk budget; both controllers finish with zero violations.

| Window | Fixed $\rho^{(1)}$ | Fixed $\rho^{(2)}$ | Adaptive $\tilde\rho^{(1)}$ | Adaptive $\tilde\rho^{(2)}$ |
|---|---:|---:|---:|---:|
| Nominal ($t<4$) | 4.66 | 4.64 | 4.67 | 4.64 |
| Agent 1 mismatch ($4\le t\le 7$) | 6.37 | 4.64 | **7.94** | **4.02** |
| Agent 2 surprise ($8\le t\le 8.6$) | 4.83 | 5.48 | 4.70 | **5.64** |

On the stressed agent the back-off rises by about **25%**; on the well-predicted agent it falls by about **13%**.

### Literature adaptive-tightening baselines (Scenario A)

Same plant, constant-velocity predictor, and Cantelli algebra; only $(\delta,\sigma)$ change.

| Method | $(\delta,\sigma)$ policy | Mean $\rho$ [m] | vs. ours |
|---|---|---:|---:|
| Fixed $\delta=0.05$ | $\delta$ fixed, $\sigma=\sigma_0$ | 4.36 | +70% |
| LPES (Ma et al., 2026) | $\delta$ fixed, $\sigma=\mathrm{EWMA}\ge\sigma_0$ | 4.36 | +70% |
| GMM-SMPC (Engelaar et al., 2025) | $\delta$ fixed, $\sigma=\sigma_{\mathrm{eff}}(c)$ | 5.20 | +103% |
| Worst-case fixed | $\delta=\delta_{\min}$, $\sigma=\sigma_0$ | 9.95 | +288% |
| **Confidence-adaptive (ours)** | $\delta(c)$, $\sigma_{\mathrm{eff}}(c)$ | **2.56** | — |

Ours is the only method that adapts $\delta$ from a derived reliability estimate, so it is the only one that can go below the Cantelli floor of $4.36$ m at the same zero-violation outcome.

### Monte Carlo (100 randomized trials of Scenario A)

| Method | Moderate back-off [m] | Moderate viol. | Severe back-off [m] | Severe viol. |
|---|---:|---:|---:|---:|
| Fixed $\delta=0.05$ | $4.36\pm 0.00$ | 0% | $4.36\pm 0.00$ | 20% |
| Worst-case fixed | $9.95\pm 0.00$ | 0% | $9.95\pm 0.00$ | 10% |
| Myopic reactive | $2.45\pm 0.24$ | 0% | $2.57\pm 0.33$ | 30% |
| **Confidence-adaptive** | **$2.57\pm 0.14$** | **0%** | **$2.62\pm 0.16$** | **20%** |

- Moderate: **41.5%** less mean back-off than fixed $\delta$, **41%** lower trial-to-trial variance than myopic-reactive, 0% violations.
- Severe: same 20% violation rate as fixed $\delta$ at about **half** the margin; myopic-reactive is 30%.

### QP solver runtime (≈40,000 solves, OSQP, $T_s=200$ ms)

| Method | Mean [ms] | Median [ms] | 95th [ms] | Max [ms] |
|---|---:|---:|---:|---:|
| Fixed $\delta=0.05$ | 3.8 | 3.8 | 4.2 | 38.4 |
| Worst-case fixed | 4.2 | 3.9 | 5.4 | 111.2 |
| Myopic reactive | 3.9 | 3.8 | 4.9 | 32.8 |
| **Confidence-adaptive** | **3.8** | **3.8** | **4.0** | **25.7** |

Confidence-update overhead is $<0.01$ ms (closed-form scalar).

### Theorem 3 (endogenous confidence bias) at $\hat c=0.60$

Grid of 2001 points on $[0,1]$. When the safety constraint is active, the joint solver overstates confidence and cuts the margin.

| $\mu$ | $c^\star_{\mathrm{grid}}$ | $c^\star_{\mathrm{KKT}}$ | Bias | $\rho(\hat c)$ [m] | $\rho(c^\star)$ [m] |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.600 | 0.600 | +0.000 | 2.75 | 2.75 |
| 0.05 | 0.650 | 0.650 | +0.050 | 2.75 | 2.65 |
| 0.10 | 0.695 | 0.695 | +0.095 | 2.75 | 2.55 |
| 0.20 | 0.772 | 0.772 | +0.172 | 2.75 | 2.42 |
| 0.50 | 0.960 | 0.960 | +0.360 | 2.75 | 2.12 |

At $\mu=0.20$ the joint solver overstates confidence by **+0.17** and cuts the back-off by **12%** (2.75 → 2.42 m). That is why Algorithm 1 never treats $c$ as a decision variable.

### Public scenario protocol

| Event | Public catalog class | Residual used to set $(\sigma_0,\kappa)$ |
|---|---|---|
| A: cut-in | NHTSA/ISO 34502 cut-in, $0.36g$ | CV FDE @ 2 s: 0.89 m → $\sigma_0=1.0$ m |
| B: pedestrian | Euro NCAP obstructed AEB, 43 km/h | Ped. FDE @ 2 s: 0.17 m ≪ $\sigma_0$ |
| C: intersection | CommonRoad unsignalized conflict | CV FDE @ 3 s: 1.70 m → $\sigma_{\mathrm{eff}}(0)=1.6$ m |

## Repository layout

```
caspc/                      Core library
  confidence.py             Prop. 1: MAP confidence estimator
  risk.py                   δ(c), σ_eff(c), ρ(c)
  allocation.py             Confidence-weighted risk split
  mpc.py                    Receding-horizon QP (Algorithm 1)
  literature.py             LPES and GMM-SMPC tightenings
scenarios/                  Closed-loop Scenarios A, B, C
experiments/                Monte Carlo, sensitivity, Theorem 3
plotting/                   Paper figures
scripts/                    Theory checks and result export
tests/                      Unit tests
outputs/                    Saved numeric summaries used in the paper
caspc_acc.tex               Manuscript
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install osqp scs               # QP backends used in the paper
```

## Reproduce

```bash
# Unit tests
python -m unittest tests.test_caspc -v

# Numerical checks of the theory (no QP)
python scripts/verify_theory.py

# Theorem 3 grid vs KKT
python experiments/eval_theorem_bias.py

# Full paper pipeline (figures + 100-trial Monte Carlo; several minutes)
python run_all.py

# Faster smoke test
python run_all.py --quick
```

Single-scenario check:

```bash
python -c "from scenarios.scenario_a import run_scenario_a; log=run_scenario_a('adaptive'); print(min(log.gaps), sum(log.violations))"
```

## Parameters (Section V)

| Parameter | Value |
|---|---|
| $\alpha,\beta$ | 0.30, 0.33 |
| $e_{\mathrm{norm}}$ | 0.5 m |
| $\delta_{\min},\delta_{\max}$ | 0.01, 0.20 |
| $\lambda$ | 3 |
| $\sigma_0,\kappa$ | 1.0 m, 0.6 |
| Fixed baseline $\delta$ | 0.05 |
| Scene budget $\delta_{\mathrm{total}}$ (C) | 0.10 |
| $N$, $T_s$ | 10, 0.2 s |

## Citation

If you use this code, please cite the paper (ACC submission):

```
Anand Singh, Johannes Betz, and Mattia Piccinini,
"Confidence-Adaptive Chance-Constrained Model Predictive Control
for Autonomous Driving."
```
