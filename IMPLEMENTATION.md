# Confidence-Adaptive Chance-Constrained MPC — Implementation Guide

This repository implements **CASPC** from the paper *Confidence-Adaptive Chance-Constrained Model Predictive Control for Autonomous Driving*.

## Project structure

```
acc_tum/
├── caspc/                    # Core library
│   ├── confidence.py         # Prop. 1: MAP confidence estimator (Eqs. 3–4)
│   ├── risk.py               # Eqs. 5–7: δ(c), σ_eff(c), ρ(c)
│   ├── allocation.py         # Prop. 5 / Thm. 4: multi-agent risk split
│   └── mpc.py                # Receding-horizon QP MPC (Algorithm 1)
├── scenarios/
│   ├── scenario_a.py         # Lane change + braking cut-in
│   ├── scenario_b.py         # Occluded pedestrian
│   ├── scenario_c.py         # Two-agent intersection
│   └── monte_carlo.py        # 100-trial Monte Carlo (Sec. V-E)
├── plotting/
│   └── generate_figures.py   # All paper PDF figures
├── scripts/
│   └── verify_theory.py      # Numerical checks of key theorems
├── tests/
│   └── test_caspc.py         # Unit tests
├── run_all.py                # Main entry point
└── requirements.txt
```

## Setup (run on your machine)

```bash
cd /Users/anandsingh/Downloads/acc_tum
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run everything

```bash
# Full reproduction (~5–15 min depending on CPU; 100 MC trials × 2 regimes)
python run_all.py

# Quick smoke test (20 MC trials)
python run_all.py --quick

# Custom output folder
python run_all.py --output figures --mc-trials 100
```

## Individual components

```bash
# Unit tests
python -m unittest tests/test_caspc.py -v

# Theoretical verification (no QP)
python scripts/verify_theory.py

# Single scenario (Python REPL)
python -c "from scenarios.scenario_a import run_scenario_a; log=run_scenario_a('adaptive'); print(min(log.gaps), sum(log.violations))"
```

## Generated outputs (`outputs/`)

| File | Paper reference |
|------|-----------------|
| `fig_confidence.pdf` | Fig. 1 — Scenario A confidence & error |
| `fig_backoff.pdf` | Fig. 2 — Adaptive vs fixed back-off |
| `fig_traj.pdf` | Fig. 3 — Gap & lateral trajectory |
| `fig_pedestrian.pdf` | Fig. 4 — Scenario B |
| `fig_intersection.pdf` | Fig. 5 — Scenario C |
| `fig_montecarlo_combined.pdf` | Fig. 6 — Monte Carlo |
| `summary_tables.txt` | Tables I–II numeric summary |
| `montecarlo_summary.txt` | Table III + solver timing |

## Algorithm mapping

### Step 1 — Confidence (Proposition 1)

```python
e_k = ||x_k - x_hat_{k|k-1}||
y_k = 1 - sat(e_k / e_norm)
c_{k+1} = sat((1-β) c_k + α y_k)
```

Implemented in `caspc/confidence.py` → `ConfidenceEstimator.step()`.

### Step 2 — Risk modulation (Eqs. 5–7)

```python
δ(c) = δ_min + (δ_max - δ_min)(1 - exp(-λ c))
σ_eff(c) = σ_0 (1 + κ(1-c))
ρ(c) = σ_eff(c) sqrt((1-δ(c))/δ(c))
```

Implemented in `caspc/risk.py` → `RiskModulator.evaluate()`.

### Step 3 — MPC (Algorithm 1)

At each step, solve convex QP:

- **Dynamics:** double integrator (Scenario A) or longitudinal (B, C)
- **Safety:** `h(x,u) ≥ d_min + ρ(c_k)` (Cantelli tightening)
- **Cost:** quadratic tracking + input penalty

Solver: OSQP with SCS fallback (matches paper’s runtime discussion).

### Step 4 — Multi-agent (Scenario C)

```python
δ̃_k^(i) = δ_total · δ(c_k^(i)) / Σ_j δ(c_k^(j))
```

Implemented in `caspc/allocation.py`.

## Parameters (match paper Sec. V)

| Parameter | Value | Scenario |
|-----------|-------|----------|
| α, β | 0.30, 0.33 | All |
| e_norm | 0.5 m | All |
| δ_min, δ_max | 0.01, 0.20 | All |
| λ | 3 | All |
| σ_0, κ | 1.0 m, 0.6 | All |
| Fixed baseline δ | 0.05 | A, B |
| δ_total | 0.10 | C |
| N, T_s | 10, 0.2 s | All |

## Controllers compared

| Mode | Description |
|------|-------------|
| `adaptive` | CASPC (Algorithm 1), c_k = ĉ_k |
| `fixed` | Constant δ = 0.05 |
| `worst_case` | Constant δ = δ_min (ρ_max) |
| `naive_reactive` | δ from raw error, no filtering |
| `fixed_split` | Equal δ/2 per agent (Scenario C) |

## Expected results (paper targets)

**Scenario A:** mean back-off ~2.56 m (adaptive) vs ~4.36 m (fixed), ~41% reduction, 0 violations.

**Scenario B:** ~48% back-off reduction, 0 violations.

**Scenario C:** reallocation under fixed δ_total = 0.10; higher back-off on stressed agent, lower on calm agent.

**Monte Carlo (moderate):** adaptive ~2.55 m mean back-off, 0% violations.

**Monte Carlo (severe):** nonzero violations for all methods (honest stress test).

## LaTeX integration

Copy generated PDFs into your paper directory:

```bash
cp outputs/*.pdf /path/to/paper/figures/
```

Figures use the same filenames referenced in your `.tex` file.

## Troubleshooting

- **CVXPY solver errors:** Install `osqp` and `scs`: `pip install osqp scs`
- **Slow Monte Carlo:** Use `--quick` or reduce `--mc-trials 20`
- **Different numbers vs paper:** Small differences are expected from QP weights and leader model details; tune `MPCWeights` in `caspc/mpc.py` if needed
