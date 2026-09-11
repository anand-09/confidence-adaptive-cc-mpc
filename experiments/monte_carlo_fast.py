"""
Fast Monte Carlo evaluation for Scenario A (lane-change + braking cut-in).

Reviewer additions addressed:
  1. Statistical evaluation over randomized trials.
  2. Baseline spectrum: fixed delta=0.05, worst-case fixed, naive-reactive,
     confidence-adaptive (Proposition 1 MAP estimator).
  3. Solver runtime statistics per QP solve.

Uses a single pre-built CVXPY problem with cp.Parameter placeholders so
100 trials x 4 methods x ~50 steps completes in minutes, not hours.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Nominal parameters (paper Sec. V)
# ---------------------------------------------------------------------------
from plotting.style import IEEE_COL_IN, apply_ieee_style, save_fig
Ts = 0.2
T_sim = 10.0
N_sim = int(T_sim / Ts)
N = 10

gap0 = 22.0
v2_0 = 22.0
vx0 = 20.0
d_min = 5.0

alpha_c, beta_c = 0.30, 0.33
e_norm = 0.5
delta_min, delta_max = 0.01, 0.20
lam = 3.0
sigma0, kappa = 1.0, 0.6

REGIMES = {
    "moderate": {
        "a_brake_range": (2.0, 3.5),
        "t_start_range": (3.5, 4.5),
        "t_dur_range": (1.5, 2.5),
    },
    "severe": {
        "a_brake_range": (2.0, 5.0),
        "t_start_range": (3.0, 5.0),
        "t_dur_range": (1.0, 3.0),
    },
}

METHODS = ["fixed", "worstcase", "naive", "adaptive"]
METHOD_LABELS = {
    "fixed": "Fixed\n$\\delta=0.05$",
    "worstcase": "Worst-case\nfixed",
    "naive": "Naive\nreactive",
    "adaptive": "Confidence\nadaptive",
}
METHOD_COLORS = ["#e08283", "#c0392b", "#f0c987", "#1f4e79"]


def delta_of_c(c: float, lam_v: float = lam) -> float:
    return delta_min + (delta_max - delta_min) * (1 - np.exp(-lam_v * c))


def sigma_eff(c: float, kappa_v: float = kappa) -> float:
    return sigma0 * (1 + kappa_v * (1 - c))


def backoff(c: float, lam_v: float = lam, kappa_v: float = kappa) -> float:
    d = delta_of_c(c, lam_v)
    return sigma_eff(c, kappa_v) * np.sqrt((1 - d) / d)


backoff_fixed = sigma0 * np.sqrt((1 - 0.05) / 0.05)
backoff_worstcase = sigma0 * np.sqrt((1 - delta_min) / delta_min)

A = np.array([[1, Ts, 0, 0], [0, 1, 0, 0], [0, 0, 1, Ts], [0, 0, 0, 1]])
B = np.array([[0.5 * Ts**2, 0], [Ts, 0], [0, 0.5 * Ts**2], [0, Ts]])
ax_max, ay_max, vy_max = 2.5, 2.0, 3.0
y0, y_ref = 0.0, 3.6

# Pre-built QP (built once, solved many times)
_Z = cp.Variable((4, N + 1))
_U = cp.Variable((2, N))
_slack = cp.Variable(N, nonneg=True)
_z0_p = cp.Parameter(4)
_veh2_p = cp.Parameter(N)
_backoff_p = cp.Parameter(nonneg=True)

_cost = 0
_constr = [_Z[:, 0] == _z0_p]
for k in range(N):
    _constr += [_Z[:, k + 1] == A @ _Z[:, k] + B @ _U[:, k]]
    _constr += [cp.abs(_U[0, k]) <= ax_max, cp.abs(_U[1, k]) <= ay_max]
    _constr += [cp.abs(_Z[3, k]) <= vy_max]
    _constr += [_veh2_p[k] - _Z[0, k + 1] >= d_min + _backoff_p - _slack[k]]
    _cost += 6.0 * (_Z[2, k + 1] - y_ref) ** 2 + 0.05 * cp.sum_squares(_U[:, k]) + 0.3 * _Z[3, k + 1] ** 2
_cost += 40.0 * (_Z[2, N] - y_ref) ** 2 + 5.0 * _Z[3, N] ** 2 + 2.0e4 * cp.sum(_slack)
_prob = cp.Problem(cp.Minimize(_cost), _constr)


def solve_mpc(z0, veh2_pred_positions, backoff_val, timing):
    _z0_p.value = z0
    _veh2_p.value = veh2_pred_positions
    _backoff_p.value = backoff_val
    t0 = time.perf_counter()
    _prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if _prob.status not in ("optimal", "optimal_inaccurate"):
        _prob.solve(solver=cp.SCS, verbose=False)
    timing.append(time.perf_counter() - t0)
    u0 = _U.value[:, 0] if _U.value is not None else np.zeros(2)
    slack0 = _slack.value[0] if _slack.value is not None else 0.0
    return u0, slack0


@dataclass
class TrialParams:
    alpha: float = alpha_c
    beta: float = beta_c
    lam: float = lam
    kappa: float = kappa


def run_trial(
    method: str,
    rng: np.random.Generator,
    a_brake_mag: float,
    t_brake_start: float,
    t_brake_dur: float,
    meas_noise_std: float,
    params: TrialParams | None = None,
) -> dict:
    p = params or TrialParams()
    v2_final = v2_0 - a_brake_mag * t_brake_dur
    t_brake_end = t_brake_start + t_brake_dur

    def veh2_true_state(t):
        if t <= t_brake_start:
            return v2_0
        if t <= t_brake_end:
            return v2_0 - a_brake_mag * (t - t_brake_start)
        return max(v2_final, 5.0)

    xs, vs, x = [], [], gap0
    for k in range(N_sim + N + 1):
        t = k * Ts
        v = veh2_true_state(t)
        xs.append(x)
        vs.append(v)
        x += v * Ts
    veh2_x_true = np.array(xs)
    veh2_x_meas = veh2_x_true + rng.normal(0, meas_noise_std, size=veh2_x_true.shape)

    z = np.array([0.0, vx0, y0, 0.0])
    c = 0.85
    timing = []
    min_gap = np.inf
    backoffs = []
    veh2_pred_from_prev = None

    for k in range(N_sim):
        actual_pos = veh2_x_meas[k]
        if veh2_pred_from_prev is not None:
            e_raw = abs(actual_pos - veh2_pred_from_prev)
            e = min(e_raw / e_norm, 1.0)
        else:
            e = 0.0
        y_k = 1 - e

        if method == "adaptive":
            c = np.clip((1 - p.beta) * c + p.alpha * y_k, 0.0, 1.0)
            b_k = backoff(c, p.lam, p.kappa)
        elif method == "naive":
            d_naive = delta_min + (delta_max - delta_min) * (1 - e)
            b_k = sigma0 * np.sqrt((1 - d_naive) / d_naive)
        elif method == "worstcase":
            b_k = backoff_worstcase
        else:
            b_k = backoff_fixed

        backoffs.append(b_k)
        v2_last = (veh2_x_meas[k] - veh2_x_meas[k - 1]) / Ts if k > 0 else v2_0
        veh2_pred_positions = np.array([actual_pos + v2_last * Ts * (j + 1) for j in range(N)])
        veh2_pred_from_prev = veh2_pred_positions[0]

        u0, _ = solve_mpc(z, veh2_pred_positions, b_k, timing)
        z = A @ z + B @ u0
        gap = actual_pos - z[0]
        min_gap = min(min_gap, gap)

    return {
        "min_gap": min_gap,
        "mean_backoff": float(np.mean(backoffs)),
        "violation": min_gap < d_min,
        "solve_times": timing,
    }


@dataclass
class MCResults:
    severity: str
    n_trials: int
    results: dict = field(default_factory=dict)
    summary: list = field(default_factory=list)
    wall_time_s: float = 0.0


def run_monte_carlo(
    n_trials: int = 100,
    severity: str = "moderate",
    seed_base: int = 1000,
    params: TrialParams | None = None,
) -> MCResults:
    regime = REGIMES[severity]
    results = {m: {"min_gap": [], "mean_backoff": [], "violation": [], "solve_times": []} for m in METHODS}

    t_start = time.perf_counter()
    for trial in range(n_trials):
        rng = np.random.default_rng(seed_base + trial)
        a_brake = rng.uniform(*regime["a_brake_range"])
        t_start_br = rng.uniform(*regime["t_start_range"])
        t_dur = rng.uniform(*regime["t_dur_range"])
        noise = rng.uniform(0.03, 0.08)
        for m in METHODS:
            rng_run = np.random.default_rng(seed_base + trial)
            res = run_trial(m, rng_run, a_brake, t_start_br, t_dur, noise, params)
            results[m]["min_gap"].append(res["min_gap"])
            results[m]["mean_backoff"].append(res["mean_backoff"])
            results[m]["violation"].append(res["violation"])
            results[m]["solve_times"].extend(res["solve_times"])

    summary = []
    for m in METHODS:
        mg = np.array(results[m]["min_gap"])
        mb = np.array(results[m]["mean_backoff"])
        vio = np.array(results[m]["violation"])
        st = np.array(results[m]["solve_times"]) * 1000
        summary.append(
            {
                "method": m,
                "mean_backoff": float(mb.mean()),
                "std_backoff": float(mb.std()),
                "mean_min_gap": float(mg.mean()),
                "viol_rate": float(100 * vio.mean()),
                "solve_mean_ms": float(st.mean()),
                "solve_median_ms": float(np.median(st)),
                "solve_p95_ms": float(np.percentile(st, 95)),
                "solve_p99_ms": float(np.percentile(st, 99)),
                "solve_max_ms": float(st.max()),
                "solve_over_50ms_pct": float(100 * np.mean(st > 50)),
            }
        )

    return MCResults(
        severity=severity,
        n_trials=n_trials,
        results=results,
        summary=summary,
        wall_time_s=time.perf_counter() - t_start,
    )


def save_mc_results(mc: MCResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"mc_results_{mc.severity}.npz",
        **{f"{m}_backoff": np.array(mc.results[m]["mean_backoff"]) for m in METHODS},
        **{f"{m}_mingap": np.array(mc.results[m]["min_gap"]) for m in METHODS},
        **{f"{m}_viol": np.array(mc.results[m]["violation"]) for m in METHODS},
        **{f"{m}_solve": np.array(mc.results[m]["solve_times"]) for m in METHODS},
    )
    with open(out_dir / f"mc_summary_{mc.severity}.json", "w") as f:
        json.dump(mc.summary, f, indent=2)


def plot_mc_boxplot(mc: MCResults, out_path: Path) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_COL_IN, 2.3), layout="constrained")
    data = [mc.results[m]["mean_backoff"] for m in METHODS]
    labels = [METHOD_LABELS[m] for m in METHODS]
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, widths=0.55, patch_artist=True)
    for patch, col in zip(bp["boxes"], METHOD_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    ax.set_ylabel("Mean back-off per trial [m]")
    save_fig(fig, out_path)


def load_mc_results(out_dir: Path, severity: str) -> MCResults:
    """Reload a saved Monte Carlo run (npz + json) so the figure can be replotted."""
    out_dir = Path(out_dir)
    data = np.load(out_dir / f"mc_results_{severity}.npz")
    with open(out_dir / f"mc_summary_{severity}.json") as f:
        summary = json.load(f)
    results = {
        m: {
            "mean_backoff": data[f"{m}_backoff"].tolist(),
            "min_gap": data[f"{m}_mingap"].tolist(),
            "violation": data[f"{m}_viol"].tolist(),
            "solve_times": data[f"{m}_solve"].tolist(),
        }
        for m in METHODS
    }
    n_trials = len(results[METHODS[0]]["mean_backoff"])
    return MCResults(
        severity=severity,
        n_trials=n_trials,
        results=results,
        summary=summary,
        wall_time_s=0.0,
    )


def plot_mc_combined(mod: MCResults, sev: MCResults, out_path: Path) -> None:
    """Paper Fig. 6: two-panel layout, 14 pt labels / 12 pt ticks / 12 pt legend."""
    apply_ieee_style()
    fs, tick, leg = 14, 12, 12
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    labels_flat = [
        "Fixed\n$\\delta{=}0.05$",
        "Worst-case\nfixed",
        "Naive\nreactive",
        "Confidence\nadaptive",
    ]

    data = [mod.results[m]["mean_backoff"] for m in METHODS]
    bp = ax1.boxplot(data, tick_labels=labels_flat, showmeans=True, widths=0.55, patch_artist=True)
    for patch, col in zip(bp["boxes"], METHOD_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    ax1.set_ylabel("Mean back-off per trial [m]", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=tick)

    x = np.arange(len(METHODS))
    w = 0.35
    viol_mod = [s["viol_rate"] for s in mod.summary]
    viol_sev = [s["viol_rate"] for s in sev.summary]
    ax2.bar(x - w / 2, viol_mod, w, label="Moderate severity", color="#5b9bd5")
    ax2.bar(x + w / 2, viol_sev, w, label="Severe (stress test)", color="#c0392b")
    ax2.set_xticks(x, labels_flat)
    ax2.set_ylabel("Violation rate [%]", fontsize=fs)
    ax2.legend(fontsize=leg, loc="upper left", framealpha=0.95)
    ax2.tick_params(axis="both", labelsize=tick)

    for ax in (ax1, ax2):
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def replot_mc_combined(mc_dir: str | Path, out_path: str | Path) -> None:
    """Rebuild fig_montecarlo_combined.pdf from saved trials (no new Monte Carlo)."""
    mc_dir = Path(mc_dir)
    mod = load_mc_results(mc_dir, "moderate")
    sev = load_mc_results(mc_dir, "severe")
    plot_mc_combined(mod, sev, Path(out_path))
    print(f"Wrote {Path(out_path).resolve()}")


def runtime_latex_table(mod: MCResults, sev: MCResults) -> str:
    """LaTeX table for solver runtime (reviewer request)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{QP solver runtime over all Monte Carlo solves ($T_s=200$\,ms control period).}",
        r"\label{tab:runtime}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Mean [ms] & Median [ms] & 95th [ms] & Max [ms] \\",
        r"\midrule",
    ]
    # Pool all solve times across both regimes for the adaptive method row as representative
    for mc in (mod,):
        for row in mc.summary:
            name = row["method"].replace("worstcase", "Worst-case").replace("adaptive", "Adaptive").replace("naive", "Naive").replace("fixed", "Fixed")
            lines.append(
                f"{name} & {row['solve_mean_ms']:.1f} & {row['solve_median_ms']:.1f} & "
                f"{row['solve_p95_ms']:.1f} & {row['solve_max_ms']:.1f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def mc_results_latex_table(mod: MCResults, sev: MCResults) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Monte Carlo results, $100$ trials per cell.}",
        r"\label{tab:montecarlo}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Moderate severity} & \multicolumn{2}{c}{Severe (stress test)} \\",
        r"Method & B.o.\ [m] & Viol.\ & B.o.\ [m] & Viol.\ \\",
        r"\midrule",
    ]
    mod_map = {r["method"]: r for r in mod.summary}
    sev_map = {r["method"]: r for r in sev.summary}
    names = {"fixed": "Fixed $\\delta=0.05$", "worstcase": "Worst-case fixed", "naive": "Naive reactive", "adaptive": "Confidence-adaptive"}
    for m in METHODS:
        a, b = mod_map[m], sev_map[m]
        lines.append(
            f"{names[m]} & ${a['mean_backoff']:.2f}\\pm{a['std_backoff']:.2f}$ & ${a['viol_rate']:.0f}\\%$ & "
            f"${b['mean_backoff']:.2f}\\pm{b['std_backoff']:.2f}$ & ${b['viol_rate']:.0f}\\%$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def print_summary(mc: MCResults) -> None:
    print(f"\n=== Monte Carlo summary [{mc.severity}, {mc.n_trials} trials, {mc.wall_time_s:.1f}s] ===")
    for row in mc.summary:
        print(
            f"{row['method']:10s}  backoff={row['mean_backoff']:.2f}+-{row['std_backoff']:.2f} m  "
            f"min_gap={row['mean_min_gap']:.2f} m  viol={row['viol_rate']:.1f}%  "
            f"solve: mean={row['solve_mean_ms']:.1f}ms p95={row['solve_p95_ms']:.1f}ms max={row['solve_max_ms']:.1f}ms"
        )
