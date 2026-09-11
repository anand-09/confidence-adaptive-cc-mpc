"""Generate all paper figures and summary tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import IEEE_COL_IN, apply_ieee_style, save_fig
from scenarios.monte_carlo import MonteCarloConfig, run_monte_carlo
from scenarios.scenario_a import ScenarioAConfig, run_scenario_a
from scenarios.scenario_b import ScenarioBConfig, run_scenario_b
from scenarios.scenario_c import ScenarioCConfig, run_scenario_c, window_backoff_stats


def _style():
    apply_ieee_style()


def fig_confidence(out_dir: Path) -> None:
    log = run_scenario_a("adaptive")
    cfg = ScenarioAConfig()
    t = np.array(log.t)
    fig, ax1 = plt.subplots(figsize=(IEEE_COL_IN, 2.15), layout="tight")
    ax1.plot(t, log.confidence, "b-", lw=1.6, label=r"$c_k=\hat c_k$")
    ax1.set_ylabel("Confidence")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(t, log.e_bar, "r--", lw=1.3, label=r"$\bar e_k$")
    ax2.set_ylabel(r"Normalized error $\bar e_k$")
    ax2.set_ylim(0, 1)
    ax1.axvspan(cfg.leader.brake_start, cfg.leader.brake_end, color="gray", alpha=0.2)
    ax1.legend(loc="lower left")
    ax2.legend(loc="upper right")
    save_fig(fig, out_dir / "fig_confidence.pdf")


def fig_backoff(out_dir: Path) -> None:
    adapt = run_scenario_a("adaptive")
    fixed = run_scenario_a("fixed")
    cfg = ScenarioAConfig()
    t = np.array(adapt.t)
    fig, ax = plt.subplots(figsize=(IEEE_COL_IN, 2.05), layout="constrained")
    ax.plot(t, adapt.rho, "b-", lw=1.6, label=r"Adaptive $\rho(c_k)$")
    ax.plot(t, fixed.rho, "k--", lw=1.3, label=r"Fixed $\delta=0.05$")
    ax.axvspan(cfg.leader.brake_start, cfg.leader.brake_end, color="gray", alpha=0.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Back-off $\rho$ [m]")
    ax.legend()
    save_fig(fig, out_dir / "fig_backoff.pdf")


def fig_traj(out_dir: Path) -> None:
    adapt = run_scenario_a("adaptive")
    fixed = run_scenario_a("fixed")
    cfg = ScenarioAConfig()
    t = np.array(adapt.t)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(IEEE_COL_IN, 3.15), sharex=True, layout="constrained")
    ax1.plot(t, adapt.gaps, "b-", label="Adaptive")
    ax1.plot(t, fixed.gaps, "k--", label="Fixed")
    ax1.axhline(cfg.d_min, color="r", ls=":", lw=1)
    ax1.axvspan(cfg.leader.brake_start, cfg.leader.brake_end, color="gray", alpha=0.2)
    ax1.set_ylabel("Longitudinal gap [m]")
    ax1.legend()
    ax2.plot(t, adapt.extra["lateral"], "b-", label="Adaptive")
    ax2.plot(t, fixed.extra["lateral"], "k--", label="Fixed")
    ax2.axhline(cfg.lane_offset, color="g", ls=":", lw=1)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Lateral position [m]")
    ax2.legend()
    save_fig(fig, out_dir / "fig_traj.pdf")


def fig_pedestrian(out_dir: Path) -> None:
    adapt = run_scenario_b("adaptive")
    fixed = run_scenario_b("fixed")
    cfg = ScenarioBConfig()
    t = np.array(adapt.t)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(IEEE_COL_IN, 3.15), sharex=True, layout="constrained")
    ax1.plot(t, adapt.confidence, "b-", lw=1.6, label=r"$c_k$")
    ax1.plot(t, adapt.e_bar, "r--", label=r"$\bar e_k$")
    ax1.axvline(cfg.reveal_time, color="k", ls=":", lw=1)
    ax1.set_ylabel("Confidence / error")
    ax1.legend()
    ax2.plot(t, adapt.extra["speed"], "b-", label="Adaptive")
    ax2.plot(t, fixed.extra["speed"], "k--", label="Fixed")
    ax2.axvline(cfg.reveal_time, color="k", ls=":", lw=1)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed [m/s]")
    ax2.legend()
    save_fig(fig, out_dir / "fig_pedestrian.pdf")


def fig_intersection(out_dir: Path) -> None:
    adapt = run_scenario_c("adaptive")
    fixed = run_scenario_c("fixed_split")
    cfg = ScenarioCConfig()
    t = np.array(adapt.t)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(IEEE_COL_IN, 4.15), sharex=True, layout="constrained")
    ax1.plot(t, adapt.extra["c1"], "b-", label=r"$c_k^{(1)}$ vehicle")
    ax1.plot(t, adapt.extra["c2"], "r-", label=r"$c_k^{(2)}$ pedestrian")
    ax1.axvspan(cfg.vehicle_no_yield_start, cfg.vehicle_no_yield_end, color="gray", alpha=0.2)
    ax1.axvline(cfg.ped_reveal_time, color="k", ls=":", lw=1)
    ax1.set_ylabel("Confidence")
    ax1.legend()
    ax2.plot(t, adapt.extra["delta1"], "b-", label=r"$\tilde\delta^{(1)}$ adaptive")
    ax2.plot(t, adapt.extra["delta2"], "r-", label=r"$\tilde\delta^{(2)}$ adaptive")
    ax2.plot(t, fixed.extra["delta1"], "b:", label="fixed 0.05")
    ax2.plot(t, fixed.extra["delta2"], "r:", label="fixed 0.05")
    ax2.set_ylabel(r"Risk allocation")
    ax2.legend(ncol=2)
    ax3.plot(t, adapt.extra["gap1"], "b-", label="Adaptive gap veh.")
    ax3.plot(t, fixed.extra["gap1"], "b--", label="Fixed gap veh.")
    ax3.plot(t, adapt.extra["gap2"], "r-", label="Adaptive gap ped.")
    ax3.plot(t, fixed.extra["gap2"], "r--", label="Fixed gap ped.")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Gap [m]")
    ax3.legend(ncol=2)
    save_fig(fig, out_dir / "fig_intersection.pdf")


def fig_montecarlo(out_dir: Path, n_trials: int = 100) -> dict:
    from experiments.monte_carlo_fast import plot_mc_combined, run_monte_carlo

    mod = run_monte_carlo(n_trials=n_trials, severity="moderate")
    sev = run_monte_carlo(n_trials=n_trials, severity="severe", seed_base=2000)
    plot_mc_combined(mod, sev, out_dir / "fig_montecarlo_combined.pdf")

    def _pack(mc):
        return {
            r["method"]: {
                "mean_backoff": r["mean_backoff"],
                "std_backoff": r["std_backoff"],
                "violation_rate": r["viol_rate"] / 100.0,
                "solve_ms_mean": r["solve_mean_ms"],
                "backoffs": mc.results[r["method"]]["mean_backoff"],
            }
            for r in mc.summary
        }

    return {"moderate": _pack(mod), "severe": _pack(sev)}


def summary_tables(out_dir: Path) -> None:
    a_ad = run_scenario_a("adaptive")
    a_fx = run_scenario_a("fixed")
    b_ad = run_scenario_b("adaptive")
    b_fx = run_scenario_b("fixed")
    c_ad = run_scenario_c("adaptive")
    c_fx = run_scenario_c("fixed_split")
    cfg_c = ScenarioCConfig()

    lines = []
    lines.append("Table: Scenarios A-B")
    for name, ad, fx in [("A", a_ad, a_fx), ("B", b_ad, b_fx)]:
        red = 100 * (1 - np.mean(ad.rho) / np.mean(fx.rho))
        lines.append(
            f"Scenario {name}: Fixed min_gap={min(ad.gaps):.2f}/{min(fx.gaps):.2f} (ad/fx), "
            f"mean rho {np.mean(ad.rho):.2f}/{np.mean(fx.rho):.2f} ({red:.0f}% reduction), "
            f"violations {sum(ad.violations)}/{sum(fx.violations)}"
        )

    win_ad = window_backoff_stats(c_ad, cfg_c)
    win_fx = window_backoff_stats(c_fx, cfg_c)
    lines.append("\nTable: Scenario C allocation windows")
    for w in win_ad:
        lines.append(
            f"{w}: adaptive rho1={win_ad[w]['rho1']:.2f}, rho2={win_ad[w]['rho2']:.2f} | "
            f"fixed rho1={win_fx[w]['rho1']:.2f}, rho2={win_fx[w]['rho2']:.2f}"
        )

    (out_dir / "summary_tables.txt").write_text("\n".join(lines))


def generate_all(output_dir: str | Path = "outputs", n_mc_trials: int = 100) -> None:
    _style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_confidence(out)
    fig_backoff(out)
    fig_traj(out)
    fig_pedestrian(out)
    fig_intersection(out)
    summary_tables(out)
    if n_mc_trials <= 0:
        print(f"Scenario figures written to {out.resolve()}")
        return
    mc = fig_montecarlo(out, n_trials=n_mc_trials)

    mc_lines = ["Monte Carlo summary"]
    for regime in ("moderate", "severe"):
        mc_lines.append(f"\n{regime.upper()}:")
        for m, s in mc[regime].items():
            mc_lines.append(
                f"  {m}: backoff={s['mean_backoff']:.2f}+/-{s['std_backoff']:.2f} m, "
                f"viol={100*s['violation_rate']:.0f}%, solve_mean={s['solve_ms_mean']:.1f} ms"
            )
    (out / "montecarlo_summary.txt").write_text("\n".join(mc_lines))
    print(f"All figures and tables written to {out.resolve()}")
