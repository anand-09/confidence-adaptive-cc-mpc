#!/usr/bin/env python3
"""Run all CASPC experiments and generate paper figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.run_experiments import main as run_full_experiments


def main() -> None:
    p = argparse.ArgumentParser(description="CASPC experiment runner")
    p.add_argument("--output", default="outputs", help="Output directory for figures")
    p.add_argument("--mc-trials", type=int, default=100, help="Monte Carlo trials per regime")
    p.add_argument("--quick", action="store_true", help="Fast run: 20 MC trials, skip sensitivity")
    p.add_argument("--legacy", action="store_true", help="Use old plotting-only pipeline")
    args = p.parse_args()

    if args.legacy:
        from plotting.generate_figures import generate_all
        n = 20 if args.quick else args.mc_trials
        generate_all(args.output, n_mc_trials=n)
    else:
        argv = ["--output", args.output, "--mc-trials", str(20 if args.quick else args.mc_trials)]
        if args.quick:
            argv.append("--quick")
        run_full_experiments(argv)


if __name__ == "__main__":
    main()

