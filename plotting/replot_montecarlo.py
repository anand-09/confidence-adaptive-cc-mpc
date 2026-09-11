"""Rebuild fig_montecarlo_combined.pdf from saved MC results (no new trials)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.monte_carlo_fast import replot_mc_combined

if __name__ == "__main__":
    replot_mc_combined(ROOT / "outputs" / "mc", ROOT / "outputs" / "fig_montecarlo_combined.pdf")
