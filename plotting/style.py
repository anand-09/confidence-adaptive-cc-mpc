"""IEEE conference figure style (ieeeconf, 10 pt body, caption font=small = 9 pt).

Figures are drawn at column width so \\includegraphics[width=\\linewidth]
does not scale the type down relative to the caption.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Typical ieeeconf / IEEEtran column width (letter, two-column).
IEEE_COL_IN = 3.40


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.borderpad": 0.25,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.4,
            "legend.labelspacing": 0.25,
            "legend.columnspacing": 0.7,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.4,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "axes.labelpad": 2.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
