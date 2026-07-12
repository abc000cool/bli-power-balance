"""Shared publication plot style for all figures (AIAA-friendly).

Categorical colors: Okabe-Ito colorblind-safe palette in fixed order
(blue, vermillion, green, orange, sky, pink).  Sequential fields: viridis
(perceptually uniform, CVD-safe).  Identity is never encoded by color alone:
line styles and markers differ per series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# Okabe-Ito, fixed assignment order
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_PINK = "#CC79A7"
CYCLE = [C_BLUE, C_VERM, C_GREEN, C_ORANGE, C_SKY, C_PINK]

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
DATADIR = Path(__file__).resolve().parents[1] / "data"


def apply() -> None:
    FIGDIR.mkdir(exist_ok=True)
    DATADIR.mkdir(exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9.5,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.prop_cycle": mpl.cycler(color=CYCLE),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )
