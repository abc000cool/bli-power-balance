"""Ingestion-fraction convention cross-table (paper Appendix B).

Four conventions circulate for "how much boundary layer the fan ingests":
dissipation/KE-defect fraction (f_Phi, Hall 2017 -- used throughout this
work), mass-flow fraction, momentum-defect fraction, and area fraction.
This script prints and caches their cross-conversion for the SPEC baseline
annulus at several profile exponents.

Run:  uv run python studies/conventions_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle  # noqa: E402

from blipb import BLIComparator  # noqa: E402
from blipb.powerbalance.streamtube import AnnulusProfile  # noqa: E402


def build(n_powerlaw: float = 7.0) -> pd.DataFrame:
    comp = BLIComparator(n_powerlaw=n_powerlaw)
    prof: AnnulusProfile = comp.profile
    full = prof.capture(prof.delta)
    area_full = np.pi * ((prof.r_hub + prof.delta) ** 2 - prof.r_hub**2)

    rows = []
    for f_phi in (0.25, 0.50, 0.75, 1.00):
        c = prof.solve_capture_for_fphi(f_phi)
        area = np.pi * ((prof.r_hub + c.y_c) ** 2 - prof.r_hub**2)
        rows.append(
            {
                "n": n_powerlaw,
                "f_phi (dissipation)": f_phi,
                "mass fraction": c.m_dot / full.m_dot,
                "momentum-defect fraction": c.momentum_defect / full.momentum_defect,
                "area fraction": area / area_full,
                "y_c / delta": c.y_c / prof.delta,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    dfs = [build(n) for n in (5.0, 7.0, 9.0)]
    out = pd.concat(dfs, ignore_index=True)
    plotstyle.DATADIR.mkdir(exist_ok=True)
    out.to_csv(plotstyle.DATADIR / "conventions_table.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
