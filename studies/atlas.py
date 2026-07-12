"""Parametric benefit atlas: sweeps cached to Parquet (paper section 4).

Products (all under data/):
  atlas_fphi_fpr.parquet       41x41 grid: subsystem PSC, net saving, validity
  atlas_saturation.parquet     1-D f_Phi slices at three FPRs: absolute and
                               normalized saving (diminishing returns)
  atlas_design_curves.parquet  force-driven design curves: FPR sweep at three
                               thrust shares -> (f_phi, PSC) pairs
  atlas_mach_alt.parquet       Mach x altitude sensitivity of the baseline point

Run:  uv run python studies/atlas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle  # noqa: E402

from blipb import BLIComparator, FlightState, Fuselage  # noqa: E402
from blipb.atmosphere import G0  # noqa: E402
from blipb.uq.model import DEFAULT_CONFIG  # noqa: E402

CFG = DEFAULT_CONFIG


def _net_saving(res, eta_elec: float = 0.92) -> float:
    """Aircraft-level net power saving (fraction of total shaft power)."""
    d_total = CFG.w_initial * G0 / CFG.lift_drag
    p_ref = d_total * CFG.flight().V / CFG.eta_prop_ref
    return (res.p_shaft_pod - res.p_shaft_bli / eta_elec) / p_ref


def sweep_fphi_fpr(n: int = 41) -> pd.DataFrame:
    comp = BLIComparator(fuselage=CFG.fuselage(), flight=CFG.flight())
    rows = []
    for f in np.linspace(0.05, 0.95, n):
        for fpr in np.linspace(1.20, 1.50, n):
            try:
                r = comp.run_design(f_phi=f, fpr=fpr)
                rows.append(
                    {
                        "f_phi": f,
                        "fpr": fpr,
                        "psc": r.psc,
                        "psc_shaft": r.psc_shaft,
                        "saving_w": r.pk_pod - r.pk_bli,
                        "net_saving": _net_saving(r),
                        "valid": r.valid,
                    }
                )
            except (ValueError, AssertionError):
                rows.append(
                    {
                        "f_phi": f,
                        "fpr": fpr,
                        "psc": np.nan,
                        "psc_shaft": np.nan,
                        "saving_w": np.nan,
                        "net_saving": np.nan,
                        "valid": False,
                    }
                )
    return pd.DataFrame(rows)


def sweep_saturation(n: int = 201) -> pd.DataFrame:
    comp = BLIComparator(fuselage=CFG.fuselage(), flight=CFG.flight())
    rows = []
    for fpr in (1.20, 1.30, 1.45):
        ref = comp.run_design(f_phi=0.95, fpr=fpr)
        ref_saving = ref.pk_pod - ref.pk_bli
        for f in np.linspace(0.02, 0.95, n):
            r = comp.run_design(f_phi=f, fpr=fpr)
            rows.append(
                {
                    "fpr": fpr,
                    "f_phi": f,
                    "saving_w": r.pk_pod - r.pk_bli,
                    "saving_norm": (r.pk_pod - r.pk_bli) / ref_saving,
                    "psc": r.psc,
                    "net_saving": _net_saving(r),
                }
            )
    return pd.DataFrame(rows)


def sweep_design_curves() -> pd.DataFrame:
    """Force-driven curves: at fixed thrust share, sweep FPR -> capture."""
    comp = BLIComparator(fuselage=CFG.fuselage(), flight=CFG.flight())
    d_total = CFG.w_initial * G0 / CFG.lift_drag
    rows = []
    for share in (0.25, 1.0 / 3.0, 0.45):
        s_req = share * d_total - comp.bl.drag
        for fpr in np.linspace(1.16, 1.50, 60):
            try:
                r = comp.run_force(s_req=s_req, fpr=fpr)
            except ValueError:
                continue
            rows.append(
                {
                    "thrust_share": share,
                    "fpr": fpr,
                    "f_phi": r.f_phi,
                    "psc": r.psc,
                    "net_saving": _net_saving(r),
                    "capture_ratio": r.y_capture / r.delta_bl,
                }
            )
    return pd.DataFrame(rows)


def sweep_mach_alt() -> pd.DataFrame:
    rows = []
    for mach in np.linspace(0.72, 0.82, 11):
        for alt in (9_144.0, 10_668.0, 12_192.0):  # FL300/350/400
            comp = BLIComparator(
                fuselage=CFG.fuselage(), flight=FlightState(mach=mach, altitude=alt)
            )
            r = comp.run_design(f_phi=0.5, fpr=1.25)
            rows.append(
                {
                    "mach": mach,
                    "altitude": alt,
                    "psc": r.psc,
                    "drag_fus": r.drag_fuselage,
                    "valid": r.valid,
                }
            )
    return pd.DataFrame(rows)


def sweep_fineness() -> pd.DataFrame:
    """Fuselage slenderness effect at fixed length (varying radius)."""
    rows = []
    for radius in np.linspace(1.4, 2.6, 13):
        comp = BLIComparator(
            fuselage=Fuselage(length=CFG.fuselage_length, radius_max=radius),
            flight=CFG.flight(),
        )
        r = comp.run_design(f_phi=0.5, fpr=1.25)
        rows.append(
            {
                "radius": radius,
                "fineness": CFG.fuselage_length / (2 * radius),
                "psc": r.psc,
                "drag_fus": r.drag_fuselage,
                "valid": r.valid,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    plotstyle.DATADIR.mkdir(exist_ok=True)
    jobs = {
        "atlas_fphi_fpr.parquet": sweep_fphi_fpr,
        "atlas_saturation.parquet": sweep_saturation,
        "atlas_design_curves.parquet": sweep_design_curves,
        "atlas_mach_alt.parquet": sweep_mach_alt,
        "atlas_fineness.parquet": sweep_fineness,
    }
    for fname, fn in jobs.items():
        df = fn()
        df.to_parquet(plotstyle.DATADIR / fname, index=False)
        n_bad = int((~df.get("valid", pd.Series(True, index=df.index))).sum())
        print(f"{fname}: {len(df)} rows written ({n_bad} invalid cells flagged)")
