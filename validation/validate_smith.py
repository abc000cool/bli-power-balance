"""V3 -- Smith (1993) analytic-limit validation.

Two independent algebra paths to the ideal wake-ingestion benefit are
compared against the comparator implementation:

(a) the uniform-wake closed form PSC = 2(1-w)/(3-w), evaluated for a range
    of wake depths;
(b) the self-propelled profile closed form (streamtube integrals only, no
    fan model, no force iteration) vs the full comparator run at low Mach
    with eta = 1, zero distortion, full ingestion, FPR solved so the jet
    exactly refills the wake (V_j = V_inf), and the podded power recomputed
    under the closed form's equal-mass-flow rule.

Target: agreement within 2% (SPEC target V3).

Run:  uv run python validation/validate_smith.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
import plotstyle  # noqa: E402

from blipb import BLIComparator, FlightState  # noqa: E402
from blipb.powerbalance.smith1993 import (  # noqa: E402
    psc_self_propelled_profile,
    psc_uniform_wake_full_ingestion,
)


def run() -> pd.DataFrame:
    rows = []

    # (a) uniform-wake closed form sanity spread
    for w in (0.6, 0.7, 0.8, 0.9):
        rows.append(
            {
                "case": f"uniform wake w={w}",
                "psc_reference": psc_uniform_wake_full_ingestion(w),
                "psc_this_work": np.nan,
                "note": "closed form (exact by construction)",
            }
        )

    # (b) comparator vs profile closed form, several power-law exponents
    for n in (5.0, 7.0, 9.0):
        comp = BLIComparator(
            flight=FlightState(mach=0.20, altitude=1_000.0),
            eta_pol=1.0,
            k_dist=0.0,
            n_powerlaw=n,
        )
        v = comp.cv.v_inf

        fpr_sp = brentq(
            lambda fpr: comp.run_design(f_phi=1.0, fpr=fpr).v_jet_bli - v,
            1.001,
            1.2,
            xtol=1e-10,
        )
        res = comp.run_design(f_phi=1.0, fpr=fpr_sp)
        full = comp.profile.capture(comp.profile.delta)
        p_pod_eq = v * full.momentum_defect + full.momentum_defect**2 / (2 * full.m_dot)
        psc_this = 1.0 - res.pk_bli / p_pod_eq
        psc_ref = psc_self_propelled_profile(comp.profile)
        rows.append(
            {
                "case": f"self-propelled profile n={n:.0f}",
                "psc_reference": psc_ref,
                "psc_this_work": psc_this,
                "note": f"rel. err {abs(psc_this / psc_ref - 1):.2%}",
            }
        )

    df = pd.DataFrame(rows)
    plotstyle.DATADIR.mkdir(exist_ok=True)
    df.to_csv(plotstyle.DATADIR / "validation_smith.csv", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    errs = [
        abs(r.psc_this_work / r.psc_reference - 1.0)
        for r in df.itertuples()
        if np.isfinite(r.psc_this_work)
    ]
    print(f"\nmax relative error vs Smith closed form: {max(errs):.2%} (target <= 2%)")
