"""V4 -- D8 wind-tunnel validation (Uranga et al. 2017, AIAA J 55(11)).

Measured target: 8.2% +/- 0.8% mechanical flow-power saving at the cruise
condition, zero net streamwise force (self-propelled model), equal-nozzle-
area rule; decomposed by Hall et al. 2017 into 5.2% jet + 2.4% surface +
0.6% wake contributions.

Reproduction rule used here (stated per SPEC: per-rule validation, no
global forced match):
  * D8-like equivalent body of revolution at 1:11 tunnel scale, tripped
    boundary layer, tunnel speed, sea-level ISA;
  * the aft propulsors ingest f_Phi ~ 0.40 of the fuselage boundary-layer
    dissipation (Uranga/Hall report ~40%);
  * self-propelled: the propulsor subsystem force equals the drag of the
    rest of the airframe (fuselage carries ~35% of total drag), with the
    FPR solved to meet that force at the given capture;
  * podded reference at the same FPR and net force.

Expected diagnosis: the low-order model resolves the jet and wake terms but
has no fan-suction surface term (identically zero here), so it should fall
short of 8.2% by roughly the measured surface contribution (~2.4 points).

Run:  uv run python validation/validate_d8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
import plotstyle  # noqa: E402

from blipb import BLIComparator, FlightState, Fuselage  # noqa: E402
from blipb.powerbalance import hall2017  # noqa: E402

# --- D8-like tunnel configuration ------------------------------------------
SCALE = 1.0 / 11.0
L_FUS = 36.0 * SCALE  # m, model fuselage length
R_EQ = 2.17 * SCALE  # m, equivalent-area radius of the double-bubble section
V_TUNNEL_MACH = 32.0 / 340.3  # ~32 m/s at sea level
X_TR = 0.05  # tripped
F_PHI = 0.40  # ingested dissipation fraction (Uranga/Hall ~40%)
FUS_DRAG_FRACTION = 0.35  # fuselage share of total model drag

TARGET = 0.082
TARGET_BAND = 0.008


def run() -> pd.DataFrame:
    comp = BLIComparator(
        fuselage=Fuselage(length=L_FUS, radius_max=R_EQ),
        flight=FlightState(mach=V_TUNNEL_MACH, altitude=0.0),
        x_tr_frac=X_TR,
        eta_pol=1.0,  # mechanical flow power basis (Uranga measures P_K)
        k_dist=0.0,
    )
    d_fus = comp.bl.drag
    s_req = d_fus * (1.0 / FUS_DRAG_FRACTION - 1.0)  # tow the rest of the model

    # Solve FPR so the f_Phi = 0.40 capture meets the self-propulsion force
    def force_err(fpr: float) -> float:
        return comp.run_design(f_phi=F_PHI, fpr=fpr).net_force - s_req

    fpr = brentq(force_err, 1.001, 1.5, xtol=1e-9)
    res = comp.run_design(f_phi=F_PHI, fpr=fpr)
    dec = hall2017.decompose(res)

    rows = [
        {"quantity": "tunnel V [m/s]", "value": comp.flight.V},
        {"quantity": "Re_L (model)", "value": comp.flight.reynolds(L_FUS)},
        {"quantity": "fuselage drag [N]", "value": d_fus},
        {"quantity": "solved FPR", "value": fpr},
        {"quantity": "PSC this work", "value": res.psc},
        {"quantity": "PSC measured (Uranga 2017)", "value": TARGET},
        {"quantity": "shortfall [pts]", "value": (TARGET - res.psc) * 100},
        {"quantity": "jet contribution (this work)", "value": dec.jet},
        {"quantity": "jet contribution (Hall 2017)", "value": 0.052},
        {"quantity": "wake contribution (this work)", "value": dec.wake},
        {"quantity": "wake contribution (Hall 2017)", "value": 0.006},
        {"quantity": "surface contribution (this work)", "value": 0.0},
        {"quantity": "surface contribution (Hall 2017)", "value": 0.024},
    ]
    df = pd.DataFrame(rows)
    plotstyle.DATADIR.mkdir(exist_ok=True)
    df.to_csv(plotstyle.DATADIR / "validation_d8.csv", index=False)

    # sensitivity of PSC to the assumed ingestion fraction
    sens = []
    for f in (0.30, 0.35, 0.40, 0.45, 0.50):
        fpr_f = brentq(
            lambda p, ff=f: comp.run_design(f_phi=ff, fpr=p).net_force - s_req,
            1.001,
            1.6,
            xtol=1e-9,
        )
        sens.append({"f_phi": f, "fpr": fpr_f, "psc": comp.run_design(f_phi=f, fpr=fpr_f).psc})
    pd.DataFrame(sens).to_csv(plotstyle.DATADIR / "validation_d8_sensitivity.csv", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    psc = float(df.loc[df.quantity == "PSC this work", "value"].iloc[0])
    lo, hi = TARGET - TARGET_BAND, TARGET + TARGET_BAND
    verdict = "WITHIN" if lo <= psc <= hi else "OUTSIDE"
    print(
        f"\nPSC = {psc:.2%} vs measured {TARGET:.1%} +/- {TARGET_BAND:.1%} -> {verdict} band."
        "\nDiagnosis: the low-order model carries no fan-suction surface term"
        f" (measured at {0.024:.1%}); see paper section 6."
    )
