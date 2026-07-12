"""V5 -- STARC-ABL bracket validation.

The STARC-ABL benefit is not a scalar: published values measure different
quantities under different rules (SPEC target V5 requires a bracket, not a
point match):

  * Yildirim et al. 2022 (J. Aircraft): 2.1-2.3% *power saving* from coupled
    RANS+NPSS, isolated-propulsor comparison -- compare with our
    aircraft-level net power saving at eta_elec = 1 (their bookkeeping has
    no electrical chain in the delta).
  * NASA/TM-20210016661 (Felder et al. 2022): 3.4% design-mission /
    2.7% economic-mission *block fuel* -- compare with our Breguet delta
    including the electrical chain and snowball.
  * Welstead & Felder 2016: 12% -- obsolete (Rev A, pre-M0.785); cited for
    history only.

Rule: cruise M0.785 / FL350, aft fan carries 1/3 of total cruise thrust
(D_total = W g / (L/D)), FPR = 1.25, force-driven capture.

Run:  uv run python validation/validate_starc_abl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
import plotstyle  # noqa: E402

from blipb import BLIComparator  # noqa: E402
from blipb.atmosphere import G0  # noqa: E402
from blipb.mission.breguet import NMI, delta_block_fuel  # noqa: E402
from blipb.uq.model import DEFAULT_CONFIG  # noqa: E402

FPR = 1.25
THRUST_SHARE = 1.0 / 3.0


def run() -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    comp = BLIComparator(
        fuselage=cfg.fuselage(), flight=cfg.flight(), eta_pol=0.92, k_dist=0.03
    )
    d_total = cfg.w_initial * G0 / cfg.lift_drag
    f_aft = THRUST_SHARE * d_total
    s_req = f_aft - comp.bl.drag

    res = comp.run_force(s_req=s_req, fpr=FPR)

    p_total_ref = d_total * cfg.flight().V / cfg.eta_prop_ref

    def net(eta_elec: float) -> float:
        return (res.p_shaft_pod - res.p_shaft_bli / eta_elec) / p_total_ref

    psc_net_mech = net(1.0)  # mechanical drive (Yildirim-comparable)
    psc_net_elec = net(0.92)  # turboelectric chain

    fuel_kwargs = dict(
        v=cfg.flight().V,
        lift_drag=cfg.lift_drag,
        tsfc=cfg.tsfc,
        w_initial=cfg.w_initial,
    )
    d_fuel_design = delta_block_fuel(
        psc_net_elec, range_m=3500 * NMI, snowball=1.35, **fuel_kwargs
    )
    d_fuel_no_snowball = delta_block_fuel(
        psc_net_elec, range_m=3500 * NMI, snowball=1.0, **fuel_kwargs
    )

    rows = [
        {"quantity": "D_total [kN]", "value": d_total / 1e3},
        {"quantity": "D_fuselage [kN]", "value": comp.bl.drag / 1e3},
        {"quantity": "aft-fan thrust [kN]", "value": f_aft / 1e3},
        {"quantity": "achieved f_phi", "value": res.f_phi},
        {"quantity": "capture / delta", "value": res.y_capture / res.delta_bl},
        {"quantity": "subsystem PSC (P_K)", "value": res.psc},
        {"quantity": "subsystem PSC (shaft)", "value": res.psc_shaft},
        {"quantity": "net power saving, mech drive", "value": psc_net_mech},
        {"quantity": "Yildirim 2022 power saving", "value": 0.022},
        {"quantity": "net power saving, turboelectric", "value": psc_net_elec},
        {"quantity": "block-fuel delta 3500nmi (snowball)", "value": d_fuel_design},
        {"quantity": "block-fuel delta 3500nmi (no snowball)", "value": d_fuel_no_snowball},
        {"quantity": "NASA TM-20210016661 design-mission", "value": -0.034},
        {"quantity": "Welstead-Felder 2016 (obsolete)", "value": -0.12},
    ]
    df = pd.DataFrame(rows)
    plotstyle.DATADIR.mkdir(exist_ok=True)
    df.to_csv(plotstyle.DATADIR / "validation_starc_abl.csv", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    mech = float(df.loc[df.quantity == "net power saving, mech drive", "value"].iloc[0])
    fuel = float(
        df.loc[df.quantity == "block-fuel delta 3500nmi (snowball)", "value"].iloc[0]
    )
    print(
        f"\nBracket check: mech-drive power saving {mech:.2%} vs Yildirim 2.1-2.3%;"
        f"\n               block fuel {fuel:.2%} vs NASA TM -3.4% (design mission)."
    )
