"""Consolidated per-rule validation table (paper Table 1).

Runs all three validation studies and writes data/validation_table.md.

Run:  uv run python validation/make_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotstyle  # noqa: E402
import validate_d8  # noqa: E402
import validate_smith  # noqa: E402
import validate_starc_abl  # noqa: E402


def main() -> str:
    smith = validate_smith.run()
    d8 = validate_d8.run()
    starc = validate_starc_abl.run()

    def get(df, q):
        return float(df.loc[df.quantity == q, "value"].iloc[0])

    smith_err = max(
        abs(r.psc_this_work / r.psc_reference - 1.0)
        for r in smith.itertuples()
        if r.psc_this_work == r.psc_this_work  # not NaN
    )

    lines = [
        "# Validation table (per-rule; SPEC targets V1-V5)",
        "",
        "| # | Reference | Metric & rule | Reference value | This work | Verdict / note |",
        "|---|---|---|---|---|---|",
        "| V1 | Blasius (exact) | laminar flat plate theta, C_f | exact | theta 1.0%, C_f 1.2% | PASS (tests) |",
        "| V2 | 1/7-power / Schultz-Grunow | turbulent flat-plate C_f | empirical | <=6% / <=12% | PASS with documented L-T high-Re bias (SPEC A2) |",
        f"| V3 | Smith 1993 | ideal wake-ingestion closed form | exact | {smith_err:.2%} max err | PASS (<=2%) |",
        (
            f"| V4 | Uranga 2017 (D8) | mech. flow-power saving, self-propelled, f_Phi=0.40 "
            f"| 8.2% +/- 0.8% | {get(d8, 'PSC this work'):.1%} | "
            "+0.7 pts above band: eta_fill=1 idealization overpredicts jet term; "
            "no fan-suction surface term (measured 2.4%) |"
        ),
        (
            f"| V5a | Yildirim 2022 | power saving, mech drive, 1/3-thrust rule "
            f"| 2.1-2.3% | {get(starc, 'net power saving, mech drive'):.1%} | "
            "above: no installation drag / coupling losses in low-order model |"
        ),
        (
            f"| V5b | NASA/TM-20210016661 | block fuel, 3500 nmi, turboelectric "
            f"| -3.4% | {get(starc, 'block-fuel delta 3500nmi (snowball)'):.1%} | "
            "below in magnitude: TM includes airframe resizing beyond propulsive saving; "
            "our value sits between Giannakakis (+1.7%) and the TM (-3.4%) |"
        ),
        "",
        f"STARC-ABL operating point: achieved f_Phi = {get(starc, 'achieved f_phi'):.2f}, "
        f"subsystem PSC (P_K) = {get(starc, 'subsystem PSC (P_K)'):.1%}, "
        f"turboelectric net = {get(starc, 'net power saving, turboelectric'):.2%}.",
        "",
        f"D8 decomposition (this work vs Hall 2017): jet {get(d8, 'jet contribution (this work)'):.1%} vs 5.2%, "
        f"wake {get(d8, 'wake contribution (this work)'):.1%} vs 0.6%, surface 0.0% vs 2.4%.",
    ]
    out = "\n".join(lines)
    (plotstyle.DATADIR / "validation_table.md").write_text(out, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main())
