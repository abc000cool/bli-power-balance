# Validation table (per-rule; SPEC targets V1-V5)

| # | Reference | Metric & rule | Reference value | This work | Verdict / note |
|---|---|---|---|---|---|
| V1 | Blasius (exact) | laminar flat plate theta, C_f | exact | theta 1.0%, C_f 1.2% | PASS (tests) |
| V2 | 1/7-power / Schultz-Grunow | turbulent flat-plate C_f | empirical | <=6% / <=12% | PASS with documented L-T high-Re bias (SPEC A2) |
| V3 | Smith 1993 | ideal wake-ingestion closed form | exact | 0.00% max err | PASS (<=2%) |
| V4 | Uranga 2017 (D8) | mech. flow-power saving, self-propelled, f_Phi=0.40 | 8.2% +/- 0.8% | 9.7% | +0.7 pts above band: eta_fill=1 idealization overpredicts jet term; no fan-suction surface term (measured 2.4%) |
| V5a | Yildirim 2022 | power saving, mech drive, 1/3-thrust rule | 2.1-2.3% | 4.0% | above: no installation drag / coupling losses in low-order model |
| V5b | NASA/TM-20210016661 | block fuel, 3500 nmi, turboelectric | -3.4% | -1.9% | below in magnitude: TM includes airframe resizing beyond propulsive saving; our value sits between Giannakakis (+1.7%) and the TM (-3.4%) |

STARC-ABL operating point: achieved f_Phi = 0.78, subsystem PSC (P_K) = 13.9%, turboelectric net = 1.53%.

D8 decomposition (this work vs Hall 2017): jet 8.6% vs 5.2%, wake 1.2% vs 0.6%, surface 0.0% vs 2.4%.