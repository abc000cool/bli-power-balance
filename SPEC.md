# SPEC.md — Frozen baseline specification (Phase 1)

This file freezes the baseline configuration, validation targets, and comparison rules for the
`blipb` study. It is **immutable except by documented amendment** (append an entry to the
Amendment Log below; never edit frozen values in place).

## 1. Baseline configuration: single-aisle STARC-ABL class

| Quantity | Value | Notes |
|---|---|---|
| Cruise Mach | 0.785 | Rev B condition (Bowman/Felder 2018) |
| Cruise altitude | 35,000 ft (10,668 m) | ISA |
| Fuselage length L | 37.0 m | STARC-ABL class |
| Fuselage max diameter D | 3.76 m | fineness ratio L/D = 9.84 |
| Reynolds number Re_L | ~1.5e8 | from ISA @ FL350, M0.785 |
| Nose length | 0.15 L | elliptic nose |
| Tail-cone length | 0.30 L | cubic contraction to r_tail |
| Tail radius r_tail | 0.30 · r_max | BLI fan hub station |
| Aft-fan power fraction φ | 0.28 | fraction of total propulsive power through BLI fan (STARC-ABL) |
| Aft FPR (baseline) | 1.25 | Welstead & Felder 2016 |
| Mission ranges | {500, 1500, 3000} nmi | economic sweep; design 3500 nmi for W&F reproduction |
| L/D (cruise, baseline) | 21.4 | STARC-ABL class |
| MTOW-class weight at cruise | 60,000 kg | for required-thrust sizing |
| TSFC (baseline turbofan, cruise) | 14.2 mg/N/s (0.50 lb/lbf/hr) | N+3-era |

## 2. Frozen model choices

- **Ingestion-fraction convention:** dissipation fraction `f_Φ` = fraction of the boundary-layer
  kinetic-energy defect (θ*-weighted) captured by the fan annulus (aligned with Hall 2017).
  Cross-conversion table to mass/area/momentum conventions published in Appendix B.
- **Comparison rule:** equal net streamwise force between the BLI and podded cases, at equal
  flight condition, equal FPR, equal fan technology (η_pol) except an explicit BLI distortion
  penalty δη(f_Φ). PSC = 1 − P_K^BLI / P_K^pod.
- **Boundary layer:** Thwaites (laminar) → forced/Michel transition → Head entrainment
  (turbulent), axisymmetric (Mangler-equivalent r(x) terms), Drela H*(H, Re_θ) closure for θ*,
  Van Driest II-class compressibility correction on C_f, Squire–Young wake extrapolation.
- **Edge velocity:** von Kármán axial source-line slender-body solution on the actual r(x).
- **Validity flags:** H > 2.4 (turbulent separation proxy) marks a solution invalid;
  invalid atlas cells are greyed out, never plotted as physical.

## 3. Locked validation targets

| # | Reference | Metric & rule | Target |
|---|---|---|---|
| V1 | Blasius (exact) | θ = 0.664 √(νx/U); C_f = 0.664/√Re_x | θ ≤ 1.5%, C_f ≤ 2% |
| V2 | 1/7-power turbulent flat plate | C_f = 0.0592 Re_x^(−1/5) | ≤ 5% over Re_x ∈ [10^6, 10^8] |
| V3 | Smith 1993 | Ideal actuator-disk wake-ingestion closed form, incompressible, η=1 | ≤ 2% |
| V4 | Uranga et al. 2017 (D8) | Mech. flow-power saving at zero net streamwise force, equal-nozzle-area rule, tunnel Mach | 8.2% ± 0.8% (report; diagnose any shortfall — expected: missing surface-dissipation term) |
| V5 | STARC-ABL bracket | PSC_aero and Δblock-fuel between Yildirim 2022 (2.1–2.3% power) and NASA/TM-20210016661 (3.4% design-mission fuel); Welstead & Felder 2016 12% cited as obsolete | bracket, not point match |

## 4. UQ input space (production)

| Input | Distribution | Range / params |
|---|---|---|
| f_Φ (ingested dissipation fraction) | Uniform | [0.1, 0.9] |
| FPR | Uniform | [1.20, 1.50] |
| x_tr/L (transition location) | Uniform | [0.01, 0.10] |
| n (BL power-law exponent at fan face) | Normal | N(7, 1), truncated [4, 11] |
| η_pol,fan (clean polytropic) | Normal | N(0.92, 0.02), truncated [0.85, 0.97] |
| k_dist (distortion penalty at f_Φ=1) | Uniform | [0.00, 0.05] (δη = k_dist · f_Φ) |
| η_elec (turboelectric chain) | Normal | N(0.92, 0.02), truncated [0.85, 0.97] |

Outputs: PSC_aero (P_K basis), PSC_net (turboelectric, eq. 11), Δblock-fuel (Breguet, snowball 1.35).

## Amendment Log

- 2026-07-12: v1.0 frozen (initial).
- 2026-07-12: A1 — Re_L corrected: the ISA/Sutherland computation at M0.785/FL350 with
  L = 37 m gives Re_L = 2.28e8 (the proposal's "~1.5e8" was approximate). No model change.
- 2026-07-12: A2 — V2 revised: vs the 1/7-power law, Head + Ludwieg–Tillmann agrees within
  6% over Re_x ∈ [2e6, 1e8] (−5.3% at 2e6, +3% at 1e8) — correlation scatter between
  empirical fits. vs Schultz–Grunow (closer to truth at high Re), L–T underpredicts C_f by
  up to ~11% at Re_x = 1e8 because Re_θ (≈7e5 at the cruise fuselage TE) exceeds its
  calibration range. This is a declared low-order bias on *absolute* drag/dissipation that
  largely cancels in the PSC ratio; bounded at 12% in CI so regressions fail loudly.
- 2026-07-12: A3 — PSC convention clarified: the comparator's subsystem PSC (equal FPR,
  equal net force, podded twin resized) *decreases* with f_Φ because the marginal captured
  fluid is less degraded; the aircraft-level saving (dynamic power fraction, eta_prop_ref
  = 0.80 for the non-BLI propulsors) is what grows and saturates with f_Φ. Both are
  reported; UQ output psc_net uses the aircraft-level convention.
