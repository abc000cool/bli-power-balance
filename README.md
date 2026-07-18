# blipb — power-balance BLI benefit model with uncertainty quantification

[![CI](https://github.com/abc000cool/bli-power-balance/actions/workflows/ci.yml/badge.svg)](https://github.com/abc000cool/bli-power-balance/actions)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

`blipb` is a low-order, open-source, reproducible implementation of the **Drela (2009)
power-balance boundary-layer-ingestion (BLI) benefit model**, unified with the
**Smith (1993)** wake-ingestion analysis and the **Hall et al. (2017)** dissipation
decomposition, coupled to **Sobol' global sensitivity analysis** (SALib) and
**polynomial-chaos + Monte-Carlo uncertainty quantification** (chaospy).

The energy bookkeeping satisfies an *exact* ledger identity —
`PK_pod − PK_BLI = ΔΦ_jet + ΔΦ_wake` — asserted at every evaluation and enforced to
1e-10 relative in CI, so a broken comparison fails loudly instead of biasing silently.

## Quickstart

```bash
# with uv (https://docs.astral.sh/uv/) — Python 3.11/3.12 auto-managed
uv sync --extra dev
uv run pytest                      # 63 tests, ~5 s

uv run python - <<'PY'
from blipb import BLIComparator
comp = BLIComparator()             # STARC-ABL-class baseline: M0.785, FL350
res = comp.run_design(f_phi=0.5, fpr=1.25)
print(f"subsystem PSC = {res.psc:.1%}, ledger residual = {res.ledger_residual:.1e}")
PY
```

## Reproduce the paper

```bash
uv run python studies/atlas.py                 # parametric sweeps  (~2 min)
uv run python validation/make_table.py         # Smith / D8 / STARC-ABL table
uv run python studies/run_uq_pilot.py          # Sobol N=1024 + PCE + MC (~20 min)
uv run python studies/conventions_table.py     # Appendix B cross-table
uv run python studies/figures.py               # all 8 paper figures
uv run python studies/export_numbers.py        # paper/numbers.tex macros
# production UQ (N=2^14, ~147k evals — minutes on a modern laptop):
uv run python studies/run_uq_production.py
uv run python studies/figures.py --uq-tag _prod
```

All sweep outputs are cached as Parquet under `data/`; figures land in `figures/`;
the manuscript (`paper/main.tex`) pulls every number from generated macros.

## Package layout

```
src/blipb/
  atmosphere.py        ISA + flight state
  geometry.py          parametric fuselage + von Kármán source-line edge velocity
  ibl/                 Thwaites (Mangler) → Head (axisymmetric) → Squire–Young,
                       XFOIL-family H*(H, Re_θ) closures, Van Driest II-class C_f
  powerbalance/        ControlVolume, ingested-annulus streamtube, two-case
                       comparator, Smith 1993 closed forms, Hall 2017 decomposition
  propulsion/          1-D compressible fan (+ distortion penalty), turboelectric chain
  mission/             Breguet block fuel
  uq/                  7-input problem spec, Saltelli/Sobol (SALib), PCE (chaospy)
studies/               atlas, UQ drivers, figures, number export
validation/            per-rule validations: Smith limit, D8 8.2±0.8%, STARC-ABL bracket
tests/                 63 verification + regression tests (Blasius, flat plate,
                       ledger identity, defect algebra, …)
paper/                 manuscript (LaTeX) + JOSS companion
SPEC.md                frozen baseline & validation targets, with amendment log
```

## Model in one paragraph

An axisymmetric integral boundary-layer chain produces the fuselage trailing-edge
state and, via the kinetic-energy-thickness identity `½ρu³θ* = Φ`, the cumulative
surface dissipation. The layer is mapped to a power-law annulus at the fan face;
a capture height ingests a prescribed **dissipation fraction f_Φ** (Hall 2017
convention; cross-conversions in Appendix B). A compressible fan applies its FPR
to the mass-averaged degraded total pressure (total temperature is preserved in an
adiabatic layer — that asymmetry *is* the BLI benefit), and a podded twin at equal
FPR and equal net streamwise force provides the reference. PSC and its exact
jet/wake decomposition follow; a turboelectric chain and Breguet close the loop to
block fuel.

## Known limitations (declared, tested, bounded)

- No fan-suction **surface-dissipation** term (measured at 2.4 points of the D8's
  8.2% — requires coupled aero-propulsive analysis).
- η_fill = 1 idealization: the fan re-energizes the captured profile to uniform pt.
- Ludwieg–Tillmann underpredicts absolute C_f by up to ~11% at Re_x = 1e8
  (bounded by a CI regression test; largely cancels in the PSC ratio).
- Axisymmetric geometry; 3-D/unsteady distortion enters only through the n and
  k_dist uncertainty inputs.

See `SPEC.md` (amendment log) and paper §6 for the complete list.

## Citing

If you use `blipb`, please cite the research paper (in preparation; see
`paper/main.tex`) and the archived software release
([doi:10.5281/zenodo.21434699](https://doi.org/10.5281/zenodo.21434699), v0.1.0).

## License

BSD-3-Clause. © 2026 Ansh Pathak.
