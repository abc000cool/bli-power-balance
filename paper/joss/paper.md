---
title: 'blipb: A power-balance boundary-layer-ingestion benefit model with uncertainty quantification'
tags:
  - Python
  - aerospace engineering
  - propulsion
  - boundary-layer ingestion
  - uncertainty quantification
  - sensitivity analysis
authors:
  - name: Ansh Pathak
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 July 2026
bibliography: paper.bib
---

# Summary

Boundary-layer ingestion (BLI), placing an aircraft propulsor so it swallows
the slow air of the airframe's boundary layer, is one of the few remaining
airframe-propulsion integration measures projected to cut transport-aircraft
fuel burn by several percent. Because the airframe wake and the propulsive jet
share a stream tube, conventional thrust/drag bookkeeping is ambiguous for BLI
configurations; the community-standard resolution is Drela's power-balance
method [@drela2009], which accounts for mechanical-energy sources and
dissipation instead of forces.

`blipb` is a pip-installable Python implementation of the power-balance BLI
benefit model, unifying the classical wake-ingestion analysis of
@smith1993, the power-balance framework of @drela2009, and the
dissipation-decomposition formulation of @hall2017 in a single reproducible
pipeline: an axisymmetric integral boundary-layer chain
(Thwaites/Head/Squire–Young with XFOIL-family closures [@drela1987]), a
two-case podded-versus-BLI comparator with an algebraically exact
bookkeeping-residual check, a one-dimensional compressible fan with an explicit
inlet-distortion penalty, a turboelectric transmission chain, Breguet
mission accounting, and formal uncertainty quantification (Saltelli–Sobol'
global sensitivity indices via SALib [@herman2017] cross-checked against
polynomial-chaos surrogates via chaospy [@feinberg2015]).

# Statement of need

Published BLI benefit numbers span 2–12% and are frequently contradictory
because they measure different quantities under different bookkeeping rules
[@moirou2023; @ma2025]. High-fidelity coupled RANS/cycle analyses
[@yildirim2022; @gray2018] are physically superior but rely on proprietary or
group-internal toolchains, cannot sweep the joint design/uncertainty space
exhaustively, and have not published formal global sensitivity analyses. No
open-source, tested, documented implementation of the Drela/Smith/Hall
power-balance BLI model existed: general conceptual-design frameworks (NASA
Aviary, OpenConcept, SUAVE/RCAIDE) treat BLI, when at all, as an assumed
coupling coefficient rather than a first-principles dissipation ledger.

`blipb` fills that gap for researchers and conceptual designers who need
(i) transparent, rule-explicit reproductions of the canonical benefit numbers
(D8 wind-tunnel measurement [@uranga2017], STARC-ABL assessments
[@welstead2016; @felder2022; @yildirim2022]); (ii) parametric benefit maps
with per-cell validity flags; and (iii) probability distributions and Sobol'
importance rankings of the benefit under realistic technology uncertainty.
The accompanying research paper reports these results; every figure and
number regenerates from the archived pipeline (`uv sync && pytest` followed
by the study scripts).

# Software design

The package enforces the bookkeeping discipline that BLI comparisons
require: a frozen `ControlVolume` object is shared by both cases of a
comparison; ingestion is quantified as the Hall-2017 dissipation fraction
with published cross-conversions; and the exact ledger identity
$P_{K,\mathrm{pod}} - P_{K,\mathrm{BLI}} = \Delta\Phi_{\mathrm{jet}} +
\Delta\Phi_{\mathrm{wake}}$ is asserted at every evaluation and enforced to
$10^{-10}$ relative in continuous integration, so any future modification
that breaks energy bookkeeping fails continuous integration
immediately. Known low-order biases (e.g.
Ludwieg–Tillmann skin-friction underprediction at very high $Re_\theta$) are
documented in the frozen specification (`SPEC.md`) and bounded by regression
tests rather than hidden.

# Research impact / AI usage disclosure

The initial implementation was developed with the assistance of an AI coding
agent (Claude, Anthropic) under continuous human direction and review; all
physics choices, validation targets and tolerances follow the pre-registered
specification in `SPEC.md`, and the full test suite and validation notebooks
were executed and inspected by the authors; the complete build-session
transcript is archived with the repository for provenance.

# References
