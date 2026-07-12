"""Conditional Sobol study: input space restricted to f_Phi in [0.5, 0.9].

Tests the paper's headline claim directly: once the installation captures at
least half the available dissipation, does further ingestion still drive the
variance of the delivered benefit, or do the technology uncertainties
(electrical chain, profile shape, distortion) take over?

Run:  uv run python studies/run_uq_conditional.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle  # noqa: E402

import blipb.uq.problem as problem  # noqa: E402

# Restrict the ingestion design range (study-level override; the frozen SPEC
# space is untouched for the main results).
problem.INPUT_SPECS = [
    ("f_phi", "uniform", (0.50, 0.90)) if name == "f_phi" else (name, kind, spec)
    for name, kind, spec in problem.INPUT_SPECS
]

from blipb.uq.model import OUTPUT_NAMES  # noqa: E402
from blipb.uq.sobol import run_sobol  # noqa: E402

if __name__ == "__main__":
    sob = run_sobol(n_base=1024)
    for name in OUTPUT_NAMES:
        sob[name].to_parquet(plotstyle.DATADIR / f"uq_cond_sobol_{name}.parquet")
        print(f"\n== conditional (f_phi >= 0.5) Sobol: {name} ==")
        print(sob[name].to_string(float_format=lambda v: f"{v:+.3f}"))
    sob["_samples"].to_parquet(plotstyle.DATADIR / "uq_cond_sobol_samples.parquet", index=False)
