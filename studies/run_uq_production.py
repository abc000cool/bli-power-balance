"""Production UQ run: Saltelli N = 2^14 (~147k evaluations, first/total-order).

Same pipeline as the pilot but at publication sampling density; outputs are
tagged `_prod` and written to data/production/ (git-ignored; archive to
Zenodo).  Runs in minutes on a modern laptop: the boundary-layer solve is
LRU-cached on quantized x_tr, so only ~90 IBL solves occur.

Run:  uv run python studies/run_uq_production.py
Then regenerate figures with:  uv run python studies/figures.py --uq-tag _prod
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle  # noqa: E402
import run_uq_pilot  # noqa: E402

if __name__ == "__main__":
    plotstyle.DATADIR = plotstyle.DATADIR / "production"
    run_uq_pilot.plotstyle.DATADIR = plotstyle.DATADIR
    run_uq_pilot.main(n_base=2**14, tag="_prod")
