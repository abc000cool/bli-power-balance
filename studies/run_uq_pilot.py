"""Pilot UQ run: Saltelli N=1024 Sobol + order-3 PCE + 1e4 MC on surrogate.

~9k model evaluations for the Sobol pass (n_base * 9 with the
first/total-order Saltelli scheme) and ~500 for the PCE design;
minutes on a laptop with joblib.  Products under data/:

  uq_sobol_<output>.parquet   S1/ST tables per output
  uq_sobol_samples.parquet    raw Saltelli samples + outputs
  uq_pce_design.parquet       PCE training design + outputs
  uq_pce_sobol_<output>.parquet  PCE-derived Sobol indices (cross-check)
  uq_pce_moments.parquet      PCE means/sds
  uq_mc_samples.parquet       1e4 LHS MC on the surrogate
  uq_mc_percentiles.parquet   5/50/95 bands

Run:  uv run python studies/run_uq_pilot.py  [--n-base 1024]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotstyle  # noqa: E402

from blipb.uq.model import OUTPUT_NAMES  # noqa: E402
from blipb.uq.pce import fit_pce, mc_on_surrogate  # noqa: E402
from blipb.uq.sobol import run_sobol  # noqa: E402


def main(n_base: int = 1024, tag: str = "") -> None:
    datadir = plotstyle.DATADIR
    datadir.mkdir(exist_ok=True)

    t0 = time.time()
    print(f"Saltelli/Sobol with N = {n_base} ({n_base * 9} evaluations)...")
    sob = run_sobol(n_base=n_base)
    for name in OUTPUT_NAMES:
        sob[name].to_parquet(datadir / f"uq{tag}_sobol_{name}.parquet")
        print(f"\n== Sobol indices: {name} ==")
        print(sob[name].to_string(float_format=lambda v: f"{v:+.3f}"))
    sob["_samples"].to_parquet(datadir / f"uq{tag}_sobol_samples.parquet", index=False)
    print(f"\nimputed rows: {sob['_samples'].attrs.get('n_imputed', 0)}")
    print(f"Sobol done in {time.time() - t0:.0f}s")

    t1 = time.time()
    print("\nFitting order-3 PCE surrogate...")
    pce = fit_pce(order=3)
    pce["design"].to_parquet(datadir / f"uq{tag}_pce_design.parquet", index=False)
    pce["moments"].to_parquet(datadir / f"uq{tag}_pce_moments.parquet")
    for name in OUTPUT_NAMES:
        pce["sobol"][name].to_parquet(datadir / f"uq{tag}_pce_sobol_{name}.parquet")
    print(pce["moments"].to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\nMonte Carlo (1e4 LHS) on the surrogate...")
    mc = mc_on_surrogate(pce, n_samples=10_000)
    mc["samples"].to_parquet(datadir / f"uq{tag}_mc_samples.parquet", index=False)
    mc["percentiles"].to_parquet(datadir / f"uq{tag}_mc_percentiles.parquet")
    print(mc["percentiles"].to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"PCE + MC done in {time.time() - t1:.0f}s")

    # Cross-check: top-3 total-order rankings must agree between methods
    for name in OUTPUT_NAMES:
        top_salib = list(sob[name]["ST"].sort_values(ascending=False).index[:3])
        top_pce = list(pce["sobol"][name]["ST"].sort_values(ascending=False).index[:3])
        flag = "OK" if set(top_salib) == set(top_pce) else "MISMATCH"
        print(f"top-3 ST cross-check [{name}]: SALib {top_salib} vs PCE {top_pce} -> {flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=1024)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    main(n_base=args.n_base, tag=args.tag)
