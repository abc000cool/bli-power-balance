"""Saltelli-sampled Sobol' global sensitivity indices via SALib.

The Saltelli sample is drawn on the unit hypercube (preserving the
low-discrepancy structure) and pushed through the inverse-CDF transform to
the physical marginals, so arbitrary input distributions are supported with
standard SALib machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from blipb.uq.model import OUTPUT_NAMES, DEFAULT_CONFIG, StudyConfig, evaluate_batch
from blipb.uq.problem import INPUT_NAMES, salib_problem, transform_unit_samples


def run_sobol(
    n_base: int = 1024,
    config: StudyConfig = DEFAULT_CONFIG,
    n_jobs: int = -1,
    seed: int = 20260712,
) -> dict[str, pd.DataFrame]:
    """Saltelli/Sobol' analysis of all model outputs.

    Total model evaluations: n_base * (2 * 7 + 2).  Returns, per output, a
    DataFrame with S1, S1_conf, ST, ST_conf indexed by input name, plus the
    raw samples/outputs under key '_samples'.
    """
    problem = salib_problem()
    u = sobol_sample.sample(problem, n_base, calc_second_order=False, seed=seed)
    x = transform_unit_samples(u)
    y = evaluate_batch(x, config=config, n_jobs=n_jobs)

    n_nan = int(np.isnan(y[:, 0]).sum())
    if n_nan:
        # Saltelli analysis needs complete blocks; impute rare failures with
        # the column median and report the count (never silently).
        med = np.nanmedian(y, axis=0)
        idx = np.isnan(y)
        y[idx] = np.take(med, np.where(idx)[1])

    results: dict[str, pd.DataFrame] = {}
    for k, name in enumerate(OUTPUT_NAMES):
        si = sobol_analyze.analyze(
            problem, y[:, k], calc_second_order=False, seed=seed, print_to_console=False
        )
        results[name] = pd.DataFrame(
            {
                "S1": si["S1"],
                "S1_conf": si["S1_conf"],
                "ST": si["ST"],
                "ST_conf": si["ST_conf"],
            },
            index=INPUT_NAMES,
        )
    samples = pd.DataFrame(x, columns=INPUT_NAMES)
    for k, name in enumerate(OUTPUT_NAMES):
        samples[name] = y[:, k]
    samples.attrs["n_imputed"] = n_nan
    results["_samples"] = samples
    return results
