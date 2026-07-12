"""Polynomial-chaos expansion surrogate via chaospy.

Order-3 total-degree expansion fitted by regularized least squares on a
Sobol-sequence experimental design (point-collocation; robust for d = 7,
order 3 -> 120 basis terms with ~4x oversampling).  The surrogate yields
analytic moments, Sobol' indices (cross-checked against SALib/Saltelli),
and cheap 1e4-sample Monte-Carlo percentile bands.
"""

from __future__ import annotations

import chaospy
import numpy as np
import pandas as pd

from blipb.uq.model import OUTPUT_NAMES, DEFAULT_CONFIG, StudyConfig, evaluate_batch
from blipb.uq.problem import INPUT_NAMES, chaospy_joint


def fit_pce(
    order: int = 3,
    oversampling: float = 4.0,
    config: StudyConfig = DEFAULT_CONFIG,
    n_jobs: int = -1,
    seed: int = 20260712,
) -> dict:
    """Fit PCE surrogates for every model output.

    Returns dict with keys: 'models' (per-output chaospy polynomial),
    'joint', 'design' (DataFrame of training points + outputs),
    'moments' (DataFrame mean/sd), 'sobol' (per-output DataFrame with
    S1/ST from the PCE coefficients).
    """
    joint = chaospy_joint()
    basis = chaospy.generate_expansion(order, joint, normed=True)
    n_train = int(oversampling * len(basis))
    x_train = joint.sample(n_train, rule="sobol", seed=seed)  # (7, N)
    y_train = evaluate_batch(x_train.T, config=config, n_jobs=n_jobs)  # (N, 3)

    ok = ~np.isnan(y_train[:, 0])
    n_dropped = int((~ok).sum())
    x_ok, y_ok = x_train[:, ok], y_train[ok]

    models = {}
    moments = {}
    sobol_tables = {}
    for k, name in enumerate(OUTPUT_NAMES):
        poly = chaospy.fit_regression(basis, x_ok, y_ok[:, k])
        models[name] = poly
        mean = float(chaospy.E(poly, joint))
        sd = float(chaospy.Std(poly, joint))
        moments[name] = {"mean": mean, "sd": sd}
        s1 = chaospy.Sens_m(poly, joint)
        st = chaospy.Sens_t(poly, joint)
        sobol_tables[name] = pd.DataFrame({"S1": s1, "ST": st}, index=INPUT_NAMES)

    design = pd.DataFrame(x_train.T, columns=INPUT_NAMES)
    for k, name in enumerate(OUTPUT_NAMES):
        design[name] = y_train[:, k]
    design.attrs["n_dropped"] = n_dropped

    return {
        "models": models,
        "joint": joint,
        "design": design,
        "moments": pd.DataFrame(moments).T,
        "sobol": sobol_tables,
        "order": order,
    }


def mc_on_surrogate(
    pce: dict,
    n_samples: int = 10_000,
    seed: int = 42,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
) -> dict:
    """Latin-hypercube Monte Carlo on the PCE surrogate.

    Returns dict with 'samples' (DataFrame of outputs) and 'percentiles'
    (DataFrame indexed by output).
    """
    joint = pce["joint"]
    x = joint.sample(n_samples, rule="latin_hypercube", seed=seed)
    out = {}
    pct_rows = {}
    for name, poly in pce["models"].items():
        vals = np.asarray(poly(*x), dtype=float)
        out[name] = vals
        pct_rows[name] = {f"p{int(p)}": float(np.percentile(vals, p)) for p in percentiles}
    return {
        "samples": pd.DataFrame(out),
        "percentiles": pd.DataFrame(pct_rows).T,
    }
