"""Polynomial-chaos expansion surrogate via chaospy.

The expansion is built on the *unit hypercube* with uniform marginals
(orthonormal Legendre basis); physical inputs enter through the same
inverse-CDF transform used by the Saltelli/Sobol path (problem.py).  Sobol'
indices are invariant under monotone per-input transforms, and Monte-Carlo
pushforwards are identical, so nothing is lost -- while expectation,
variance and Sens_m/Sens_t on the uniform joint are algebraic and fast
(the same operations on a truncated-normal joint in chaospy cost minutes).

Order-3 total-degree expansion fitted by point-collocation least squares on
a Sobol-sequence design with ~4x oversampling (d = 7 -> 120 basis terms,
~480 training points).
"""

from __future__ import annotations

import chaospy
import numpy as np
import pandas as pd

from blipb.uq.model import OUTPUT_NAMES, DEFAULT_CONFIG, StudyConfig, evaluate_batch
from blipb.uq.problem import INPUT_NAMES, transform_unit_samples


def fit_pce(
    order: int = 3,
    oversampling: float = 4.0,
    config: StudyConfig = DEFAULT_CONFIG,
    n_jobs: int = -1,
    seed: int = 20260712,
) -> dict:
    """Fit PCE surrogates for every model output.

    Returns dict with keys: 'models' (per-output chaospy polynomial over the
    unit hypercube), 'joint' (the uniform unit joint), 'design' (DataFrame
    of physical training points + outputs), 'moments' (DataFrame mean/sd),
    'sobol' (per-output DataFrame with S1/ST from the PCE coefficients).
    """
    joint = chaospy.J(*[chaospy.Uniform(0.0, 1.0) for _ in INPUT_NAMES])
    basis = chaospy.generate_expansion(order, joint, normed=True)
    n_train = int(oversampling * len(basis))
    u_train = joint.sample(n_train, rule="sobol", seed=seed)  # (7, N) in [0,1]
    x_train = transform_unit_samples(u_train.T)  # physical space
    y_train = evaluate_batch(x_train, config=config, n_jobs=n_jobs)  # (N, 3)

    ok = ~np.isnan(y_train[:, 0])
    n_dropped = int((~ok).sum())
    u_ok, y_ok = u_train[:, ok], y_train[ok]

    models = {}
    moments = {}
    sobol_tables = {}
    for k, name in enumerate(OUTPUT_NAMES):
        poly = chaospy.fit_regression(basis, u_ok, y_ok[:, k])
        models[name] = poly
        mean = float(chaospy.E(poly, joint))
        sd = float(chaospy.Std(poly, joint))
        moments[name] = {"mean": mean, "sd": sd}
        s1 = chaospy.Sens_m(poly, joint)
        st = chaospy.Sens_t(poly, joint)
        sobol_tables[name] = pd.DataFrame({"S1": s1, "ST": st}, index=INPUT_NAMES)

    design = pd.DataFrame(x_train, columns=INPUT_NAMES)
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
    """Latin-hypercube Monte Carlo on the PCE surrogate (unit-space inputs).

    Returns dict with 'samples' (DataFrame of outputs) and 'percentiles'
    (DataFrame indexed by output).
    """
    joint = pce["joint"]
    u = joint.sample(n_samples, rule="latin_hypercube", seed=seed)
    out = {}
    pct_rows = {}
    for name, poly in pce["models"].items():
        vals = np.asarray(poly(*u), dtype=float)
        out[name] = vals
        pct_rows[name] = {f"p{int(p)}": float(np.percentile(vals, p)) for p in percentiles}
    return {
        "samples": pd.DataFrame(out),
        "percentiles": pd.DataFrame(pct_rows).T,
    }
