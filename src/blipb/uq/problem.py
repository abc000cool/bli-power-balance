"""UQ input-space definition (SPEC.md section 4), shared by SALib and chaospy.

Seven inputs; design variables are Uniform, uncertain physics/technology
parameters are truncated Normals.  SALib's Saltelli sampler works on the
(finite) support bounds with the matching distributions; chaospy builds the
same joint for PCE.
"""

from __future__ import annotations

import chaospy

INPUT_NAMES = ["f_phi", "fpr", "x_tr", "n_powerlaw", "eta_pol", "k_dist", "eta_elec"]

# (name, kind, params): Uniform -> (lo, hi); TruncNormal -> (mu, sd, lo, hi)
INPUT_SPECS: list[tuple[str, str, tuple[float, ...]]] = [
    ("f_phi", "uniform", (0.10, 0.90)),
    ("fpr", "uniform", (1.20, 1.50)),
    ("x_tr", "uniform", (0.01, 0.10)),
    ("n_powerlaw", "truncnorm", (7.0, 1.0, 4.0, 11.0)),
    ("eta_pol", "truncnorm", (0.92, 0.02, 0.85, 0.97)),
    ("k_dist", "uniform", (0.00, 0.05)),
    ("eta_elec", "truncnorm", (0.92, 0.02, 0.85, 0.97)),
]


def salib_problem() -> dict:
    """SALib problem dict with distribution information.

    SALib supports 'unif' and 'truncnorm' via the `dists` key; truncnorm
    bounds entries are [lo, hi] with (mu, sd) appended per SALib convention:
    bounds = [lo, hi, mu, sd] is not supported, so we sample truncated
    normals by transforming uniform Saltelli samples through the inverse CDF
    in model.py instead.  Here the problem is defined on [0, 1]^7 and the
    transform is applied downstream -- this keeps Saltelli's low-discrepancy
    structure intact for arbitrary marginals.
    """
    return {
        "num_vars": len(INPUT_SPECS),
        "names": [s[0] for s in INPUT_SPECS],
        "bounds": [[0.0, 1.0]] * len(INPUT_SPECS),
    }


def transform_unit_samples(u):
    """Map unit-hypercube samples (N, 7) to physical space via inverse CDFs."""
    import numpy as np
    from scipy import stats

    u = np.atleast_2d(np.asarray(u, dtype=float))
    x = np.empty_like(u)
    for j, (_, kind, p) in enumerate(INPUT_SPECS):
        uj = np.clip(u[:, j], 1e-9, 1.0 - 1e-9)
        if kind == "uniform":
            lo, hi = p
            x[:, j] = lo + (hi - lo) * uj
        elif kind == "truncnorm":
            mu, sd, lo, hi = p
            a, b = (lo - mu) / sd, (hi - mu) / sd
            x[:, j] = stats.truncnorm.ppf(uj, a, b, loc=mu, scale=sd)
        else:  # pragma: no cover
            raise ValueError(f"unknown distribution kind {kind}")
    return x


def chaospy_joint() -> chaospy.Distribution:
    """Joint distribution for PCE (chaospy)."""
    margins = []
    for _, kind, p in INPUT_SPECS:
        if kind == "uniform":
            lo, hi = p
            margins.append(chaospy.Uniform(lo, hi))
        else:
            mu, sd, lo, hi = p
            margins.append(chaospy.TruncNormal(lo, hi, mu, sd))
    return chaospy.J(*margins)
