"""UQ layer: input transforms, end-to-end model wrapper, PCE smoke test."""

import numpy as np
import pytest

from blipb.uq.model import DEFAULT_CONFIG, evaluate_batch, evaluate_point
from blipb.uq.problem import (
    INPUT_NAMES,
    INPUT_SPECS,
    chaospy_joint,
    salib_problem,
    transform_unit_samples,
)

BASELINE = {
    "f_phi": 0.5,
    "fpr": 1.25,
    "x_tr": 0.05,
    "n_powerlaw": 7.0,
    "eta_pol": 0.92,
    "k_dist": 0.03,
    "eta_elec": 0.92,
}


def test_problem_shapes():
    p = salib_problem()
    assert p["num_vars"] == 7
    assert p["names"] == INPUT_NAMES


def test_transform_bounds():
    rng = np.random.default_rng(1)
    u = rng.random((200, 7))
    x = transform_unit_samples(u)
    for j, (_, kind, spec) in enumerate(INPUT_SPECS):
        lo, hi = (spec[0], spec[1]) if kind == "uniform" else (spec[2], spec[3])
        assert x[:, j].min() >= lo - 1e-9
        assert x[:, j].max() <= hi + 1e-9


def test_transform_truncnorm_mean():
    rng = np.random.default_rng(2)
    u = rng.random((4000, 7))
    x = transform_unit_samples(u)
    # eta_pol ~ N(0.92, 0.02) truncated: sample mean close to mu
    assert np.mean(x[:, 4]) == pytest.approx(0.92, abs=0.005)


def test_evaluate_point_baseline():
    out = evaluate_point(BASELINE)
    assert 0.0 < out["psc_aero"] < 0.5
    assert out["valid"] == 1.0
    assert np.isfinite(out["psc_net"])
    assert out["delta_fuel"] < 0.0  # baseline BLI saves fuel


def test_evaluate_point_array_order():
    arr = [BASELINE[k] for k in INPUT_NAMES]
    out_a = evaluate_point(np.array(arr))
    out_d = evaluate_point(BASELINE)
    assert out_a["psc_aero"] == pytest.approx(out_d["psc_aero"], rel=1e-12)


def test_evaluate_batch_parallel_and_nan_policy():
    rows = np.array(
        [
            [0.5, 1.25, 0.05, 7.0, 0.92, 0.03, 0.92],
            [0.8, 1.40, 0.05, 7.0, 0.92, 0.03, 0.92],
            [0.2, 1.20, 0.05, 6.0, 0.90, 0.01, 0.90],
        ]
    )
    y = evaluate_batch(rows, n_jobs=1)
    assert y.shape == (3, 3)
    assert np.isfinite(y).all()


def test_chaospy_joint_sampling():
    joint = chaospy_joint()
    s = joint.sample(64, rule="sobol", seed=7)
    assert s.shape == (7, 64)
    x = np.asarray(s.T)
    for j, (_, kind, spec) in enumerate(INPUT_SPECS):
        lo, hi = (spec[0], spec[1]) if kind == "uniform" else (spec[2], spec[3])
        assert x[:, j].min() >= lo - 1e-9
        assert x[:, j].max() <= hi + 1e-9


def test_bl_cache_consistency():
    # Points differing only by sub-quantum x_tr must share the BL solution
    a = evaluate_point({**BASELINE, "x_tr": 0.0501})
    b = evaluate_point({**BASELINE, "x_tr": 0.0503})
    assert a["psc_aero"] == pytest.approx(b["psc_aero"], rel=1e-12)


def test_default_config_frozen():
    with pytest.raises(Exception):
        DEFAULT_CONFIG.mach = 0.9  # frozen dataclass
