"""Smith 1993 closed forms and Hall 2017 helpers."""

import pytest

from blipb.powerbalance.smith1993 import (
    psc_parametric,
    psc_uniform_wake_full_ingestion,
)


def test_uniform_wake_limits():
    assert psc_uniform_wake_full_ingestion(1.0) == pytest.approx(0.0)
    # w = 0.8: PSC = 2(1-w)/(3-w) = 0.4/2.2
    assert psc_uniform_wake_full_ingestion(0.8) == pytest.approx(0.4 / 2.2)
    # deeper wake -> larger benefit
    assert psc_uniform_wake_full_ingestion(0.6) > psc_uniform_wake_full_ingestion(0.9)


def test_uniform_wake_invalid():
    with pytest.raises(ValueError):
        psc_uniform_wake_full_ingestion(0.0)
    with pytest.raises(ValueError):
        psc_uniform_wake_full_ingestion(1.2)


def test_parametric_estimator():
    # Proposal eq. 8 sanity: zero fill -> zero PSC; jet == wake -> zero
    assert psc_parametric(0.5, 0.0, 1.2, 0.8) == 0.0
    assert psc_parametric(0.5, 1.0, 0.8, 0.8) == pytest.approx(0.0)
    val = psc_parametric(0.5, 0.9, 1.2, 0.85)
    assert 0.0 < val < 0.2
