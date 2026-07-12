"""ISA and flight-state verification against standard-atmosphere tables."""

import numpy as np
import pytest

from blipb.atmosphere import FlightState, isa


def test_sea_level():
    atm = isa(0.0)
    assert atm.T == pytest.approx(288.15)
    assert atm.p == pytest.approx(101_325.0)
    assert atm.rho == pytest.approx(1.225, rel=1e-3)
    assert atm.a == pytest.approx(340.29, rel=1e-3)


def test_tropopause():
    atm = isa(11_000.0)
    assert atm.T == pytest.approx(216.65)
    assert atm.p == pytest.approx(22_632.0, rel=1e-3)


def test_fl350():
    atm = isa(10_668.0)
    assert atm.T == pytest.approx(218.81, rel=1e-3)
    assert atm.p == pytest.approx(23_842.0, rel=2e-3)  # standard table value
    assert atm.rho == pytest.approx(0.3796, rel=2e-3)


def test_stratosphere_exponential():
    atm = isa(15_000.0)
    assert atm.T == pytest.approx(216.65)
    assert atm.p == pytest.approx(12_045.0, rel=3e-3)


def test_out_of_range():
    with pytest.raises(ValueError):
        isa(-10.0)
    with pytest.raises(ValueError):
        isa(30_000.0)


def test_flight_state_cruise():
    fl = FlightState(mach=0.785, altitude=10_668.0)
    assert fl.V == pytest.approx(0.785 * fl.atm.a)
    # Total temperature ratio 1 + 0.2 M^2
    assert fl.Tt / fl.atm.T == pytest.approx(1.0 + 0.2 * 0.785**2)
    # Total pressure ratio (isentropic)
    assert fl.pt / fl.atm.p == pytest.approx((1.0 + 0.2 * 0.785**2) ** 3.5)
    # Reynolds number: SPEC baseline documents Re_L = 2.3e8 at L = 37 m
    assert fl.reynolds(37.0) == pytest.approx(2.28e8, rel=0.02)


def test_viscosity_sutherland():
    atm = isa(0.0)
    assert atm.mu == pytest.approx(1.789e-5, rel=5e-3)
    assert np.isfinite(atm.nu)
