"""Fuselage geometry and slender-body edge velocity sanity checks."""

import numpy as np
import pytest

from blipb.geometry import Fuselage


@pytest.fixture(scope="module")
def fus() -> Fuselage:
    return Fuselage()


def test_radius_endpoints(fus):
    assert fus.radius(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert fus.radius(np.array([fus.length]))[0] == pytest.approx(fus.r_tail)
    mid = fus.radius(np.array([0.5 * fus.length]))[0]
    assert mid == pytest.approx(fus.radius_max)


def test_radius_monotone_segments(fus):
    x = np.linspace(0, fus.length, 1000)
    r = fus.radius(x)
    nose = x < fus.nose_frac * fus.length
    tail = x > (1 - fus.tail_frac) * fus.length
    assert np.all(np.diff(r[nose]) >= -1e-12)
    assert np.all(np.diff(r[tail]) <= 1e-12)
    assert np.all(r >= 0.0)


def test_wetted_area_plausible(fus):
    # Bounded by the full cylinder of the same length, above 60% of it
    s_cyl = 2 * np.pi * fus.radius_max * fus.length
    s = fus.wetted_area()
    assert 0.6 * s_cyl < s < s_cyl


def test_edge_velocity_midbody_near_freestream(fus):
    v = 232.0
    x = fus.grid(400)
    ue = fus.edge_velocity(x, v, mach=0.0)
    mid = (x > 0.3 * fus.length) & (x < 0.6 * fus.length)
    assert np.allclose(ue[mid], v, rtol=0.03)


def test_edge_velocity_tail_adverse(fus):
    # The tail-cone contraction must decelerate the edge flow (adverse
    # gradient region), and never produce negative or absurd velocities.
    v = 232.0
    x = fus.grid(400)
    ue = fus.edge_velocity(x, v, mach=0.785)
    tail = x > 0.85 * fus.length
    assert ue[tail].min() < v
    assert np.all(ue > 0.0)
    assert np.all(ue < 1.5 * v)


def test_compressibility_amplifies_perturbation(fus):
    v = 232.0
    x = fus.grid(200)
    ue0 = fus.edge_velocity(x, v, mach=0.0)
    ue8 = fus.edge_velocity(x, v, mach=0.785)
    # Perturbation magnitude grows with the Goethert factor (midbody and
    # tail; the inner nose is a stagnation ramp in both cases)
    mid = x > 0.2 * fus.length
    assert np.max(np.abs(ue8[mid] - v)) > np.max(np.abs(ue0[mid] - v))
