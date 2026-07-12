"""Ingested-annulus integral identities (the defect-ledger algebra)."""

import numpy as np
import pytest

from blipb.powerbalance.streamtube import AnnulusProfile


@pytest.fixture(scope="module")
def profile() -> AnnulusProfile:
    return AnnulusProfile(
        theta_eq=0.11,
        n_powerlaw=7.0,
        r_hub=0.56,
        v_edge=232.0,
        rho=0.38,
        p_static=23_842.0,
        tt=246.0,
    )


def test_delta_sizing_matches_defect(profile):
    # The annulus momentum defect must equal the IBL-delivered defect
    target = 2 * np.pi * profile.rho * profile.v**2 * profile.r_hub * profile.theta_eq
    got = profile.momentum_defect_total()
    assert got == pytest.approx(target, rel=1e-6)


def test_fphi_monotone_and_bounded(profile):
    fs = [profile.capture(y).f_phi for y in np.linspace(0.05, 1.0, 12) * profile.delta]
    assert all(0.0 < f <= 1.0 for f in fs)
    assert all(b >= a for a, b in zip(fs, fs[1:]))
    assert profile.capture(profile.delta).f_phi == pytest.approx(1.0, abs=1e-9)


def test_capture_beyond_delta_adds_no_defect(profile):
    c1 = profile.capture(profile.delta)
    c2 = profile.capture(2.0 * profile.delta)
    assert c2.momentum_defect == pytest.approx(c1.momentum_defect, rel=1e-9)
    assert c2.m_dot > c1.m_dot  # but it does add freestream mass flow


def test_energy_identity_exact(profile):
    # 1/2 u (u^2 - V^2) = -V u (V - u) + 1/2 u (V - u)^2, integrated:
    # E_exc = -V D_dot + E_a  -- must hold to machine precision because the
    # comparator's exact ledger identity rests on it.
    for frac in (0.2, 0.5, 0.8, 1.0):
        c = profile.capture(frac * profile.delta)
        lhs = c.ke_flux_excess
        rhs = -profile.v * c.momentum_defect + c.ea_defect
        assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-9)


def test_pt_degraded_below_freestream(profile):
    from blipb.atmosphere import GAMMA

    c = profile.capture(0.5 * profile.delta)
    mach_inf = profile.v / np.sqrt(GAMMA * 287.05287 * (profile.tt - profile.v**2 / 2010.0))
    pt_inf = profile.p * (1 + 0.2 * mach_inf**2) ** 3.5
    assert c.pt_mass_avg < pt_inf
    assert c.pt_mass_avg > profile.p  # but still above static


def test_solve_capture_for_fphi(profile):
    for f in (0.25, 0.5, 0.75):
        c = profile.solve_capture_for_fphi(f)
        assert c.f_phi == pytest.approx(f, abs=1e-6)


def test_mass_average_velocity_below_edge(profile):
    c = profile.capture(profile.delta)
    assert 0.5 * profile.v < c.u_mass_avg < profile.v
