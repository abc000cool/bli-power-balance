"""1-D fan model verification."""

import numpy as np
import pytest

from blipb.atmosphere import CP_AIR, GAMMA, FlightState
from blipb.propulsion.fan import fan_jet


@pytest.fixture(scope="module")
def cruise() -> FlightState:
    return FlightState(mach=0.785, altitude=10_668.0)


def test_unity_fpr_returns_freestream(cruise):
    # FPR = 1 on a clean inlet: the "jet" is just the freestream again.
    res = fan_jet(cruise.pt, cruise.Tt, fpr=1.0, p_inf=cruise.atm.p, eta_pol=0.92)
    assert res.v_jet == pytest.approx(cruise.V, rel=1e-6)
    assert res.w_shaft == pytest.approx(0.0, abs=1e-6)


def test_ideal_shaft_work(cruise):
    fpr = 1.3
    res = fan_jet(cruise.pt, cruise.Tt, fpr=fpr, p_inf=cruise.atm.p, eta_pol=1.0)
    w_ideal = CP_AIR * cruise.Tt * (fpr ** ((GAMMA - 1) / GAMMA) - 1.0)
    assert res.w_shaft == pytest.approx(w_ideal, rel=1e-9)


def test_losses_increase_shaft_work_and_jet_temp(cruise):
    kw = dict(pt_in=cruise.pt, tt_in=cruise.Tt, fpr=1.3, p_inf=cruise.atm.p)
    ideal = fan_jet(eta_pol=1.0, **kw)
    real = fan_jet(eta_pol=0.90, **kw)
    assert real.w_shaft > ideal.w_shaft
    assert real.tt_out > ideal.tt_out
    # Same pt_out -> hot jet is slightly faster (thermal recovery)
    assert real.pt_out == pytest.approx(ideal.pt_out)
    assert real.v_jet > ideal.v_jet


def test_distortion_penalty_applies(cruise):
    kw = dict(pt_in=cruise.pt, tt_in=cruise.Tt, fpr=1.3, p_inf=cruise.atm.p, eta_pol=0.92)
    clean = fan_jet(delta_eta_distortion=0.0, **kw)
    dist = fan_jet(delta_eta_distortion=0.03, **kw)
    assert dist.eta_pol_eff == pytest.approx(0.89)
    assert dist.w_shaft > clean.w_shaft


def test_jet_velocity_increases_with_fpr(cruise):
    vjs = [
        fan_jet(cruise.pt, cruise.Tt, fpr=f, p_inf=cruise.atm.p).v_jet
        for f in np.linspace(1.05, 1.6, 8)
    ]
    assert all(b > a for a, b in zip(vjs, vjs[1:]))


def test_degraded_inlet_gives_slower_jet(cruise):
    clean = fan_jet(cruise.pt, cruise.Tt, fpr=1.25, p_inf=cruise.atm.p)
    degraded = fan_jet(0.85 * cruise.pt, cruise.Tt, fpr=1.25, p_inf=cruise.atm.p)
    assert degraded.v_jet < clean.v_jet


def test_invalid_inputs(cruise):
    with pytest.raises(ValueError):
        fan_jet(cruise.pt, cruise.Tt, fpr=0.9, p_inf=cruise.atm.p)
    with pytest.raises(ValueError):
        # inlet so degraded the jet cannot reach ambient pressure
        fan_jet(0.5 * cruise.atm.p, cruise.Tt, fpr=1.05, p_inf=cruise.atm.p)
