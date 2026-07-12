"""Breguet fuel burn and turboelectric chain."""

import numpy as np
import pytest

from blipb.atmosphere import G0
from blipb.mission.breguet import NMI, block_fuel, delta_block_fuel
from blipb.propulsion.turboelectric import TurboelectricChain, net_psc


def test_block_fuel_hand_value():
    # 3000 nmi, V=233 m/s, L/D=21.4, TSFC=14.2 mg/N/s, W=60 t
    r, v, ld, tsfc, w = 3000 * NMI, 233.0, 21.4, 14.2e-6, 60_000.0
    expo = r * G0 * tsfc / (v * ld)
    expected = w * (1 - np.exp(-expo))
    assert block_fuel(r, v, ld, tsfc, w) == pytest.approx(expected)
    # sanity: a 3000-nmi cruise burns O(10%) of aircraft mass
    assert 0.05 * w < expected < 0.30 * w


def test_delta_block_fuel_sign_and_snowball():
    kw = dict(range_m=3000 * NMI, v=233.0, lift_drag=21.4, tsfc=14.2e-6, w_initial=60_000.0)
    d1 = delta_block_fuel(0.03, snowball=1.0, **kw)
    d2 = delta_block_fuel(0.03, snowball=1.35, **kw)
    assert d1 < 0.0  # positive PSC saves fuel
    assert d2 == pytest.approx(1.35 * d1, rel=1e-9)
    # first-order: dFuel/Fuel ~ -PSC (Breguet exponent small)
    assert d1 == pytest.approx(-0.03, rel=0.15)
    # a power *penalty* burns more fuel
    assert delta_block_fuel(-0.02, snowball=1.0, **kw) > 0.0


def test_breguet_invalid():
    with pytest.raises(ValueError):
        block_fuel(-1.0, 233.0, 21.4, 14.2e-6, 60_000.0)


def test_turboelectric_chain_product():
    ch = TurboelectricChain(eta_gen=0.96, eta_cable=0.99, eta_motor=0.96)
    assert ch.eta_elec == pytest.approx(0.96 * 0.99 * 0.96)
    assert 0.90 < ch.eta_elec < 0.93  # NASA reference band


def test_net_psc_limits():
    # Identical shaft powers + lossless chain -> no net change
    assert net_psc(1e6, 1e6, phi=0.3, eta_elec=1.0) == pytest.approx(0.0)
    # Lossless chain: net saving = phi * subsystem shaft saving
    val = net_psc(1e6, 0.9e6, phi=0.28, eta_elec=1.0)
    assert val == pytest.approx(0.28 * 0.1, rel=1e-9)
    # Chain losses can flip the benefit negative (Giannakakis caution)
    val_lossy = net_psc(1e6, 0.97e6, phi=0.28, eta_elec=0.90)
    assert val_lossy < 0.0


def test_net_psc_invalid():
    with pytest.raises(ValueError):
        net_psc(1e6, 9e5, phi=0.0)
    with pytest.raises(ValueError):
        net_psc(1e6, 9e5, phi=0.3, eta_elec=0.4)
