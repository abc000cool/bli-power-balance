"""Breguet cruise fuel burn and BLI block-fuel delta.

    R = (V / (g TSFC)) (L/D) ln(W_i / W_f)

At fixed range, payload and L/D, the fuel fraction is

    W_fuel / W_i = 1 - exp(-R g TSFC / (V L/D))

A power saving PSC_net (net of transmission losses, already weighted by the
BLI power fraction phi -- see propulsion.turboelectric.net_psc) reduces the
effective TSFC multiplicatively: at fixed thrust and thermal efficiency the
fuel flow scales with the required mechanical power, so
TSFC_eff = TSFC (1 - PSC_net).  A snowball factor (empty-weight resizing
cascade, 1.2-1.5 in conceptual design; SPEC default 1.35) multiplies the
resulting block-fuel delta for rubber-airframe studies.
"""

from __future__ import annotations

import numpy as np

from blipb.atmosphere import G0

NMI = 1852.0  # m


def block_fuel(
    range_m: float,
    v: float,
    lift_drag: float,
    tsfc: float,
    w_initial: float,
) -> float:
    """Cruise fuel burn [kg] for a mission flown at constant V, L/D, TSFC.

    Parameters
    ----------
    range_m : still-air range [m].
    v : cruise speed [m/s].
    lift_drag : cruise lift-to-drag ratio.
    tsfc : thrust-specific fuel consumption [kg/(N s)].
    w_initial : start-of-cruise mass [kg].
    """
    if min(range_m, v, lift_drag, tsfc, w_initial) <= 0:
        raise ValueError("all Breguet inputs must be positive")
    exponent = range_m * G0 * tsfc / (v * lift_drag)
    return float(w_initial * (1.0 - np.exp(-exponent)))


def delta_block_fuel(
    psc_net: float,
    range_m: float,
    v: float,
    lift_drag: float,
    tsfc: float,
    w_initial: float,
    snowball: float = 1.35,
) -> float:
    """Relative block-fuel change from a net power saving (negative = saving).

    Returns (fuel_BLI - fuel_ref) / fuel_ref, snowball-amplified.
    """
    ref = block_fuel(range_m, v, lift_drag, tsfc, w_initial)
    bli = block_fuel(range_m, v, lift_drag, tsfc * (1.0 - psc_net), w_initial)
    return float(snowball * (bli - ref) / ref)
