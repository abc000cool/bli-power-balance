"""One-dimensional compressible fan with polytropic efficiency.

Given the (possibly boundary-layer-degraded, mass-averaged) inlet total state
(pt_in, Tt_in), a fan total-pressure ratio FPR and polytropic efficiency
eta_pol, the model returns the fully-expanded jet velocity and the specific
shaft work:

    pt_out = FPR * pt_in
    Tt_out = Tt_in * FPR^((gamma-1)/(gamma eta_pol))
    V_j    = sqrt(2 cp Tt_out [1 - (p_inf/pt_out)^((gamma-1)/gamma)])
    w_sh   = cp (Tt_out - Tt_in)

A key property of BLI exploited here: an adiabatic boundary layer preserves
total temperature while destroying total pressure, so the BLI fan inherits
Tt_in = Tt_inf but a mass-averaged pt_in < pt_inf, which automatically
yields the lower jet velocity that drives the power saving.

The distortion penalty enters as an explicit decrement of polytropic
efficiency, delta_eta = k_dist * f_phi (Fidalgo/Hall 2012, Gray 2018 class
values: ~1.5% at f_phi = 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blipb.atmosphere import CP_AIR, GAMMA


@dataclass(frozen=True)
class FanResult:
    v_jet: float  # fully expanded jet velocity [m/s]
    w_shaft: float  # specific shaft work [J/kg]
    pt_out: float  # jet total pressure [Pa]
    tt_out: float  # jet total temperature [K]
    eta_pol_eff: float  # effective polytropic efficiency used


def fan_jet(
    pt_in: float,
    tt_in: float,
    fpr: float,
    p_inf: float,
    eta_pol: float = 0.92,
    delta_eta_distortion: float = 0.0,
) -> FanResult:
    """Fan + fully-expanded nozzle, returning jet velocity and shaft work."""
    if fpr < 1.0:
        raise ValueError("FPR must be >= 1")
    eta = eta_pol - delta_eta_distortion
    if not 0.5 < eta <= 1.0:
        raise ValueError(f"effective polytropic efficiency {eta} outside (0.5, 1]")

    g = GAMMA
    pt_out = fpr * pt_in
    if pt_out <= p_inf:
        raise ValueError(
            f"jet total pressure {pt_out:.0f} Pa <= ambient {p_inf:.0f} Pa; "
            "inlet too degraded for this FPR"
        )
    tt_out = tt_in * fpr ** ((g - 1.0) / (g * eta))
    v_jet = float(np.sqrt(2.0 * CP_AIR * tt_out * (1.0 - (p_inf / pt_out) ** ((g - 1.0) / g))))
    w_shaft = float(CP_AIR * (tt_out - tt_in))
    return FanResult(v_jet=v_jet, w_shaft=w_shaft, pt_out=pt_out, tt_out=tt_out, eta_pol_eff=eta)
