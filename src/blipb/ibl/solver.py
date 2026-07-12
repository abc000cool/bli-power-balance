"""Fuselage boundary-layer orchestrator: Thwaites -> transition -> Head -> wake.

Produces the trailing-edge (fan-face) boundary-layer state and the cumulative
surface dissipation used by the power-balance ledger.  The cumulative surface
dissipation follows from the kinetic-energy thickness identity (Drela 2009
eq. 5, axisymmetric form):

    Phi_surf(x) = pi rho r(x) U_e(x)^3 theta*(x)

with theta* = H*(H, Re_theta) theta reconstructed from the XFOIL-family
closures.  This is exactly the "the KE thickness is the ledger of upstream
dissipation" property that makes the power balance work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blipb.atmosphere import FlightState
from blipb.geometry import Fuselage
from blipb.ibl import closures
from blipb.ibl.head import solve_head
from blipb.ibl.thwaites import solve_thwaites
from blipb.ibl.wake import WakeResult, squire_young


@dataclass(frozen=True)
class BLSolution:
    """Boundary-layer solution along the fuselage plus derived quantities."""

    x: np.ndarray
    ue: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    cf: np.ndarray
    theta_star: np.ndarray
    x_tr: float
    # Trailing-edge (fan-face) state
    theta_te: float
    h_te: float
    ue_te: float
    r_te: float
    # Ledger quantities
    phi_surf: float  # cumulative surface dissipation at TE [W]
    wake: WakeResult
    drag: float  # far-wake profile drag [N]
    # Validity
    separated: bool
    x_sep: float

    @property
    def valid(self) -> bool:
        return not self.separated


def solve_fuselage_bl(
    fuselage: Fuselage,
    flight: FlightState,
    x_tr_frac: float = 0.05,
    n_points: int = 400,
) -> BLSolution:
    """Solve the axisymmetric IBL chain over the fuselage.

    Parameters
    ----------
    fuselage : body geometry.
    flight : cruise state.
    x_tr_frac : forced transition location x_tr / L.  The Michel criterion
        is also evaluated; transition occurs at whichever is earlier
        (at Re_L ~ 1.5e8 the forced location almost always governs).
    n_points : streamwise resolution.
    """
    atm = flight.atm
    nu = atm.nu
    v_inf = flight.V

    x = fuselage.grid(n_points)
    r = fuselage.radius(x)
    ue = fuselage.edge_velocity(x, v_inf, mach=flight.mach)
    mach_e = ue / atm.a  # edge Mach from kinematic scaling (low-order)

    # --- Laminar segment (Thwaites, Mangler form) --------------------------
    lam = solve_thwaites(x, ue, nu, r=r)

    # Transition: forced location or Michel, whichever first
    re_x = ue * x / nu
    michel = closures.michel_transition(lam.re_theta, re_x)
    x_tr_forced = x_tr_frac * fuselage.length
    idx_forced = int(np.searchsorted(x, x_tr_forced))
    idx_michel = int(np.argmax(michel)) if michel.any() else len(x) - 1
    i_tr = max(1, min(idx_forced, idx_michel))
    x_tr = float(x[i_tr])

    # --- Turbulent segment (Head, axisymmetric) ----------------------------
    theta0 = float(lam.theta[i_tr])
    turb = solve_head(
        x[i_tr:],
        ue[i_tr:],
        nu,
        theta0=theta0,
        H0=1.35,
        r=r[i_tr:],
        mach_e=mach_e[i_tr:],
    )

    # --- Assemble full-body arrays -----------------------------------------
    theta = np.concatenate([lam.theta[:i_tr], turb.theta])
    H = np.concatenate([lam.H[:i_tr], turb.H])
    cf = np.concatenate([lam.cf[:i_tr], turb.cf])
    re_theta = ue * theta / nu
    hstar = np.concatenate(
        [
            closures.hstar_laminar(lam.H[:i_tr]),
            closures.hstar_turbulent(turb.H, re_theta[i_tr:]),
        ]
    )
    theta_star = hstar * theta

    theta_te = float(theta[-1])
    h_te = float(H[-1])
    ue_te = float(ue[-1])
    r_te = float(r[-1])

    # Cumulative surface dissipation at the TE (KE-thickness identity)
    phi_surf = float(np.pi * atm.rho * r_te * ue_te**3 * theta_star[-1])

    wk = squire_young(theta_te, h_te, ue_te, v_inf, atm.rho, r_te)

    return BLSolution(
        x=x,
        ue=ue,
        r=r,
        theta=theta,
        H=H,
        cf=cf,
        theta_star=theta_star,
        x_tr=x_tr,
        theta_te=theta_te,
        h_te=h_te,
        ue_te=ue_te,
        r_te=r_te,
        phi_surf=phi_surf,
        wake=wk,
        drag=wk.drag,
        separated=turb.separated,
        x_sep=turb.x_sep,
    )
