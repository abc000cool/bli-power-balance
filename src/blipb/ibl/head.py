"""Head's entrainment method for turbulent boundary layers, axisymmetric form.

State vector y = [theta, H1] integrated in x with scipy LSODA:

  dtheta/dx = C_f/2 - (H + 2 - M_e^2) (theta/U_e) dU_e/dx - (theta/r) dr/dx
  d(theta H1)/dx = F(H1) - theta H1 (r'/r + U_e'/U_e)

The r-terms are the thin-boundary-layer axisymmetric (Mangler-equivalent)
contributions; setting r = const recovers the planar equations used for
flat-plate verification.  C_f uses Ludwieg-Tillmann scaled by the
adiabatic-wall compressibility factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from blipb.ibl import closures


@dataclass(frozen=True)
class HeadResult:
    x: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    H1: np.ndarray
    cf: np.ndarray
    re_theta: np.ndarray
    separated: bool  # True if H exceeded the separation proxy anywhere
    x_sep: float  # separation station (nan if attached throughout)


def solve_head(
    x: np.ndarray,
    ue: np.ndarray,
    nu: float,
    theta0: float,
    H0: float = 1.35,
    r: np.ndarray | None = None,
    mach_e: np.ndarray | None = None,
) -> HeadResult:
    """Integrate Head's method from x[0] to x[-1].

    Parameters
    ----------
    x : stations (monotone increasing); x[0] is the transition station.
    ue : edge velocity at stations.
    nu : kinematic viscosity.
    theta0 : initial momentum thickness (from the laminar solution).
    H0 : initial turbulent shape factor.
    r : body radius at stations (None = planar).
    mach_e : edge Mach number at stations (None = incompressible).
    """
    x = np.asarray(x, dtype=float)
    ue = np.asarray(ue, dtype=float)
    rr = np.ones_like(x) if r is None else np.asarray(r, dtype=float)
    me = np.zeros_like(x) if mach_e is None else np.asarray(mach_e, dtype=float)

    due_dx = np.gradient(ue, x)
    dr_dx = np.gradient(rr, x)

    def interp(arr: np.ndarray):
        return lambda s: np.interp(s, x, arr)

    f_ue, f_due, f_r, f_drdx, f_me = (
        interp(ue),
        interp(due_dx),
        interp(rr),
        interp(dr_dx),
        interp(me),
    )

    def rhs(s: float, y: np.ndarray) -> list[float]:
        theta, H1 = max(y[0], 1e-10), max(y[1], 3.35)
        ue_s, due_s = f_ue(s), f_due(s)
        r_s, drdx_s = max(f_r(s), 1e-9), f_drdx(s)
        me_s = f_me(s)
        H = float(closures.head_H(H1))
        re_theta = ue_s * theta / nu
        cf = float(
            closures.ludwieg_tillmann_cf(H, re_theta)
            * closures.cf_compressibility_factor(me_s)
        )
        dtheta = (
            0.5 * cf
            - (H + 2.0 - me_s**2) * theta / ue_s * due_s
            - theta / r_s * drdx_s
        )
        F = float(closures.head_entrainment(H1))
        dthetaH1 = F - theta * H1 * (drdx_s / r_s + due_s / ue_s)
        dH1 = (dthetaH1 - dtheta * H1) / theta
        return [dtheta, dH1]

    H1_0 = float(closures.head_H1(H0))
    sol = solve_ivp(
        rhs,
        (x[0], x[-1]),
        [theta0, H1_0],
        method="LSODA",
        t_eval=x,
        rtol=1e-7,
        atol=[1e-12, 1e-9],
    )
    if not sol.success:  # pragma: no cover - LSODA failure is exceptional
        raise RuntimeError(f"Head integration failed: {sol.message}")

    theta = sol.y[0]
    H1 = np.clip(sol.y[1], 3.35, None)
    H = closures.head_H(H1)
    re_theta = ue * theta / nu
    cf = closures.ludwieg_tillmann_cf(H, re_theta) * closures.cf_compressibility_factor(me)

    sep_mask = H >= closures.H_TURB_SEP
    separated = bool(sep_mask.any())
    x_sep = float(x[sep_mask][0]) if separated else float("nan")

    return HeadResult(
        x=x,
        theta=theta,
        H=H,
        H1=H1,
        cf=cf,
        re_theta=re_theta,
        separated=separated,
        x_sep=x_sep,
    )
