"""Thwaites' laminar integral method, axisymmetric (Mangler) form.

Planar Thwaites:      theta^2 U_e^6           = 0.45 nu int_0^x U_e^5 dxi
Axisymmetric Mangler: theta^2 U_e^6 r^2       = 0.45 nu int_0^x U_e^5 r^2 dxi

Setting r = const recovers the planar form, so a single implementation
covers both (pass r = ones for 2-D verification cases).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from blipb.ibl import closures


@dataclass(frozen=True)
class ThwaitesResult:
    x: np.ndarray
    theta: np.ndarray
    H: np.ndarray
    cf: np.ndarray
    re_theta: np.ndarray
    lam: np.ndarray  # pressure-gradient parameter lambda
    separated: np.ndarray  # bool mask, lambda < -0.09


def solve_thwaites(
    x: np.ndarray,
    ue: np.ndarray,
    nu: float,
    r: np.ndarray | None = None,
) -> ThwaitesResult:
    """March Thwaites' method along stations x with edge velocity ue.

    Parameters
    ----------
    x : monotonically increasing arc-length stations, x[0] may be 0.
    ue : edge velocity at the stations (ue > 0 except possibly x=0).
    nu : kinematic viscosity.
    r : body radius at the stations for the Mangler (axisymmetric) form;
        None means planar flow.
    """
    x = np.asarray(x, dtype=float)
    ue = np.clip(np.asarray(ue, dtype=float), 1e-9, None)
    rr = np.ones_like(x) if r is None else np.clip(np.asarray(r, dtype=float), 1e-9, None)

    integrand = ue**5 * rr**2
    integral = cumulative_trapezoid(integrand, x, initial=0.0)
    theta2 = 0.45 * nu * integral / (ue**6 * rr**2)
    # Stagnation-point limit at x = 0: theta^2 = 0.075 nu / (dUe/dx)
    if theta2[0] <= 0.0:
        due = np.gradient(ue, x)[0]
        theta2[0] = 0.075 * nu / max(due, 1e-9) if due > 0 else theta2[1]
    theta = np.sqrt(np.clip(theta2, 1e-20, None))

    due_dx = np.gradient(ue, x)
    lam = theta2 / nu * due_dx
    lam = np.clip(lam, -0.25, 0.25)
    H = closures.thwaites_H(lam)
    re_theta = ue * theta / nu
    cf = 2.0 * closures.thwaites_l(lam) / np.clip(re_theta, 1e-9, None)
    separated = lam <= closures.LAMBDA_SEP

    return ThwaitesResult(
        x=x, theta=theta, H=H, cf=cf, re_theta=re_theta, lam=lam, separated=separated
    )
