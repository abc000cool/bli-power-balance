"""Ingested-annulus stream-tube integrals.

The boundary layer arriving at the fan face is represented by a power-law
profile u(y) = V_e (y/delta)^(1/n) on the annulus around the tail-cone hub
(radius r_hub), evaluated at the freestream-static reference station
(edge velocity V_inf, p = p_inf) so that podded and BLI cases share one
defect ledger.  The profile thickness delta is sized so the annulus carries
exactly the momentum-defect flux delivered by the IBL + Squire-Young chain,
which keeps the transverse-curvature (thick-BL) effect on the tail cone
consistent with the drag bookkeeping.

All integrals are simple quadratures over y; area element dA = 2 pi (r+y) dy.

Defect integrals (per Drela 2009 / Hall 2017):
  mass flux          m_dot(yc)   = rho int u dA
  momentum defect    D_dot(yc)   = rho int u (V - u) dA
  KE-thickness flux  E_ket(yc)   = rho/2 int u (V^2 - u^2) dA   (theta* ledger)
  wake KE (E_a)      E_a(yc)     = rho/2 int u (V - u)^2 dA     (wake mixing loss)
  KE flux            E_dot(yc)   = rho/2 int u^3 dA

The Hall-2017 ingestion fraction is f_Phi = E_ket(yc) / E_ket(delta): the
fraction of the boundary layer's kinetic-energy defect (equivalently, of the
cumulative upstream surface dissipation footprint) captured by the fan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from blipb._compat import trapezoid
from blipb.atmosphere import CP_AIR, GAMMA, R_AIR

_NY = 400  # quadrature points across the profile


@dataclass(frozen=True)
class CaptureState:
    """Integral quantities of the captured stream tube."""

    y_c: float  # capture height above hub [m]
    m_dot: float  # mass flow [kg/s]
    momentum_defect: float  # rho int u (V-u) dA [N]
    ke_thickness_flux: float  # rho/2 int u (V^2-u^2) dA [W]
    ea_defect: float  # rho/2 int u (V-u)^2 dA [W]
    ke_flux_excess: float  # rho/2 int u (u^2 - V^2) dA [W] (negative for BL)
    pt_mass_avg: float  # mass-averaged total pressure [Pa]
    u_mass_avg: float  # mass-averaged velocity [m/s]
    f_phi: float  # ingested dissipation fraction


class AnnulusProfile:
    """Power-law boundary-layer profile on the tail-cone annulus."""

    def __init__(
        self,
        theta_eq: float,
        n_powerlaw: float,
        r_hub: float,
        v_edge: float,
        rho: float,
        p_static: float,
        tt: float,
    ) -> None:
        """
        Parameters
        ----------
        theta_eq : equivalent (freestream-edge) momentum thickness from the
            IBL + Squire-Young chain [m].  The profile delta is sized so the
            annulus momentum-defect flux equals 2 pi rho V^2 r_hub theta_eq.
        n_powerlaw : power-law exponent (u/V = (y/delta)^(1/n)).
        r_hub : tail-cone hub radius [m].
        v_edge : edge velocity (= V_inf at the reference station) [m/s].
        rho, p_static, tt : freestream static density/pressure, total temp.
        """
        self.n = float(n_powerlaw)
        self.r_hub = float(r_hub)
        self.v = float(v_edge)
        self.rho = float(rho)
        self.p = float(p_static)
        self.tt = float(tt)
        self.theta_eq = float(theta_eq)
        self.delta = self._size_delta()

    # -- profile ------------------------------------------------------------

    def u_of_y(self, y: np.ndarray, delta: float | None = None) -> np.ndarray:
        d = self.delta if delta is None else delta
        y = np.asarray(y, dtype=float)
        return self.v * np.clip(y / d, 0.0, 1.0) ** (1.0 / self.n)

    def _momentum_defect(self, delta: float) -> float:
        y = np.linspace(0.0, delta, _NY)
        u = self.u_of_y(y, delta)
        integrand = u * (self.v - u) * (self.r_hub + y)
        return float(2.0 * np.pi * self.rho * trapezoid(integrand, y))

    def _size_delta(self) -> float:
        """Solve for delta so the annulus defect matches the IBL defect."""
        target = 2.0 * np.pi * self.rho * self.v**2 * self.r_hub * self.theta_eq
        if target <= 0.0:
            return 1e-9

        def resid(delta: float) -> float:
            return self._momentum_defect(delta) - target

        # Planar first guess: theta/delta = n/((n+1)(n+2))
        d0 = self.theta_eq * (self.n + 1.0) * (self.n + 2.0) / self.n
        lo, hi = 1e-4 * d0, 10.0 * d0
        return float(brentq(resid, lo, hi, xtol=1e-10, rtol=1e-12))

    # -- capture integrals ----------------------------------------------------

    def capture(self, y_c: float) -> CaptureState:
        """Integral quantities for capture height y_c (may exceed delta)."""
        y_c = float(max(y_c, 1e-9))
        # Quadrature grid: full resolution across the boundary layer, plus a
        # coarse segment for any freestream capture beyond delta (where the
        # integrands are polynomials in y and trapezoid is exact).
        y_bl = np.linspace(0.0, min(y_c, self.delta), _NY)
        if y_c > self.delta:
            y = np.concatenate([y_bl, np.linspace(self.delta, y_c, 64)[1:]])
        else:
            y = y_bl
        u = self.u_of_y(y)
        area_w = 2.0 * np.pi * (self.r_hub + y)  # dA/dy

        rho = self.rho
        m_dot = rho * trapezoid(u * area_w, y)
        d_dot = rho * trapezoid(u * (self.v - u) * area_w, y)
        e_ket = 0.5 * rho * trapezoid(u * (self.v**2 - u**2) * area_w, y)
        e_a = 0.5 * rho * trapezoid(u * (self.v - u) ** 2 * area_w, y)
        e_exc = 0.5 * rho * trapezoid(u * (u**2 - self.v**2) * area_w, y)

        # Mass-averaged total pressure: adiabatic BL, Tt = Tt_inf uniform.
        ts = np.clip(self.tt - u**2 / (2.0 * CP_AIR), 50.0, None)
        mach = u / np.sqrt(GAMMA * R_AIR * ts)
        pt = self.p * (1.0 + 0.5 * (GAMMA - 1.0) * mach**2) ** (GAMMA / (GAMMA - 1.0))
        w = u * area_w  # mass-flux weight
        pt_avg = float(trapezoid(pt * w, y) / trapezoid(w, y))
        u_avg = float(trapezoid(u * w, y) / trapezoid(w, y))

        total_ket = self.ke_thickness_flux_total()
        f_phi = float(e_ket / total_ket) if total_ket > 0 else 0.0

        return CaptureState(
            y_c=y_c,
            m_dot=float(m_dot),
            momentum_defect=float(d_dot),
            ke_thickness_flux=float(e_ket),
            ea_defect=float(e_a),
            ke_flux_excess=float(e_exc),
            pt_mass_avg=pt_avg,
            u_mass_avg=u_avg,
            f_phi=min(f_phi, 1.0),
        )

    def ke_thickness_flux_total(self) -> float:
        """E_ket over the full boundary layer (y_c = delta)."""
        y = np.linspace(0.0, self.delta, _NY)
        u = self.u_of_y(y)
        area_w = 2.0 * np.pi * (self.r_hub + y)
        return float(0.5 * self.rho * trapezoid(u * (self.v**2 - u**2) * area_w, y))

    def ea_defect_total(self) -> float:
        """Wake-mixing KE defect E_a over the full boundary layer."""
        return self.capture(self.delta).ea_defect

    def momentum_defect_total(self) -> float:
        return self.capture(self.delta).momentum_defect

    def solve_capture_for_fphi(self, f_phi: float) -> CaptureState:
        """Find the capture height that ingests dissipation fraction f_phi."""
        f_phi = float(np.clip(f_phi, 1e-4, 1.0))
        if f_phi >= 0.9999:
            return self.capture(self.delta)

        def resid(y_c: float) -> float:
            return self.capture(y_c).f_phi - f_phi

        y_c = brentq(resid, 1e-6 * self.delta, self.delta, xtol=1e-12)
        return self.capture(float(y_c))

