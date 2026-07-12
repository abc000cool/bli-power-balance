"""Parametric axisymmetric fuselage geometry and inviscid edge velocity.

The fuselage is a body of revolution: elliptic nose, cylindrical midbody,
cubic-Hermite tail-cone contraction ending at a finite tail radius (the BLI
fan hub station).  The inviscid edge-velocity distribution U_e(x) is computed
with the classical von Karman axial source-line method: a line distribution
of sources on the axis with local strength q(x) = V_inf dS/dx (S = pi r^2),
superposed with the freestream and evaluated on the body surface.  An
optional Goethert-type factor 1/beta amplifies the perturbation velocities
to approximate subsonic compressibility (beta = sqrt(1 - M^2)).

This is the standard low-order treatment for slender bodies; it produces the
expected acceleration over the nose shoulder, U_e ~ V_inf over the midbody,
and a mild adverse gradient on the tail cone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blipb._compat import trapezoid


@dataclass(frozen=True)
class Fuselage:
    """STARC-ABL-class body of revolution (dimensions in metres)."""

    length: float = 37.0
    radius_max: float = 1.88  # D = 3.76 m
    nose_frac: float = 0.15  # elliptic nose length / L
    tail_frac: float = 0.30  # tail-cone length / L
    tail_radius_frac: float = 0.30  # r_tail / r_max (fan hub radius)

    @property
    def r_tail(self) -> float:
        return self.tail_radius_frac * self.radius_max

    @property
    def fineness(self) -> float:
        return self.length / (2.0 * self.radius_max)

    def radius(self, x: np.ndarray) -> np.ndarray:
        """Local body radius r(x), x in [0, L]."""
        x = np.asarray(x, dtype=float)
        L, R = self.length, self.radius_max
        Ln = self.nose_frac * L
        Lt = self.tail_frac * L
        xt = L - Lt
        r = np.full_like(x, R)
        # Elliptic nose: r = R sqrt(1 - ((Ln - x)/Ln)^2)
        nose = x < Ln
        arg = 1.0 - ((Ln - x[nose]) / Ln) ** 2
        r[nose] = R * np.sqrt(np.clip(arg, 0.0, None))
        # Cubic Hermite tail: zero slope at both ends, contracting to r_tail
        tail = x > xt
        xi = (x[tail] - xt) / Lt
        blend = 3.0 * xi**2 - 2.0 * xi**3
        r[tail] = R + (self.r_tail - R) * blend
        return r

    def area(self, x: np.ndarray) -> np.ndarray:
        """Cross-sectional area S(x) = pi r^2."""
        return np.pi * self.radius(x) ** 2

    def wetted_area(self, n: int = 2000) -> float:
        """Wetted surface area by quadrature of 2 pi r ds."""
        x = np.linspace(0.0, self.length, n)
        r = self.radius(x)
        drdx = np.gradient(r, x)
        ds = np.sqrt(1.0 + drdx**2)
        return float(trapezoid(2.0 * np.pi * r * ds, x))

    def grid(self, n: int = 400) -> np.ndarray:
        """Streamwise grid clustered at nose and tail (cosine spacing)."""
        beta = np.linspace(0.0, np.pi, n)
        return self.length * 0.5 * (1.0 - np.cos(beta))

    def edge_velocity(
        self,
        x: np.ndarray,
        v_inf: float,
        mach: float = 0.0,
        n_sources: int = 600,
    ) -> np.ndarray:
        """Inviscid surface velocity U_e(x) by the axial source-line method.

        Parameters
        ----------
        x : surface stations [m]
        v_inf : freestream velocity [m/s]
        mach : freestream Mach number; perturbations are amplified by
            1/beta (Goethert-type subsonic correction). Pass 0 to disable.
        n_sources : number of axial source panels.
        """
        x = np.asarray(x, dtype=float)
        L = self.length
        # Source strengths from area growth: q dxi = V dS
        xi_edges = np.linspace(0.0, L, n_sources + 1)
        xi_mid = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        dS = np.diff(self.area(xi_edges))
        q = v_inf * dS  # source strength integrated over each panel [m^3/s]

        r_surf = self.radius(x)
        # Keep evaluation off the axis at the very nose tip
        r_eval = np.maximum(r_surf, 1e-3 * self.radius_max)
        dx = x[:, None] - xi_mid[None, :]
        d3 = (dx**2 + r_eval[:, None] ** 2) ** 1.5
        u_x = (q[None, :] * dx / (4.0 * np.pi * d3)).sum(axis=1)
        u_r = (q[None, :] * r_eval[:, None] / (4.0 * np.pi * d3)).sum(axis=1)

        beta = np.sqrt(max(1.0 - mach**2, 1e-6))
        u_x = u_x / beta
        u_r = u_r / beta

        ue = np.sqrt((v_inf + u_x) ** 2 + u_r**2)

        # The source-line model diverges at the blunt nose tip, where the
        # body is not slender and the true flow stagnates.  Replace the
        # inner-nose solution with a stagnation ramp that rises from 0 and
        # joins the source-line value (with zero ramp slope) at x_b.
        x_b = 0.3 * self.nose_frac * L
        i_b = int(np.searchsorted(x, x_b))
        if 0 < i_b < len(x):
            ue_b = ue[i_b]
            xi = np.clip(x[:i_b] / x_b, 0.0, 1.0)
            ue[:i_b] = ue_b * xi * (2.0 - xi)
        ue = np.clip(ue, 1e-3 * v_inf, None)
        return ue

