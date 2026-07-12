"""Integral boundary-layer closure correlations.

Laminar:  Thwaites (1949) with the Cebeci-Bradshaw two-branch fits for
          H(lambda) and the shear correlate l(lambda).
Turbulent: Head (1958) entrainment F(H1), Cebeci-Bradshaw H1(H) and its
          inverse, Ludwieg-Tillmann skin friction.
Kinetic-energy shape factor H* = theta*/theta: Drela's XFOIL closures
          (laminar: Drela & Giles 1987 eq. 26; turbulent: XFOIL 6.9 form).
Compressibility: adiabatic-wall turbulent C_f ratio, the widely used
          (1 + 0.144 M^2)^-0.65 correlation (equivalent to Van Driest II
          within ~1% for M < 1, adiabatic wall).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Thwaites laminar closures (Cebeci & Bradshaw 1977 two-branch fits)
# --------------------------------------------------------------------------

LAMBDA_SEP = -0.09  # laminar separation


def thwaites_H(lam: np.ndarray) -> np.ndarray:
    """Laminar shape factor H(lambda)."""
    lam = np.clip(np.asarray(lam, dtype=float), LAMBDA_SEP, 0.25)
    H = np.where(
        lam >= 0.0,
        2.61 - 3.75 * lam + 5.24 * lam**2,
        2.088 + 0.0731 / (lam + 0.14),
    )
    return H


def thwaites_l(lam: np.ndarray) -> np.ndarray:
    """Laminar shear correlate l(lambda); C_f = 2 l / Re_theta."""
    lam = np.clip(np.asarray(lam, dtype=float), LAMBDA_SEP, 0.25)
    lpos = 0.22 + 1.57 * lam - 1.8 * lam**2
    lneg = 0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107)
    return np.where(lam >= 0.0, lpos, lneg)


# --------------------------------------------------------------------------
# Head turbulent closures
# --------------------------------------------------------------------------

H_TURB_SEP = 2.4  # turbulent separation proxy (validity flag threshold)


def head_H1(H: np.ndarray) -> np.ndarray:
    """Cebeci-Bradshaw correlation H1(H)."""
    H = np.asarray(H, dtype=float)
    H = np.clip(H, 1.05, 3.0)
    return np.where(
        H <= 1.6,
        3.3 + 0.8234 * (H - 1.1) ** -1.287,
        3.3 + 1.5501 * (H - 0.6778) ** -3.064,
    )


def head_H(H1: np.ndarray) -> np.ndarray:
    """Inverse correlation H(H1) (Moran 1984 fits)."""
    H1 = np.asarray(H1, dtype=float)
    H1 = np.clip(H1, 3.35, 50.0)
    return np.where(
        H1 >= 5.3,
        1.1 + 0.86 * (H1 - 3.3) ** -0.777,
        0.6778 + 1.1536 * (H1 - 3.3) ** -0.326,
    )


def head_entrainment(H1: np.ndarray) -> np.ndarray:
    """Head's entrainment function F(H1) = 0.0306 (H1 - 3)^-0.6169."""
    H1 = np.asarray(H1, dtype=float)
    return 0.0306 * np.clip(H1 - 3.0, 1e-3, None) ** -0.6169


def ludwieg_tillmann_cf(H: np.ndarray, re_theta: np.ndarray) -> np.ndarray:
    """Turbulent skin friction C_f(H, Re_theta), Ludwieg & Tillmann (1950)."""
    H = np.asarray(H, dtype=float)
    re_theta = np.clip(np.asarray(re_theta, dtype=float), 20.0, None)
    return 0.246 * 10.0 ** (-0.678 * H) * re_theta**-0.268


# --------------------------------------------------------------------------
# Kinetic-energy shape factor H* = theta* / theta (Drela closures)
# --------------------------------------------------------------------------


def hstar_laminar(H: np.ndarray) -> np.ndarray:
    """Laminar H*(H), Drela & Giles (1987)."""
    H = np.asarray(H, dtype=float)
    return np.where(
        H < 4.0,
        1.515 + 0.076 * (4.0 - H) ** 2 / H,
        1.515 + 0.040 * (H - 4.0) ** 2 / H,
    )


def hstar_turbulent(H: np.ndarray, re_theta: np.ndarray) -> np.ndarray:
    """Turbulent H*(H, Re_theta), XFOIL closure (Drela 1989)."""
    H = np.asarray(H, dtype=float)
    re_theta = np.clip(np.asarray(re_theta, dtype=float), 400.0, None)
    H0 = 3.0 + 400.0 / re_theta
    lower = 1.505 + 4.0 / re_theta
    below = lower + (0.165 - 1.6 / np.sqrt(re_theta)) * np.clip(H0 - H, 0.0, None) ** 1.6 / H
    ln_rt = np.log(re_theta)
    dH = np.clip(H - H0, 0.0, None)
    above = lower + dH**2 * (0.04 / H + 0.007 * ln_rt / (dH + 4.0 / ln_rt) ** 2)
    return np.where(H < H0, below, above)


# --------------------------------------------------------------------------
# Compressibility and transition
# --------------------------------------------------------------------------


def cf_compressibility_factor(mach_e: np.ndarray) -> np.ndarray:
    """Adiabatic-wall turbulent C_f ratio  C_f / C_f,inc = (1+0.144 M^2)^-0.65.

    Matches Van Driest II within ~1% for M_e < 1 (adiabatic wall, air).
    """
    mach_e = np.asarray(mach_e, dtype=float)
    return (1.0 + 0.144 * mach_e**2) ** -0.65


def michel_transition(re_theta: np.ndarray, re_x: np.ndarray) -> np.ndarray:
    """Michel (1951) criterion: transition where Re_theta exceeds the bound."""
    re_x = np.clip(np.asarray(re_x, dtype=float), 1e3, None)
    bound = 1.174 * (1.0 + 22_400.0 / re_x) * re_x**0.46
    return np.asarray(re_theta, dtype=float) >= bound
