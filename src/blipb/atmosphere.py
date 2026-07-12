"""International Standard Atmosphere and cruise flight state.

Only the troposphere and lower stratosphere (h <= 20 km) are implemented,
which covers every condition in the study envelope (SPEC.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Gas properties (dry air)
GAMMA = 1.4
R_AIR = 287.05287  # J/(kg K)
CP_AIR = GAMMA * R_AIR / (GAMMA - 1.0)  # 1004.7 J/(kg K)
G0 = 9.80665  # m/s^2

# ISA sea-level state
T0 = 288.15  # K
P0 = 101_325.0  # Pa
LAPSE = 0.0065  # K/m, troposphere
H_TROPOPAUSE = 11_000.0  # m
T_TROPOPAUSE = T0 - LAPSE * H_TROPOPAUSE  # 216.65 K

# Sutherland's law constants
MU_REF = 1.716e-5  # Pa s at T_ref
T_REF_MU = 273.15  # K
S_MU = 110.4  # K


def sutherland(T: float) -> float:
    """Dynamic viscosity of air [Pa s] via Sutherland's law."""
    return MU_REF * (T / T_REF_MU) ** 1.5 * (T_REF_MU + S_MU) / (T + S_MU)


@dataclass(frozen=True)
class Atmosphere:
    """Static atmospheric state at a geopotential altitude."""

    h: float  # m
    T: float  # K
    p: float  # Pa
    rho: float  # kg/m^3
    a: float  # speed of sound, m/s
    mu: float  # dynamic viscosity, Pa s

    @property
    def nu(self) -> float:
        """Kinematic viscosity [m^2/s]."""
        return self.mu / self.rho


def isa(h: float) -> Atmosphere:
    """ISA state at geopotential altitude h [m], valid for h <= 20 km."""
    if h < 0 or h > 20_000.0:
        raise ValueError(f"altitude {h} m outside implemented ISA range [0, 20 km]")
    if h <= H_TROPOPAUSE:
        T = T0 - LAPSE * h
        p = P0 * (T / T0) ** (G0 / (R_AIR * LAPSE))
    else:
        T = T_TROPOPAUSE
        p_tp = P0 * (T_TROPOPAUSE / T0) ** (G0 / (R_AIR * LAPSE))
        p = p_tp * np.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * T_TROPOPAUSE))
    rho = p / (R_AIR * T)
    a = float(np.sqrt(GAMMA * R_AIR * T))
    return Atmosphere(h=h, T=T, p=p, rho=rho, a=a, mu=sutherland(T))


@dataclass(frozen=True)
class FlightState:
    """Cruise flight state: freestream static + stagnation quantities."""

    mach: float
    altitude: float  # m
    atm: Atmosphere = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atm", isa(self.altitude))

    @property
    def V(self) -> float:
        """Freestream velocity [m/s]."""
        return self.mach * self.atm.a

    @property
    def q(self) -> float:
        """Freestream dynamic pressure (incompressible definition) [Pa]."""
        return 0.5 * self.atm.rho * self.V**2

    @property
    def Tt(self) -> float:
        """Freestream total temperature [K]."""
        return self.atm.T * (1.0 + 0.5 * (GAMMA - 1.0) * self.mach**2)

    @property
    def pt(self) -> float:
        """Freestream total pressure [Pa]."""
        return self.atm.p * (1.0 + 0.5 * (GAMMA - 1.0) * self.mach**2) ** (
            GAMMA / (GAMMA - 1.0)
        )

    def reynolds(self, length: float) -> float:
        """Reynolds number based on a reference length [m]."""
        return self.V * length / self.atm.nu
