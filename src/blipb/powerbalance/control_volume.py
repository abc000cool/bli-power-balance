"""Canonical control volume shared by the podded and BLI cases.

The single most common BLI bookkeeping error is comparing two cases whose
control volumes differ (jet clipped before mixing, different downstream
extents, lateral boundaries at p != p_inf).  Here the CV is a first-class
frozen object; the comparator asserts that both cases carry the *same*
instance, making an inconsistent comparison a hard error rather than a
silent bias.

The CV extends to a Trefftz plane far enough downstream that wake and jet
static pressure have relaxed to p_inf, with lateral boundaries at freestream
static conditions.  All ledger terms (P_K, E_a, Phi) are evaluated against
this common reference state (V_inf, p_inf, rho_inf).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlVolume:
    """Reference state and bookkeeping conventions for one comparison."""

    v_inf: float  # freestream velocity [m/s]
    rho: float  # freestream density [kg/m^3]
    p_inf: float  # freestream static pressure [Pa]
    tt_inf: float  # freestream total temperature [K]
    pt_inf: float  # freestream total pressure [Pa]
    # Bookkeeping conventions (frozen per SPEC.md)
    ingestion_convention: str = "dissipation_fraction"  # f_Phi, Hall 2017
    comparison_rule: str = "equal_net_streamwise_force"

    def __post_init__(self) -> None:
        if self.v_inf <= 0 or self.rho <= 0 or self.p_inf <= 0:
            raise ValueError("ControlVolume requires positive freestream state")
