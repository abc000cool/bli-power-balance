"""Hall et al. (2017) style decomposition and effective parameters.

Hall, Huang, Uranga, Greitzer, Drela & Sato (J. Propulsion & Power 33(5),
2017) decompose the D8 BLI benefit into jet-, surface- and wake-dissipation
contributions (measured: 5.2% + 2.4% + 0.6% = 8.2%).  This module maps a
ComparatorResult onto that decomposition and extracts the effective
parameters (f_Phi, eta_fill) of the compact Smith/Hall estimator.

The low-order comparator resolves the jet and wake terms exactly (they are
the two terms of the exact ledger identity); the *surface* term -- the
reduction of fuselage surface dissipation caused by the fan's suction
lowering edge velocities on the tail cone -- requires coupled aero-propulsive
analysis and is identically zero here.  This is a declared limitation
(paper section 6), and is the expected primary diagnosis for any shortfall
against the D8 measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from blipb.powerbalance.comparator import ComparatorResult


@dataclass(frozen=True)
class HallDecomposition:
    """PSC contributions by dissipation mechanism (fractions of P_K,pod)."""

    jet: float  # reduced jet-mixing dissipation
    wake: float  # ingested (avoided) wake-mixing dissipation
    surface: float  # fan-suction surface-dissipation change (0 in low-order)
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "jet": self.jet,
            "wake": self.wake,
            "surface": self.surface,
            "total": self.total,
        }


def decompose(result: ComparatorResult) -> HallDecomposition:
    """Split the PSC into Hall-2017 mechanism contributions."""
    pk = result.pk_pod
    jet = result.delta_phi_jet / pk
    wake = result.delta_phi_wake / pk
    return HallDecomposition(jet=jet, wake=wake, surface=0.0, total=jet + wake)


def effective_fill_factor(result: ComparatorResult) -> float:
    """Effective eta_fill of the compact estimator, back-computed.

    Defined as the achieved saving relative to the ideal saving available
    from the ingested stream (jet-mixing recovery of the podded reference
    plus the ingested wake KE), bounded to [0, 1] for reporting.
    """
    ideal = result.phi_jet_pod + result.delta_phi_wake
    if ideal <= 0:
        return 0.0
    achieved = result.pk_pod - result.pk_bli
    return float(min(max(achieved / ideal, 0.0), 1.0))
