"""Turboelectric transmission chain (STARC-ABL architecture).

Underwing turbofans -> generator -> cable/inverter -> motor -> aft BLI fan.
NASA reference component efficiencies (Welstead & Felder 2016; NASA
TM-20210016661): eta_gen ~ 0.96, eta_cable/inverter ~ 0.98-0.996,
eta_motor ~ 0.96, giving a chain efficiency of ~0.90-0.93.

Net-benefit accounting (transparent power-flow form; equivalent to eq. 11 of
the proposal to first order):

    P_nonBLI  = P_shaft,rest + P_shaft,pod
    P_BLI     = P_shaft,rest + P_shaft,bli / eta_elec
    PSC_net   = 1 - P_BLI / P_nonBLI

where P_shaft,rest = P_shaft,pod (1 - phi)/phi is the (unchanged) shaft power
of the rest of the propulsion system, and phi is the fraction of total
propulsive power routed through the aft propulsor.  Setting eta_elec = 1
recovers a mechanically driven aft fan (D8-style architecture).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurboelectricChain:
    eta_gen: float = 0.96
    eta_cable: float = 0.99
    eta_motor: float = 0.96

    @property
    def eta_elec(self) -> float:
        return self.eta_gen * self.eta_cable * self.eta_motor


def net_psc(
    p_shaft_pod: float,
    p_shaft_bli: float,
    phi: float = 0.28,
    eta_elec: float = 0.92,
) -> float:
    """Net power-saving coefficient including the electrical-chain loss.

    Parameters
    ----------
    p_shaft_pod : shaft power of the podded reference aft propulsor [W].
    p_shaft_bli : shaft power of the BLI aft propulsor [W].
    phi : fraction of total propulsive power through the aft propulsor
        (STARC-ABL ~ 0.28; D8 ~ 1.0).
    eta_elec : end-to-end electrical transmission efficiency; 1.0 for a
        mechanically driven fan.
    """
    if not 0.0 < phi <= 1.0:
        raise ValueError("phi must be in (0, 1]")
    if not 0.5 < eta_elec <= 1.0:
        raise ValueError("eta_elec must be in (0.5, 1]")
    p_rest = p_shaft_pod * (1.0 - phi) / phi
    p_non_bli = p_rest + p_shaft_pod
    p_bli = p_rest + p_shaft_bli / eta_elec
    return 1.0 - p_bli / p_non_bli
