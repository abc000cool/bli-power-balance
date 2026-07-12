"""Smith (1993) wake-ingestion closed forms.

These are independent algebraic paths to the BLI benefit used as analytic
verification targets for the comparator (SPEC.md target V3).  They follow
Smith's ideal actuator-disk wake-ingestion analysis expressed in the power
balance language.

Full ingestion of a *uniform* wake (velocity ratio w = V_w / V_inf), the
self-propelled condition (jet exactly refills the wake to V_inf), compared
with a podded propulsor producing the equivalent force at equal mass flow:

    PSC = 2 (1 - w) / (3 - w)

which recovers 0 as w -> 1 (no wake) and rises monotonically as the wake
deepens -- Smith's classic result that wake ingestion benefit grows with
wake depth.

For an arbitrary (non-uniform) fully-ingested profile, the same ledger gives
a closed form in terms of the profile's defect integrals; see
``psc_self_propelled_profile``.
"""

from __future__ import annotations

import numpy as np

from blipb.powerbalance.streamtube import AnnulusProfile


def psc_uniform_wake_full_ingestion(w: float) -> float:
    """Smith ideal PSC for full ingestion of a uniform wake, equal mass flow.

    Parameters
    ----------
    w : wake velocity ratio V_w / V_inf, 0 < w <= 1.
    """
    if not 0.0 < w <= 1.0:
        raise ValueError("wake velocity ratio must be in (0, 1]")
    return 2.0 * (1.0 - w) / (3.0 - w)


def psc_self_propelled_profile(profile: AnnulusProfile) -> float:
    """Ideal-disk PSC for full ingestion of a power-law profile.

    Self-propelled condition: the fan restores the whole ingested layer to
    exactly V_inf (jet velocity = V_inf, zero net force, zero residual wake).
    The podded reference produces the equivalent force D (the profile's
    momentum defect) from freestream air at equal mass flow.

    Ledger algebra (incompressible, ideal fan):
        P_BLI = V_inf D_dot - Ea            (refill power of the defect)
        P_pod = V_inf D_dot + D_dot^2 / (2 m_dot)
        PSC   = 1 - P_BLI / P_pod

    This is an independent derivation path -- no fan model, no force
    iteration -- used to cross-check the comparator to <= 2%.
    """
    full = profile.capture(profile.delta)
    d_dot = full.momentum_defect
    ea = full.ea_defect
    m_dot = full.m_dot
    v = profile.v

    p_bli = v * d_dot - ea
    p_pod = v * d_dot + d_dot**2 / (2.0 * m_dot)
    if p_pod <= 0:
        raise ValueError("degenerate profile: non-positive podded power")
    return float(1.0 - p_bli / p_pod)


def psc_parametric(
    f_bli: float,
    eta_fill: float,
    v_jet_ratio: float,
    v_wake_ratio: float,
) -> float:
    """Compact parametric estimator (proposal eq. 8, Smith/Hall form).

    PSC ~ f_BLI * eta_fill * (Vj - Vw) / (Vj + V_inf), velocities as ratios
    to V_inf.  Used only for orientation plots; the comparator is the model.
    """
    if v_jet_ratio + 1.0 <= 0:
        raise ValueError("invalid jet velocity ratio")
    return float(
        np.clip(f_bli, 0.0, 1.0)
        * np.clip(eta_fill, 0.0, 1.0)
        * (v_jet_ratio - v_wake_ratio)
        / (v_jet_ratio + 1.0)
    )
