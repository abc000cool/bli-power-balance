"""Wake extrapolation: Squire-Young closure.

The Squire-Young formula carries the trailing-edge momentum thickness to the
far wake where the edge velocity has relaxed to V_inf and static pressure to
p_inf:

    theta_inf = theta_TE (U_e,TE / V_inf)^((H_TE + 5) / 2)

For a body of revolution the momentum-defect *area* is what is conserved
far downstream, so the closure is applied to (r_TE theta_TE).  The profile
drag then follows from the far-wake momentum defect:

    D = 2 pi rho V_inf^2 (r theta)_inf
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WakeResult:
    theta_inf_area: float  # (r theta)_inf, momentum-defect area / (2 pi) [m^2]
    drag: float  # far-wake profile drag [N]
    theta_eq: float  # equivalent planar momentum thickness at V_inf edge [m]


def squire_young(
    theta_te: float,
    h_te: float,
    ue_te: float,
    v_inf: float,
    rho: float,
    r_te: float,
) -> WakeResult:
    """Extrapolate the TE boundary-layer state to the far wake."""
    factor = (ue_te / v_inf) ** (0.5 * (h_te + 5.0))
    theta_eq = theta_te * factor
    theta_inf_area = r_te * theta_eq
    drag = 2.0 * 3.141592653589793 * rho * v_inf**2 * theta_inf_area
    return WakeResult(theta_inf_area=theta_inf_area, drag=drag, theta_eq=theta_eq)
