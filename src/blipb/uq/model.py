"""End-to-end model wrapper: parameter vector -> (PSC_aero, PSC_net, dFuel).

This is the function the atlas, Sobol and PCE layers all call.  The
boundary-layer solve depends only on x_tr among the seven UQ inputs, so BL
solutions are memoized on x_tr rounded to 1e-3 of the fuselage length
(the induced discretization error is far below closure uncertainty and cuts
the cost of a Saltelli sweep by ~100x).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from joblib import Parallel, delayed

from blipb.atmosphere import FlightState
from blipb.geometry import Fuselage
from blipb.ibl.solver import solve_fuselage_bl
from blipb.mission.breguet import NMI, delta_block_fuel
from blipb.powerbalance.comparator import BLIComparator

OUTPUT_NAMES = ["psc_aero", "psc_net", "delta_fuel"]


@dataclass(frozen=True)
class StudyConfig:
    """Frozen study-level configuration (SPEC.md baseline)."""

    mach: float = 0.785
    altitude: float = 10_668.0  # m (FL350)
    fuselage_length: float = 37.0
    fuselage_radius: float = 1.88
    eta_prop_ref: float = 0.80  # underwing-turbofan propulsive efficiency
    lift_drag: float = 21.4
    tsfc: float = 14.2e-6  # kg/(N s)
    range_nmi: float = 3000.0
    w_initial: float = 60_000.0  # kg
    snowball: float = 1.35
    n_points: int = 400

    def fuselage(self) -> Fuselage:
        return Fuselage(length=self.fuselage_length, radius_max=self.fuselage_radius)

    def flight(self) -> FlightState:
        return FlightState(mach=self.mach, altitude=self.altitude)


DEFAULT_CONFIG = StudyConfig()

_XTR_QUANTUM = 1e-3  # x_tr/L cache resolution


@lru_cache(maxsize=256)
def _cached_bl(config: StudyConfig, x_tr_q: float):
    return solve_fuselage_bl(
        config.fuselage(), config.flight(), x_tr_frac=x_tr_q, n_points=config.n_points
    )


def evaluate_point(
    params: dict[str, float] | np.ndarray,
    config: StudyConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    """Evaluate one design/uncertainty point.

    Parameters may be a dict keyed by INPUT_NAMES or an array in that order:
    [f_phi, fpr, x_tr, n_powerlaw, eta_pol, k_dist, eta_elec].
    Returns dict with psc_aero, psc_net, delta_fuel (and validity flag).
    """
    from blipb.uq.problem import INPUT_NAMES

    if not isinstance(params, dict):
        arr = np.asarray(params, dtype=float).ravel()
        params = dict(zip(INPUT_NAMES, arr))

    x_tr_q = round(float(params["x_tr"]) / _XTR_QUANTUM) * _XTR_QUANTUM
    bl = _cached_bl(config, x_tr_q)

    comp = BLIComparator(
        fuselage=config.fuselage(),
        flight=config.flight(),
        x_tr_frac=x_tr_q,
        n_powerlaw=float(params["n_powerlaw"]),
        eta_pol=float(params["eta_pol"]),
        k_dist=float(params["k_dist"]),
        n_points=config.n_points,
        bl=bl,
    )
    res = comp.run_design(f_phi=float(params["f_phi"]), fpr=float(params["fpr"]))

    # Aircraft-level net PSC: the aft-fan shaft saving (charged with the
    # electrical-chain loss) relative to the total propulsive shaft power of
    # the aircraft.  The BLI power fraction is therefore dynamic -- it grows
    # with the captured stream tube -- rather than frozen at a design-point
    # value, which keeps the f_phi sweep physically consistent.
    from blipb.atmosphere import G0

    d_total = config.w_initial * G0 / config.lift_drag
    p_shaft_total_ref = d_total * config.flight().V / config.eta_prop_ref
    eta_elec = float(params["eta_elec"])
    psc_net = (res.p_shaft_pod - res.p_shaft_bli / eta_elec) / p_shaft_total_ref
    d_fuel = delta_block_fuel(
        psc_net,
        range_m=config.range_nmi * NMI,
        v=config.flight().V,
        lift_drag=config.lift_drag,
        tsfc=config.tsfc,
        w_initial=config.w_initial,
        snowball=config.snowball,
    )
    return {
        "psc_aero": res.psc,
        "psc_net": psc_net,
        "delta_fuel": d_fuel,
        "valid": float(res.valid),
    }


def evaluate_batch(
    x: np.ndarray,
    config: StudyConfig = DEFAULT_CONFIG,
    n_jobs: int = -1,
) -> np.ndarray:
    """Evaluate an (N, 7) sample matrix in parallel -> (N, 3) outputs.

    Failed evaluations (physically inconsistent corners) return NaN rows;
    callers must mask them and report the dropped count (SPEC: no silent
    invalid cells).
    """

    def one(row: np.ndarray) -> list[float]:
        try:
            out = evaluate_point(row, config)
            return [out["psc_aero"], out["psc_net"], out["delta_fuel"]]
        except (ValueError, AssertionError, RuntimeError):
            return [np.nan, np.nan, np.nan]

    x = np.atleast_2d(np.asarray(x, dtype=float))
    results = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(one)(row) for row in x)
    return np.asarray(results, dtype=float)
