"""Two-case power-balance comparator: podded reference vs BLI.

Both cases are evaluated from a *shared* parameter set and a *shared*
ControlVolume, at equal net streamwise force, following Drela's power
balance: no thrust/drag split is ever performed inside a case.

Bookkeeping (all quantities per the freestream reference state):

  BLI case (determined by ingestion fraction f_Phi and FPR):
    capture   -> m_dot, momentum defect D_ing, wake-KE defect Ea_ing,
                 KE-flux excess E_exc (< 0), mass-averaged pt_in
    fan       -> jet velocity Vj' from FPR on the degraded inlet
    net force S = m_dot (Vj' - V_inf) - D_res
    P_K,BLI   = 1/2 m_dot (Vj'^2 - V_inf^2) - E_exc

  Podded case (same FPR, clean inlet, sized to the same net force):
    Vj from FPR on pt_inf;  m_dot_pod = (S + D_full) / (Vj - V_inf)
    P_K,pod   = 1/2 m_dot_pod (Vj^2 - V_inf^2)

  PSC = 1 - P_K,BLI / P_K,pod

Exact ledger identity (the automated bookkeeping-residual check):

    P_K,pod - P_K,BLI = (Phi_jet,pod - Phi_jet,BLI) + Ea_ing

which is the Drela/Hall decomposition: the entire saving is the reduced
jet-mixing dissipation plus the ingested wake-mixing dissipation that no
longer occurs.  The identity is *algebraically exact* for any jet
velocities, so any violation beyond floating-point noise indicates an
implementation inconsistency; it is enforced in CI to 1e-10 relative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.optimize import brentq

from blipb.atmosphere import FlightState
from blipb.geometry import Fuselage
from blipb.ibl.solver import BLSolution, solve_fuselage_bl
from blipb.powerbalance.control_volume import ControlVolume
from blipb.powerbalance.streamtube import AnnulusProfile, CaptureState
from blipb.propulsion.fan import fan_jet
from blipb.propulsion.turboelectric import net_psc


@dataclass(frozen=True)
class ComparatorResult:
    """Everything the atlas, UQ, and validation layers need from one case."""

    # Headline metrics
    psc: float  # power-saving coefficient on P_K basis
    psc_shaft: float  # on shaft-power basis
    # Powers [W]
    pk_pod: float
    pk_bli: float
    p_shaft_pod: float
    p_shaft_bli: float
    # Decomposition [W]
    phi_jet_pod: float
    phi_jet_bli: float
    delta_phi_jet: float
    delta_phi_wake: float  # = ingested wake-mixing KE defect Ea_ing
    ledger_residual: float  # relative residual of the exact identity
    # Operating point
    f_phi: float
    y_capture: float
    m_dot_bli: float
    m_dot_pod: float
    v_jet_bli: float
    v_jet_pod: float
    net_force: float  # subsystem net streamwise force S [N]
    fpr: float
    pt_in_ratio: float  # mass-averaged pt_in / pt_inf
    # Context
    drag_fuselage: float
    delta_bl: float  # profile thickness at fan face [m]
    valid: bool
    cv: ControlVolume = field(repr=False)


class BLIComparator:
    """Podded-vs-BLI comparator for one fuselage + flight condition.

    The boundary layer is solved once per instance (geometry, flight state
    and transition location fixed); design points (f_Phi, FPR) or force
    requirements are then evaluated cheaply against it.
    """

    def __init__(
        self,
        fuselage: Fuselage | None = None,
        flight: FlightState | None = None,
        x_tr_frac: float = 0.05,
        n_powerlaw: float = 7.0,
        eta_pol: float = 0.92,
        k_dist: float = 0.03,
        n_points: int = 400,
        bl: BLSolution | None = None,
    ) -> None:
        self.fuselage = fuselage or Fuselage()
        self.flight = flight or FlightState(mach=0.785, altitude=10_668.0)
        self.x_tr_frac = x_tr_frac
        self.n_powerlaw = n_powerlaw
        self.eta_pol = eta_pol
        self.k_dist = k_dist

        self.bl: BLSolution = bl if bl is not None else solve_fuselage_bl(
            self.fuselage, self.flight, x_tr_frac=x_tr_frac, n_points=n_points
        )
        atm = self.flight.atm
        self.cv = ControlVolume(
            v_inf=self.flight.V,
            rho=atm.rho,
            p_inf=atm.p,
            tt_inf=self.flight.Tt,
            pt_inf=self.flight.pt,
        )
        self.profile = AnnulusProfile(
            theta_eq=self.bl.wake.theta_eq,
            n_powerlaw=n_powerlaw,
            r_hub=self.bl.r_te,
            v_edge=self.cv.v_inf,
            rho=self.cv.rho,
            p_static=self.cv.p_inf,
            tt=self.cv.tt_inf,
        )

    # ------------------------------------------------------------------ API

    def run_design(self, f_phi: float, fpr: float) -> ComparatorResult:
        """Design-driven mode: ingestion fraction and FPR prescribed."""
        capture = self.profile.solve_capture_for_fphi(f_phi)
        return self._evaluate(capture, fpr)

    def run_force(self, s_req: float, fpr: float) -> ComparatorResult:
        """Force-driven mode: solve capture height for a required net force.

        s_req is the (fuselage + aft propulsor) subsystem net streamwise
        force; e.g. for an aft fan carrying thrust F_aft on a fuselage of
        drag D_fus, s_req = F_aft - D_fus.
        """
        delta = self.profile.delta

        def resid(y_c: float) -> float:
            res = self._evaluate(self.profile.capture(y_c), fpr, check=False)
            return res.net_force - s_req

        lo, hi = 1e-3 * delta, 8.0 * delta
        r_lo, r_hi = resid(lo), resid(hi)
        if r_lo * r_hi > 0:
            raise ValueError(
                f"required force {s_req:.0f} N unreachable at FPR={fpr}: "
                f"net-force range [{r_lo + s_req:.0f}, {r_hi + s_req:.0f}] N"
            )
        y_c = brentq(resid, lo, hi, xtol=1e-10 * delta)
        return self._evaluate(self.profile.capture(float(y_c)), fpr)

    # ----------------------------------------------------------------- core

    def _evaluate(
        self, capture: CaptureState, fpr: float, check: bool = True
    ) -> ComparatorResult:
        cv = self.cv
        v = cv.v_inf

        # ---- BLI case: fan on the degraded, mass-averaged inlet ----------
        delta_eta = self.k_dist * capture.f_phi
        fan_bli = fan_jet(
            pt_in=capture.pt_mass_avg,
            tt_in=cv.tt_inf,
            fpr=fpr,
            p_inf=cv.p_inf,
            eta_pol=self.eta_pol,
            delta_eta_distortion=delta_eta,
        )
        m_bli = capture.m_dot
        vj_b = fan_bli.v_jet

        d_full = self.profile.momentum_defect_total()
        d_res = d_full - capture.momentum_defect
        s_force = m_bli * (vj_b - v) - d_res

        pk_bli = 0.5 * m_bli * (vj_b**2 - v**2) - capture.ke_flux_excess
        p_shaft_bli = m_bli * fan_bli.w_shaft

        # ---- Podded case: clean inlet, sized to the same net force -------
        fan_pod = fan_jet(
            pt_in=cv.pt_inf,
            tt_in=cv.tt_inf,
            fpr=fpr,
            p_inf=cv.p_inf,
            eta_pol=self.eta_pol,
            delta_eta_distortion=0.0,
        )
        vj_p = fan_pod.v_jet
        if vj_p <= v * 1.001:
            raise ValueError(f"podded jet velocity {vj_p:.1f} <= V_inf; FPR too low")
        m_pod = (s_force + d_full) / (vj_p - v)
        if m_pod <= 0:
            raise ValueError("podded mass flow non-positive; force target inconsistent")

        pk_pod = 0.5 * m_pod * (vj_p**2 - v**2)
        p_shaft_pod = m_pod * fan_pod.w_shaft

        # ---- Ledger decomposition and exact-identity residual -------------
        phi_jet_pod = 0.5 * m_pod * (vj_p - v) ** 2
        phi_jet_bli = 0.5 * m_bli * (vj_b - v) ** 2
        d_phi_jet = phi_jet_pod - phi_jet_bli
        d_phi_wake = capture.ea_defect

        saving = pk_pod - pk_bli
        residual = (saving - (d_phi_jet + d_phi_wake)) / max(abs(pk_pod), 1e-300)
        if check and abs(residual) > 1e-8:
            raise AssertionError(
                f"power-balance bookkeeping violated: relative residual {residual:.3e}"
            )

        psc = saving / pk_pod
        psc_shaft = 1.0 - p_shaft_bli / p_shaft_pod

        return ComparatorResult(
            psc=float(psc),
            psc_shaft=float(psc_shaft),
            pk_pod=float(pk_pod),
            pk_bli=float(pk_bli),
            p_shaft_pod=float(p_shaft_pod),
            p_shaft_bli=float(p_shaft_bli),
            phi_jet_pod=float(phi_jet_pod),
            phi_jet_bli=float(phi_jet_bli),
            delta_phi_jet=float(d_phi_jet),
            delta_phi_wake=float(d_phi_wake),
            ledger_residual=float(residual),
            f_phi=float(capture.f_phi),
            y_capture=float(capture.y_c),
            m_dot_bli=float(m_bli),
            m_dot_pod=float(m_pod),
            v_jet_bli=float(vj_b),
            v_jet_pod=float(vj_p),
            net_force=float(s_force),
            fpr=float(fpr),
            pt_in_ratio=float(capture.pt_mass_avg / cv.pt_inf),
            drag_fuselage=float(self.bl.drag),
            delta_bl=float(self.profile.delta),
            valid=self.bl.valid,
            cv=cv,
        )

    # ------------------------------------------------------------ mission

    def net_psc(self, result: ComparatorResult, phi: float = 0.28, eta_elec: float = 0.92) -> float:
        """Turboelectric net PSC (shaft-power basis, see propulsion.turboelectric)."""
        return net_psc(result.p_shaft_pod, result.p_shaft_bli, phi=phi, eta_elec=eta_elec)
