"""Two-case comparator: bookkeeping identity, physical trends, force mode."""

import pytest

from blipb import BLIComparator
from blipb.powerbalance import hall2017
from blipb.powerbalance.smith1993 import psc_self_propelled_profile


@pytest.fixture(scope="module")
def comp() -> BLIComparator:
    return BLIComparator()  # SPEC baseline: M0.785, FL350, STARC-ABL body


def test_baseline_sane(comp):
    res = comp.run_design(f_phi=0.5, fpr=1.25)
    assert 0.0 < res.psc < 0.5
    assert 0.0 < res.psc_shaft < 0.5
    assert res.f_phi == pytest.approx(0.5, abs=1e-6)
    assert res.pt_in_ratio < 1.0
    assert res.v_jet_bli < res.v_jet_pod
    assert res.valid


def test_ledger_identity_machine_precision(comp):
    # The Drela/Hall decomposition identity is algebraically exact:
    # PK_pod - PK_bli = dPhi_jet + dPhi_wake.  CI-enforced bookkeeping.
    for f in (0.2, 0.5, 0.9):
        for fpr in (1.2, 1.35, 1.5):
            res = comp.run_design(f_phi=f, fpr=fpr)
            assert abs(res.ledger_residual) < 1e-10


def test_decomposition_sums_to_psc(comp):
    res = comp.run_design(f_phi=0.6, fpr=1.3)
    d = hall2017.decompose(res)
    assert d.total == pytest.approx(res.psc, rel=1e-9)
    assert d.jet > 0 and d.wake > 0
    assert d.surface == 0.0  # declared low-order limitation
    assert 0.0 < hall2017.effective_fill_factor(res) <= 1.0


def test_saving_increases_with_ingestion_and_saturates(comp):
    # The *absolute* power saving grows with ingestion but with diminishing
    # returns: each marginal slice of captured boundary layer is faster and
    # less degraded than the last.  (The subsystem PSC *ratio* actually
    # falls with f_phi because the podded twin grows too -- the aircraft
    # -level benefit is what saturates; see paper section 4.)
    fs = (0.2, 0.4, 0.6, 0.8)
    savings = [
        (lambda r: r.pk_pod - r.pk_bli)(comp.run_design(f_phi=f, fpr=1.25)) for f in fs
    ]
    assert all(b > a for a, b in zip(savings, savings[1:]))  # monotone
    increments = [b - a for a, b in zip(savings, savings[1:])]
    assert all(b < a for a, b in zip(increments, increments[1:]))  # concave


def test_distortion_penalty_reduces_psc():
    clean = BLIComparator(k_dist=0.0).run_design(f_phi=0.6, fpr=1.25)
    dirty = BLIComparator(k_dist=0.05).run_design(f_phi=0.6, fpr=1.25)
    assert dirty.psc < clean.psc


def test_zero_ingestion_limit(comp):
    # f_phi -> 0: the captured slice is the *deepest* (most degraded) part
    # of the boundary layer, so the per-fan PSC stays finite and large --
    # but the fan itself vanishes, so the absolute saving goes to zero.
    res = comp.run_design(f_phi=1e-3, fpr=1.25)
    assert 0.0 < res.psc < 0.7  # bounded, physical
    full = comp.run_design(f_phi=0.9, fpr=1.25)
    assert (res.pk_pod - res.pk_bli) < 0.01 * (full.pk_pod - full.pk_bli)


def test_force_mode_hits_target(comp):
    s_req = 2000.0  # N
    res = comp.run_force(s_req=s_req, fpr=1.3)
    assert res.net_force == pytest.approx(s_req, rel=1e-6)
    assert 0.0 < res.f_phi <= 1.0
    assert abs(res.ledger_residual) < 1e-10


def test_force_mode_unreachable_raises(comp):
    with pytest.raises(ValueError):
        comp.run_force(s_req=1e9, fpr=1.25)


def test_smith_limit_low_mach():
    """SPEC target V3: ideal-fan comparator vs the Smith closed form, <= 2%.

    Low Mach, eta = 1, no distortion, full ingestion, self-propelled
    condition (jet refills the wake to V_inf).  The closed form assumes an
    equal-mass-flow podded reference, so the comparator's PK_bli is compared
    against the closed-form ledger using the same profile integrals.
    """
    from scipy.optimize import brentq

    comp = BLIComparator(
        flight=__import__("blipb").FlightState(mach=0.20, altitude=1000.0),
        eta_pol=1.0,
        k_dist=0.0,
    )
    v = comp.cv.v_inf

    def vj_err(fpr):
        return comp.run_design(f_phi=1.0, fpr=fpr).v_jet_bli - v

    fpr_sp = brentq(vj_err, 1.001, 1.2, xtol=1e-10)
    res = comp.run_design(f_phi=1.0, fpr=fpr_sp)

    # closed-form equal-mass-flow podded power from the same profile
    full = comp.profile.capture(comp.profile.delta)
    d_dot, m_dot = full.momentum_defect, full.m_dot
    p_pod_eq = v * d_dot + d_dot**2 / (2.0 * m_dot)
    psc_comparator_rule = 1.0 - res.pk_bli / p_pod_eq

    psc_closed = psc_self_propelled_profile(comp.profile)
    assert psc_comparator_rule == pytest.approx(psc_closed, rel=0.02)


def test_control_volume_shared(comp):
    r1 = comp.run_design(f_phi=0.3, fpr=1.25)
    r2 = comp.run_design(f_phi=0.7, fpr=1.45)
    assert r1.cv is r2.cv  # one canonical CV per comparison set


def test_net_psc_electric_penalty(comp):
    res = comp.run_design(f_phi=0.5, fpr=1.25)
    lossless = comp.net_psc(res, phi=0.28, eta_elec=1.0)
    lossy = comp.net_psc(res, phi=0.28, eta_elec=0.90)
    assert lossy < lossless
    # phi-weighting: whole-aircraft saving is smaller than subsystem saving
    assert lossless == pytest.approx(0.28 * res.psc_shaft, rel=1e-9)
