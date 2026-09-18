"""Sanity tests for particle selectors and mixture-model variants."""
import pytest

from nanofluid_hx import MaterialProperties


def test_defaults_reproduce_bahmani_fits():
    # phi=0.05: mu_r = 123*0.05^2 + 7.3*0.05 + 1, k_r = 4.97*0.05^2 + 2.72*0.05 + 1
    p = MaterialProperties(0.05)
    assert p.rho_nf == pytest.approx(0.95 * 997.1 + 0.05 * 3970.0)
    assert p.cp_nf == pytest.approx(0.95 * 4179.0 + 0.05 * 765.0)
    assert p.mu_nf == pytest.approx(1.6725 * 8.91e-4)
    assert p.k_nf == pytest.approx((4.97 * 0.05**2 + 2.72 * 0.05 + 1) * 0.613)


def test_bahmani_fits_rejected_for_other_particles():
    with pytest.raises(ValueError):
        MaterialProperties(0.05, particle="cuo")            # defaults = bahmani
    with pytest.raises(ValueError):
        MaterialProperties(0.05, particle="cu", k_model="maxwell")


def test_cuo_with_classical_models():
    # Maxwell at phi=0.05, k_p=20, k_f=0.613 (hand value); Brinkman (0.95)^-2.5
    p = MaterialProperties(0.05, particle="cuo", k_model="maxwell",
                           mu_model="brinkman")
    k_r = (20.0 + 2 * 0.613 - 2 * 0.05 * (0.613 - 20.0)) / \
          (20.0 + 2 * 0.613 + 0.05 * (0.613 - 20.0))
    assert p.k_nf == pytest.approx(k_r * 0.613)
    assert p.mu_nf == pytest.approx(0.95 ** -2.5 * 8.91e-4)


def test_corcione_hand_anchors():
    # Hand values at phi=0.05, d_p=47 nm, T=300 K (Corcione 2011 forms)
    p = MaterialProperties(0.05, mu_model="corcione", k_model="corcione")
    assert p.mu_nf / p.mu_f == pytest.approx(1.233, rel=0.01)
    assert p.k_nf / p.k_f == pytest.approx(1.127, rel=0.01)


def test_batchelor_close_to_brinkman():
    # Both dilute-limit theories: within ~1% at phi=0.05
    b = MaterialProperties(0.05, mu_model="batchelor")
    r = MaterialProperties(0.05, mu_model="brinkman")
    assert b.mu_nf == pytest.approx(r.mu_nf, rel=0.01)


def test_mixture_cp_below_volume_weighted():
    # rho_p*cp_p < rho_f*cp_f => energy-consistent cp is the lower estimate
    v = MaterialProperties(0.05)
    m = MaterialProperties(0.05, cp_model="mixture")
    assert m.cp_nf < v.cp_nf


def test_all_model_variants_monotone_in_phi():
    combos = [{}, {"mu_model": "brinkman"}, {"mu_model": "corcione"},
              {"k_model": "maxwell"}, {"k_model": "corcione"},
              {"cp_model": "mixture"}]
    for kw in combos:
        lo = MaterialProperties(0.02, **kw)
        hi = MaterialProperties(0.05, **kw)
        assert hi.k_nf > lo.k_nf
        assert hi.mu_nf > lo.mu_nf


def test_unknown_selectors_raise():
    with pytest.raises(ValueError):
        MaterialProperties(0.05, particle="znO")
    with pytest.raises(ValueError):
        MaterialProperties(0.05, mu_model="einstein")
    with pytest.raises(ValueError):
        MaterialProperties(0.05, k_model="hamilton")
