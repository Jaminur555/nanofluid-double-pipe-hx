import pytest
from nanofluid_hx import MaterialProperties


def test_pure_water_at_phi_zero():
    p = MaterialProperties(phi=0.0)
    assert p.rho_nf == pytest.approx(p.rho_f)
    assert p.cp_nf  == pytest.approx(p.cp_f)
    assert p.mu_nf  == pytest.approx(p.mu_f)
    assert p.k_nf   == pytest.approx(p.k_f)


def test_properties_increase_with_phi():
    lo, hi = MaterialProperties(0.02), MaterialProperties(0.05)
    assert hi.rho_nf > lo.rho_nf
    assert hi.mu_nf  > lo.mu_nf
    assert hi.k_nf   > lo.k_nf


def test_phi_out_of_range_raises():
    with pytest.raises(ValueError):
        MaterialProperties(0.11)