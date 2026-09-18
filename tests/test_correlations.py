"""Sanity tests for experimental Nusselt correlations (validation baselines)."""
import pytest

from nanofluid_hx.properties import MaterialProperties
from nanofluid_hx.correlations import (prandtl, dittus_boelter, pak_cho,
                                       xuan_li)


@pytest.fixture(scope="module")
def water():
    return MaterialProperties(phi=0.0)


def test_prandtl_water_reference_value(water):
    # mu*cp/k = 8.91e-4 * 4179 / 0.613 = 6.07 (hand value)
    assert prandtl(water) == pytest.approx(6.07, rel=0.01)


def test_phi_zero_collapses_to_dittus_boelter(water):
    # Pak-Cho at phi=0 must land near the pure-water DB baseline (within 10%:
    # 0.021/0.023 * Pr^(0.5-0.4) ~ +9% by construction)
    Re = 1.0e4
    ratio = pak_cho(Re, water) / dittus_boelter(Re, water)
    assert ratio == pytest.approx(1.09, abs=0.03)


def test_nusselt_rises_with_phi():
    # Pr_nf rises with phi (mu grows faster than k) => Nu_corr rises, the same
    # qualitative trend as the CFD Nu_nf sweep
    Re = 3.0e4
    assert pak_cho(Re, MaterialProperties(0.1)) > pak_cho(Re, MaterialProperties(0.0))


def test_xuan_li_phi_zero_form(water):
    # phi=0 kills the bracket => Nu = 0.0059 Re^0.9238 Pr^0.4 exactly
    Re = 3.0e4
    expected = 0.0059 * Re**0.9238 * prandtl(water)**0.4
    assert xuan_li(Re, water) == pytest.approx(expected, rel=1e-12)


def test_xuan_li_rises_with_re_and_phi():
    p0, p1 = MaterialProperties(0.0), MaterialProperties(0.05)
    assert xuan_li(6.0e4, p0) > xuan_li(1.0e4, p0)
    assert xuan_li(3.0e4, p1) > xuan_li(3.0e4, p0)
