"""Sanity tests for experimental Nusselt correlations (validation baselines)."""
import pytest

from nanofluid_hx.properties import MaterialProperties
from nanofluid_hx.correlations import prandtl, dittus_boelter, pak_cho


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
