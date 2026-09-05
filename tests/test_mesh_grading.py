"""Regraded-mesh geometry guards (Stage 3c step 1).

Wall-function sizing: every wall-adjacent FLUID cell center must sit in the
log layer (y+ ~ 14-165 by the Blasius correlation) for the whole Re sweep.
Catches the old failure modes: one-sided annulus grading (outer-wall first
cell y+ ~ 2.6-8) and uniform annulus (y+ ~ 9.5 at low Re).
"""
import numpy as np
import pytest

from nanofluid_hx.mesh import AxisymmetricMesh


def _yplus(Re, y_p, D):
    """First-cell-center y+ from a friction correlation (f = 0.184 Re^-0.2)."""
    f = 0.184 * Re ** -0.2
    return y_p * Re * np.sqrt(f / 8.0) / D


@pytest.fixture(scope="module")
def mesh():
    return AxisymmetricMesh()  # regraded defaults 15 / 5 / 16 / 150


def test_wall_yplus_in_wall_function_band(mesh):
    dr = np.diff(mesh.r_faces)
    i2 = mesh.Nr_inner + mesh.Nr_wall
    walls = [
        ("pipe@r1", 0.5 * dr[mesh.Nr_inner - 1], 2.0 * mesh.r1),
        ("annulus@r2", 0.5 * dr[i2], 2.0 * (mesh.r3 - mesh.r2)),
        ("annulus@r3", 0.5 * dr[-1], 2.0 * (mesh.r3 - mesh.r2)),
    ]
    for name, y_p, D in walls:
        for Re in (1e4, 3e4, 1e5):
            yp = _yplus(Re, y_p, D)
            assert 14.0 <= yp <= 165.0, f"{name} Re={Re:.0g}: y+ = {yp:.1f}"


def test_pipe_sizes_grow_toward_wall(mesh):
    dr = np.diff(mesh.r_faces)[: mesh.Nr_inner]
    assert np.all(np.diff(dr) > 0.0)                      # fine at axis, coarse at wall
    assert abs(dr[-1] - 1.359e-3) < 2e-5                  # pipe wall cell
    assert np.max(dr[1:] / dr[:-1]) < 2.5                 # axis end unconstrained (2.03 there)
    assert np.max(dr[mesh.Nr_inner // 2:] / dr[mesh.Nr_inner // 2 - 1:-1]) < 1.5  # wall-side smooth


def test_annulus_two_sided_symmetric(mesh):
    i2 = mesh.Nr_inner + mesh.Nr_wall
    dr = np.diff(mesh.r_faces)[i2:]
    assert np.allclose(dr, dr[::-1])                      # mirrored widths
    assert np.argmin(dr) in (len(dr) // 2 - 1, len(dr) // 2)  # smallest at mid-gap
    assert abs(dr[0] - 1.166e-3) < 2e-5                   # wall cells both sides
    assert abs(dr[-1] - 1.166e-3) < 2e-5
    assert np.max(dr[1:] / dr[:-1]) < 2.0


def test_wall_zone_uniform(mesh):
    i1 = mesh.Nr_inner
    drw = np.diff(mesh.r_faces)[i1: i1 + mesh.Nr_wall]
    assert np.allclose(drw, (mesh.r2 - mesh.r1) / mesh.Nr_wall)  # 5 x 0.4 mm


def test_wall_adjacent_cells_in_band(mesh):
    dr = np.diff(mesh.r_faces)
    i2 = mesh.Nr_inner + mesh.Nr_wall
    for w in (dr[mesh.Nr_inner - 1], dr[i2], dr[-1]):
        assert 0.9e-3 <= w <= 1.6e-3


def test_odd_nr_outer_constructs():
    """Odd annulus counts (e.g. 15) must still close exactly at r3."""
    m = AxisymmetricMesh(Nr_inner=8, Nr_wall=3, Nr_outer=15, Nz=10)
    dr = np.diff(m.r_faces)
    assert np.all(dr > 0.0)
    assert np.isclose(m.r_faces[-1], m.r3)
    assert np.isclose(np.sum(m.V), np.pi * m.r3 ** 2 * m.L, rtol=1e-10)
    dro = dr[m.Nr_inner + m.Nr_wall:]
    assert np.allclose(dro, dro[::-1])                    # still symmetric
