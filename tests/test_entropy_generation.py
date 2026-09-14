"""Tests for entropy_generation.py (Phase A1, Eqs. EGM-1..4 in its docstring)."""
import numpy as np
import pytest

from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.turbulence import get_model
from nanofluid_hx.entropy_generation import (local_entropy_generation,
                                             from_case,
                                             integrate_entropy_generation)


@pytest.fixture(scope="module")
def mesh():
    return AxisymmetricMesh(Nr_inner=8, Nr_wall=3, Nr_outer=8, Nz=40)


def _fields(mesh, T, u, mu, k):
    """Array-aware manufactured fields: T(r, z) and u(r) as vector lambdas."""
    Nr, Nz = mesh.Nr, mesh.Nz
    u_face = np.tile(np.atleast_1d(u(mesh.r_center))[:, None], (1, Nz + 1))
    T_arr = np.full((Nr, Nz), 0.0)
    for j, z in enumerate(mesh.z_center):
        for i, r in enumerate(mesh.r_center):
            T_arr[i, j] = T(r, z)
    mu_eff = np.full((Nr, Nz), mu)
    k_eff = np.full((Nr, Nz), k)
    return T_arr, u_face, mu_eff, k_eff


def test_uniform_field_gives_zero(mesh):
    T, u_face, mu_eff, k_eff = _fields(mesh, lambda r, z: 350.0 + 0.0 * r,
                                       lambda r: 1.0 + 0.0 * r, 1e-3, 0.6)
    s_ht, s_ff = local_entropy_generation(mesh, T, u_face, mu_eff, k_eff)
    # np.gradient leaves ~1e-30 roundoff on constant fields: tolerance, not ==
    assert np.all(s_ht < 1e-20) and np.all(s_ff < 1e-20)


def test_linear_temperature_hand_value(mesh):
    # T = 300 + 10*r  =>  dT/dr = 10 everywhere except one-sided edge cells;
    # interior cells must equal s = k*100/T^2 exactly (EGM-1a)
    G = 10.0
    T, u_face, mu_eff, k_eff = _fields(mesh, lambda r, z: 300.0 + G * r,
                                       lambda r: 0.0 * r, 1e-3, 0.7)
    s_ht, s_ff = local_entropy_generation(mesh, T, u_face, mu_eff, k_eff)
    interior = np.zeros_like(s_ht, dtype=bool)
    interior[1:-1, :] = True
    expected = 0.7 * G**2 / (300.0 + G * mesh.r_center)[:, None]**2
    assert np.allclose(s_ht[interior], np.broadcast_to(expected, s_ht.shape)[interior])
    assert np.all(s_ff == 0.0)


def test_linear_velocity_hand_value_and_solid_zero(mesh):
    # u = 5*r  =>  du/dr = 5 interior; s_ff = mu*25/T exactly (EGM-1b).
    # Wall-zone cells must be EXACTLY zero regardless of mu (solid, EGM-2).
    T, u_face, mu_eff, k_eff = _fields(mesh, lambda r, z: 320.0 + 0.0 * r,
                                       lambda r: 5.0 * r, 2e-3, 0.7)
    mu_eff[:] = 2e-3                       # even inside the steel wall
    s_ht, s_ff = local_entropy_generation(mesh, T, u_face, mu_eff, k_eff)
    assert np.all(s_ht < 1e-20)                   # roundoff tolerance
    wall = mesh.zone_map == 1
    assert np.all(s_ff[wall] == 0.0)
    fluid_interior = np.zeros_like(s_ff, dtype=bool)
    fluid_interior[1:-1, :] = True
    fluid_interior[wall] = False
    assert np.allclose(s_ff[fluid_interior], 2e-3 * 25.0 / 320.0)


def test_integration_exact_for_uniform_rate(mesh):
    # Constant s''' = c in pipe zone only => S_pipe = c * sum(V_pipe) (EGM-4)
    c = 3.0
    T, u_face, mu_eff, k_eff = _fields(mesh, lambda r, z: 320.0 + 0.0 * r,
                                       lambda r: 0.0 * r, 1e-3, 0.7)
    s_ht = np.where((mesh.zone_map == 0)[:, None], c, 0.0)
    s_ff = np.zeros_like(s_ht)
    out = integrate_entropy_generation(mesh, s_ht, s_ff)
    assert out["pipe"]["ht"] == pytest.approx(c * mesh.V[mesh.zone_map == 0].sum())
    assert out["pipe"]["Be"] == 1.0
    assert out["wall"]["S"] == 0.0
    assert out["total"]["ht"] == pytest.approx(c * mesh.V[mesh.zone_map == 0].sum())


@pytest.fixture(scope="module")
def real_case(mesh):
    fd = get_model("simplec_k_epsilon")(mesh, MaterialProperties(0.05),
                                        MaterialProperties(0.0),
                                        Re_inner=2.0e4, Re_outer=2.0e4,
                                        max_outer_iter=500)
    solver = ThermalSolver(mesh, fd, parallel_flow=True)
    solver.assemble_system()
    return fd, solver.solve(), solver


def test_real_case_anchors(real_case, mesh):
    fd, T, solver = real_case
    s_ht, s_ff = from_case(mesh, fd, T)
    assert np.all(s_ht > 0.0)                       # heated flow everywhere
    assert np.all(np.isfinite(s_ht)) and np.all(np.isfinite(s_ff))
    out = integrate_entropy_generation(mesh, s_ht, s_ff)
    for zone in ("pipe", "annulus", "total"):
        assert 0.0 < out[zone]["Be"] < 1.0          # (EGM-3) both mechanisms
        assert out[zone]["S"] > 0.0
    assert out["wall"]["Be"] == 1.0                 # solid: pure HTI by physics
    assert out["wall"]["ff"] == 0.0
    # Positive-temperature guard: solver T is absolute kelvin
    assert T.min() > 0.0
