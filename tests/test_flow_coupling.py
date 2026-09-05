"""Structural tests for the SimplecFlow provider (Stage 3c).

Small mesh, Re = 20000, phi = 0.05: convergence is NOT required -- these
assert shape/identity/mass-conservation structure, not flow accuracy.
"""
import numpy as np
import pytest

from nanofluid_hx import AxisymmetricMesh, MaterialProperties
from nanofluid_hx.flow.coupling import SimplecFlow


@pytest.fixture(scope="module")
def case():
    mesh = AxisymmetricMesh(Nr_inner=8, Nr_wall=3, Nr_outer=8, Nz=40)
    pi, po = MaterialProperties(0.05), MaterialProperties(0.0)
    fd = SimplecFlow(mesh, pi, po, 20000, 20000)
    return mesh, pi, po, fd


def test_shapes(case):
    mesh, pi, po, fd = case
    Nr, Nz = mesh.Nr, mesh.Nz
    assert fd.u_face.shape == (Nr, Nz + 1)
    assert fd.mu_t.shape == (Nr, Nz)
    assert fd.k_eff.shape == (Nr, Nz)
    assert fd.v.shape == (Nr + 1, Nz)
    assert fd.k.shape == fd.eps.shape == (Nr, Nz)
    assert fd.u.shape == (Nr,)                     # legacy 1-D view
    assert np.allclose(fd.u, fd.u_face[:, -1])     # = outlet column


def test_wall_rows(case):
    mesh, pi, po, fd = case
    i1, i2 = fd.i1, fd.i2
    assert np.all(fd.u_face[i1:i2, :] == 0.0)
    assert np.all(fd.mu_t[i1:i2, :] == 0.0)
    assert np.all(fd.k_eff[i1:i2, :] == pi.k_s)


def test_node_identity(case):
    """The zero-interpolation core: sub-meshes ARE slices of thermal faces."""
    mesh, pi, po, fd = case
    i1, i2 = fd.i1, fd.i2
    assert np.array_equal(fd.pipe_mesh.r_faces, mesh.r_faces[:i1 + 1])
    assert np.array_equal(fd.annulus_mesh.r_faces, mesh.r_faces[i2:])
    assert np.array_equal(fd.pipe_mesh.z_faces, mesh.z_faces)
    assert np.array_equal(fd.annulus_mesh.z_faces, mesh.z_faces)
    assert np.array_equal(fd.u_face[:i1, :], fd.result_inner["u"])
    assert np.array_equal(fd.u_face[i2:, :], fd.result_outer["u"])


def test_inlet_columns_uniform(case):
    mesh, pi, po, fd = case
    assert np.allclose(fd.u_face[:fd.i1, 0], fd.U_in_inner)
    assert np.allclose(fd.u_face[fd.i2:, 0], fd.U_in_outer)


def test_mass_flow_conserved(case):
    mesh, pi, po, fd = case
    inner = slice(0, fd.i1)
    outer = slice(fd.i2, mesh.Nr)

    mdot_i = np.sum(pi.rho_nf * fd.u_face[inner, -1] * mesh.A_e[inner, -1])
    assert mdot_i == pytest.approx(
        pi.rho_nf * fd.U_in_inner * np.pi * mesh.r1 ** 2, rel=0.01)

    mdot_o = np.sum(po.rho_f * fd.u_face[outer, -1] * mesh.A_e[outer, -1])
    assert mdot_o == pytest.approx(
        po.rho_f * fd.U_in_outer * np.pi * (mesh.r3 ** 2 - mesh.r2 ** 2), rel=0.01)


def test_eddy_enhances_conductivity(case):
    mesh, pi, po, fd = case
    assert np.all(fd.k_eff[:fd.i1, :] > pi.k_nf)
    assert np.all(fd.k_eff[fd.i2:, :] > po.k_f)


def test_eddy_conductivity_units(case):
    """mu_t is DYNAMIC [Pa s]: eddy conductivity is cp*mu_t/Pr_t (no rho)."""
    mesh, pi, po, fd = case
    i, j = 3, 20
    assert fd.k_eff[i, j] == pytest.approx(
        pi.k_nf + pi.cp_nf * fd.mu_t[i, j] / 0.85)
    io = fd.i2 + 4
    assert fd.k_eff[io, j] == pytest.approx(
        po.k_f + po.cp_f * fd.mu_t[io, j] / 0.85)
