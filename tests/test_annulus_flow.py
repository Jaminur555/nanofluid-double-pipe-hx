"""
Stage 3a/3b validation: annulus flow on the staggered mesh.

1. from_faces() reproduces constructor geometry exactly (Stage 3's mesh
   alignment with the thermal mesh relies on this).
2. Laminar annulus vs the EXACT analytical solution (closed form, no-slip at
   both walls) -- validates the r_min geometry and the south-wall momentum BC.
3. Turbulent (k-eps) annulus sanity: profile shape, field positivity, friction
   factor within ~12% of Blasius based on D_h. Approximate for an annulus at
   radius ratio 0.6 -- a sanity check, not a precision validation.

Mesh note (Stage 3a/3b verification): the equilibrium wall functions are only
valid with the first cell center in the log layer (y+ ~ 15-150). The turbulent
test therefore uses a UNIFORM radial mesh via from_faces() (first cell y+ ~
26-47 over Re 30000-60000). One-sided power grading (the pipe convention,
which clusters at s=0) puts the ANNULUS inner wall at y+ ~ 4 and inflates the
converged friction factor by 26-120% (mesh study, 2026-09-03); with the
uniform mesh the solver lands within +/-2% of Blasius.
"""
import numpy as np
import pytest

from nanofluid_hx.flow.staggered_mesh import StaggeredPipeMesh
from nanofluid_hx.flow.simple_solver import run_simplec

R2, R3 = 0.015, 0.025          # annulus inner / outer radii [m]
RHO, MU = 997.1, 8.91e-4
DH = 2.0 * (R3 - R2)


def test_from_faces_matches_constructor():
    a = StaggeredPipeMesh(R=R3, L=2.0, Nr=20, Nz=60, r_min=R2, r_stretch=1.5)
    b = StaggeredPipeMesh.from_faces(a.r_faces, a.z_faces, south_is_wall=True)
    assert b.Nr == a.Nr and b.Nz == a.Nz
    assert b.south_is_wall
    for attr in ("V", "A_e", "A_n", "A_s", "A_e_u",
                 "An_u_perlen", "As_u_perlen", "dr_v"):
        assert np.allclose(getattr(b, attr), getattr(a, attr))
    assert np.isclose(b.Dh, DH)


def exact_annulus_solution():
    """
    Closed-form laminar annulus (radius ratio kappa = R2/R3):
        u(r) = (G/4mu) * [R3^2 - r^2 - B*ln(R3/r)],  B = (R3^2-R2^2)/ln(R3/R2)
    Returns (r_max, u_max/U_mean, G_per_U_mean) with G = -dp/dz.
    """
    B = (R3 ** 2 - R2 ** 2) / np.log(R3 / R2)
    r_max = np.sqrt(B / 2.0)
    bracket_flow = R3 ** 4 - R2 ** 4 - B * (R3 ** 2 - R2 ** 2)
    ratio = 2.0 * (R3 ** 2 - r_max ** 2 - B * np.log(R3 / r_max)) \
        * (R3 ** 2 - R2 ** 2) / bracket_flow
    G_per_U = 8.0 * MU * (R3 ** 2 - R2 ** 2) / bracket_flow
    return r_max, ratio, G_per_U


@pytest.mark.parametrize("Re", [500, 1000])
def test_laminar_annulus_vs_exact(Re):
    U_mean = Re * MU / (RHO * DH)
    L = max(1.0, 1.3 * 0.05 * Re * DH)

    mesh = StaggeredPipeMesh(R=R3, L=L, Nr=18, Nz=70, r_min=R2, r_stretch=1.3)
    result = run_simplec(mesh, RHO, MU, U_in=U_mean, turbulence_model=None,
                         max_outer_iter=800, mass_tol=1e-6, vel_tol=1e-6)
    assert result["history"]["max_du"][-1] < 1e-3   # solution has settled

    r_max_exact, ratio_exact, G_per_U = exact_annulus_solution()

    u_c = 0.5 * (result["u"][:, :-1] + result["u"][:, 1:])
    col = u_c[:, -2]                               # developed station
    i_max = int(np.argmax(col))
    i_exact = int(np.argmin(np.abs(mesh.r_center - r_max_exact)))
    assert abs(i_max - i_exact) <= 1               # peak at the right radius
    assert col[i_max] / U_mean == pytest.approx(ratio_exact, rel=0.03)

    # purely axial flow in the DEVELOPED region, no-slip honored at both walls.
    # (the global max |v| is the physical inlet displacement peak, a few % of
    # U_mean for a uniform inlet -- the validated pipe solver shows it too)
    assert np.max(np.abs(result["v"][:, mesh.Nz // 2:])) / U_mean < 1e-3

    # friction: exact pressure gradient for this U_mean
    p = result["p"]
    j1, j2 = int(mesh.Nz * 0.55), mesh.Nz - 6
    dpdz = (p[0, j2] - p[0, j1]) / (mesh.z_center[j2] - mesh.z_center[j1])
    assert -dpdz == pytest.approx(G_per_U * U_mean, rel=0.03)


@pytest.mark.parametrize("Re,rel_tol", [(30000, 0.12), (60000, 0.12)])
def test_turbulent_annulus_kepsilon_sanity(Re, rel_tol):
    U_mean = Re * MU / (RHO * DH)
    # UNIFORM radial mesh (both wall first cells in the log layer, y+ ~ 26-47):
    # equilibrium wall functions are invalid below y+ ~ 15 -- see the mesh note
    # in the module docstring. Nr=16 verified: f within +/-2% of Blasius.
    Nr, Nz = 16, 60
    mesh = StaggeredPipeMesh.from_faces(np.linspace(R2, R3, Nr + 1),
                                        np.linspace(0.0, 2.0, Nz + 1),
                                        south_is_wall=True)
    result = run_simplec(mesh, RHO, MU, U_in=U_mean, turbulence_model="k_epsilon",
                         alpha_u=0.5, alpha_v=0.5, alpha_p=1.0,
                         alpha_k=0.5, alpha_eps=0.5,
                         max_outer_iter=300, mass_tol=1e-6, vel_tol=1e-6)
    assert result["history"]["max_du"][-1] < 5e-4

    u_c = 0.5 * (result["u"][:, :-1] + result["u"][:, 1:])
    col = u_c[:, -2]
    assert 1.1 < np.max(col) / U_mean < 1.4        # turbulent flattening (laminar = 1.51)
    assert 0 < int(np.argmax(col)) < mesh.Nr - 1   # peak strictly between the walls

    assert (result["k"] > 0).all()
    assert (result["eps"] > 0).all()
    assert result["k"].max() < 10.0 * U_mean ** 2

    p = result["p"]
    j1, j2 = int(mesh.Nz * 0.5), mesh.Nz - 6
    dpdz = (p[0, j2] - p[0, j1]) / (mesh.z_center[j2] - mesh.z_center[j1])
    f_num = -dpdz * DH / (0.5 * RHO * U_mean ** 2)
    f_blasius = 0.184 * Re ** -0.20
    assert f_num == pytest.approx(f_blasius, rel=rel_tol)
