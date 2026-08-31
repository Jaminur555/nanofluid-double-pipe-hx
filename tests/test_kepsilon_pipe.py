""" Standard k-epsilon (with equilibrium wall functions) vs. Blasius friction-factor correlation"""

import pytest

from nanofluid_hx.flow.staggered_mesh import StaggeredPipeMesh
from nanofluid_hx.flow.simple_solver import run_simplec

R, L = 0.13, 2.0
RHO  = 997.1
MU   = 8.91e-4
D    = 2 * R


@pytest.mark.parametrize("Re,rel_tol", [(30000, 1.0), (60000, 0.10), (100000, 0.1)])
def test_kepsilon_friction_factor_vs_blasius(Re, rel_tol):
    U_mean = Re * MU / (RHO * D)
    mesh   = StaggeredPipeMesh(R=R, L=L, Nr=20, Nz=60, r_stretch=1.6)
    result = run_simplec(mesh, RHO, MU, U_in=U_mean, turbulence_model='k_epsilon',
                        alpha_u=0.5, alpha_v=0.5, alpha_p=1.0,
                        alpha_k=0.5, alpha_eps=0.5,
                        max_outer_iter=300, mass_tol=1e-6, vel_tol=1e-6)
    assert result["history"]["max_du"][-1] < 5e-4

    u = result["u"]

    ratio = 0.5 * (u[0, -2] + u[0, -1]) / U_mean
    assert 1.1 < ratio < 1.5

    assert (result['k'] > 0).all()
    assert (result['eps'] > 0).all()
    assert (result['k'].max() < 10.0 * U_mean ** 2)

    p      = result['p']
    j1, j2 = int(mesh.Nz * 0.5), mesh.Nz - 6
    dp_dz  = (p[0, j2] - p[0, j1]) / (mesh.z_center[j2] - mesh.z_center[j1])

    f_num     = -dp_dz * D / (0.5 * RHO * U_mean ** 2)
    f_blasius = 0.184 * Re ** -0.20

    assert f_num == pytest.approx(f_blasius, rel = rel_tol)