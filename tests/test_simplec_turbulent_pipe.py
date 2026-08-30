"""
turbulent sanity check: mixing-length closure + log-law wall function vs. the Blasius 
friction-factor at the Re similar to the paper's operating range.
"""
import pytest

from nanofluid_hx.flow.staggered_mesh import StaggeredPipeMesh
from nanofluid_hx.flow.simple_solver import run_simplec
from nanofluid_hx.flow.eddy_viscosity import mixing_length_viscosity

R = 0.013
L = 2.0
RHO = 997.1
MU = 8.91e-4
D = 2 * R

@pytest.mark.parametrize("Re,rel_tol", [(30000, 0.15), (60000, 0.15), (100000, 0.15)])
def test_turbulent_friction_factor_vs_blasius(Re, rel_tol):
    U_mean = Re * MU / (RHO * D)
    mesh   = StaggeredPipeMesh(R=R, L=L, Nr=25, Nz=80, r_stretch=1.6)
    result = run_simplec(mesh, RHO, MU, U_in=U_mean, turbulence_model=mixing_length_viscosity,
                         alpha_u=0.5, alpha_v=0.5, alpha_p=1.0,
                         max_outer_iter=800, mass_tol=1e-6, vel_tol=1e-7)

    u = result['u']

    ratio= 0.5 * (u[0, -2] + u[0, -1]) / U_mean

    assert 1.1 < ratio < 1.5

    p      = result['p']
    j1, j2 = int(mesh.Nz * 0.5), mesh.Nz - 6
    dp_dz  = (p[0, j2] - p[0, j1]) / (mesh.z_center[j2] - mesh.z_center[j1])

    f_num     = -dp_dz * D / (0.5 * RHO * U_mean ** 2)
    f_blasius = 0.184 * Re ** -0.20

    assert f_num == pytest.approx(f_blasius, rel = rel_tol)