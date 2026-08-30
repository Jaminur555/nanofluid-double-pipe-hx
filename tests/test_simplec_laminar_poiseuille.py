"""
SIMPLEC-solved laminar pipe flow vs the exact analytical Hagen-POiseulile solution
(closed form, empiricism involved). This is the rigorous check for the momentum +
pressure_correction machinery itself, decoupled from any turbulence-closer uncertainity
"""

import numpy as np
import pytest

from nanofluid_hx.flow.staggered_mesh import StaggeredPipeMesh
from nanofluid_hx.flow.simple_solver import run_simplec

R   = 0.013
RHO = 997.1
MU  = 8.91e-4
D   = 2 * R

@pytest.mark.parametrize("Re", [200, 500, 1000])
def test_laminar_centerline_velocity_and_friction_factor(Re):
    U_mean = Re * MU / (RHO * D)
    L      = max(1.0, 1.3 * 0.05 * Re * D)

    mesh   = StaggeredPipeMesh(R=R, L=L, Nr=20, Nz=70, r_stretch=1.3)
    result = run_simplec(mesh, RHO, MU, U_in=U_mean, turbulence_model=None,
                         max_outer_iter=800, mass_tol=1e-6, vel_tol=1e-6)

    assert result['history']['max_du'][-1] < 1e-3      # solution has settled down

    u = result['u']

    u_centerline = 0.5 *(u[0, -2] + u[0, -1])
    assert u_centerline / U_mean == pytest.approx(2.0, rel=0.02)

    p      = result['p']
    j1, j2 = int(mesh.Nz * 0.55), mesh.Nz - 6
    dp_dz  = (p[0, j2] - p[0, j1]) / (mesh.z_center[j2] - mesh.z_center[j1])

    f_num    = -dp_dz * D / (0.5 * RHO * U_mean ** 2)
    f_theory = 64.0 / Re
    assert f_num == pytest.approx(f_theory, rel = 0.03)
    
