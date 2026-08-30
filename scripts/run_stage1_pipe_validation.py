"""
SIMPLEC validation suite standalone and print a summery table: laminar vs exact Poiseuille,
turbulent (mixing-length + wall function) vs. Blasius.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from nanofluid_hx.flow.staggered_mesh import StaggeredPipeMesh
from nanofluid_hx.flow.simple_solver  import run_simplec
from nanofluid_hx.flow.eddy_viscosity import mixing_length_viscosity

R   = 0.013
RHO = 997.1
MU  = 8.91e-4
D   = 2 * R

def laminar_case(Re):
    U_mean = Re * MU / (RHO * D)
    L      = max(1.0, 1.3 * 0.05 * Re * D)

    mesh   = StaggeredPipeMesh(R=R, L=L, Nr=20, Nz=70, r_stretch=1.3)
    result = run_simplec(mesh, RHO, MU, U_in=U_mean, turbulence_model=None,
                          max_outer_iter=800, mass_tol=1e-6, vel_tol=1e-6)

    u      = result["u"]
    ratio  = 0.5 * (u[0, -2] + u[0, -1]) / U_mean
    p      = result["p"]
    j1, j2 = int(mesh.Nz * 0.55), mesh.Nz - 6
    dpdz   = (p[0, j2] - p[0, j1]) / (mesh.z_center[j2] - mesh.z_center[j1])

    f_num    = -dpdz * D / (0.5 * RHO * U_mean ** 2)
    f_theory = 64.0 / Re

    return ratio, f_num, f_theory

def turbulent_case(Re):
    U_mean = Re * MU / (RHO * D)
    mesh   = StaggeredPipeMesh(R=R, L=2.0, Nr=25, Nz=80, r_stretch=1.6)
    result = run_simplec(mesh, RHO, MU, U_in=U_mean, turbulence_model=mixing_length_viscosity,
                          alpha_u=0.5, alpha_v=0.5, alpha_p=1.0,
                          max_outer_iter=800, mass_tol=1e-6, vel_tol=1e-7)

    u      = result["u"]
    ratio  = 0.5 * (u[0, -2] + u[0, -1]) / U_mean
    p      = result["p"]
    j1, j2 = int(mesh.Nz * 0.5), mesh.Nz - 6
    dpdz   = (p[0, j2] - p[0, j1]) / (mesh.z_center[j2] - mesh.z_center[j1])

    f_num     = -dpdz * D / (0.5 * RHO * U_mean ** 2)
    f_blasius = 0.184 * Re ** -0.20

    return ratio, f_num, f_blasius


if __name__ == "__main__":
    print("=== Laminar: SIMPLEC vs. exact Hagan-Poiseuille ===")
    print(f"{'Re':>8} {'centerline/mean': >16} {'f_numeric':>12} {'f_theory = 64/Re': .12} {'err%':>7}")

    for Re in [200, 500, 1000]:
        ratio, f_num, f_theory = laminar_case(Re)

        err = abs(f_num - f_theory) /f_theory * 100

        print(f"{Re:.8d} {ratio:16.4f} {f_num:12.5f} {f_theory: 16.5} {err:7.2f}")

    print("\n=== Turbulent: mixing-length + wall function vs. Blasius ===")
    print(f"{'Re':>8} {'centerline/mean': >16} {'f_numeric':>12} {'f_theory = 64/Re': .12} {'err%':>7}")

    for Re in [30000, 60000, 100000]:
        ratio, f_num, f_blasius = turbulent_case(Re)

        err = abs(f_num - f_blasius) / f_blasius * 100

        print(f"{Re:.8d} {ratio:16.4f} {f_num:12.5f} {f_blasius: 16.5} {err:7.2f}")

