"""Y+ probe for the grid-independence anomaly (paper Step 0.2 diagnostic).

Reproduces EXACTLY the wall-function inputs the solver uses at the hot wall
(r1, pipe north wall) and the annulus south wall (r2): the same near-wall
u row, the same y_P (cell-center to wall), the same log-law Newton solve.
Prints y+ statistics per grid so the wall-function validity band
(y+ >= ~25-30 for equilibrium log-law WFs) can be checked per (grid, Re).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_parameter_sweep import make_provider                          # noqa: E402

from nanofluid_hx import AxisymmetricMesh                              # noqa: E402
from nanofluid_hx.flow.wall_function import wall_shear_stress_array    # noqa: E402

GRIDS = [12, 15, 18, 21]
CASES = [(1.0e4, 0.0), (1.0e5, 0.1)]


def main(max_iter):
    print(f"{'Re':>7} {'phi':>5} {'Nr':>5} {'yP_r1[mm]':>10} "
          f"{'y+_r1 mean':>11} {'min':>6} {'max':>6} "
          f"{'y+_r2 mean':>11} | {'it_in':>5} {'conv':>5}")
    for Re, phi in CASES:
        for nri in GRIDS:
            mesh = AxisymmetricMesh(Nr_inner=nri, Nr_wall=5,
                                    Nr_outer=round(nri * 16 / 15), Nz=150)
            fd = make_provider("simplec_k_epsilon", mesh, phi, Re,
                               max_outer_iter=max_iter)
            i1, i2 = mesh.Nr_inner, mesh.Nr_inner + mesh.Nr_wall

            yP_r1 = mesh.r_faces[i1] - mesh.r_center[i1 - 1]
            tau = wall_shear_stress_array(fd.result_inner["u"][-1, :], yP_r1,
                                          fd.pi.rho_nf, fd.pi.mu_nf)
            u_tau = np.sqrt(tau / fd.pi.rho_nf)
            yp1 = fd.pi.rho_nf * u_tau * yP_r1 / fd.pi.mu_nf

            yP_r2 = mesh.r_center[i2] - mesh.r_faces[i2]
            tau2 = wall_shear_stress_array(fd.result_outer["u"][0, :], yP_r2,
                                           fd.po.rho_f, fd.po.mu_f)
            u_tau2 = np.sqrt(tau2 / fd.po.rho_f)
            yp2 = fd.po.rho_f * u_tau2 * yP_r2 / fd.po.mu_f

            print(f"{Re:>7.0e} {phi:>5.2f} {nri:>3}/{round(nri*16/15):<2} "
                  f"{yP_r1*1e3:>10.3f} {yp1.mean():>11.1f} {yp1.min():>6.1f} "
                  f"{yp1.max():>6.1f} {yp2.mean():>11.1f} | "
                  f"{fd.iterations['inner']:>5} "
                  f"{str(fd.converged['inner'])[0]:>5}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Near-wall y+ probe")
    parser.add_argument("--max-iter", type=int, default=500)
    main(parser.parse_args().max_iter)
