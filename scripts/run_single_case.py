"""Run one parallel and one counter case; save temperature contours to results/."""
import argparse

import matplotlib.pyplot as plt

from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.turbulence import get_model
from nanofluid_hx.plotting import plot_temperature_field, save_figure


def run_case(parallel_flow=True, model="simplec_k_epsilon", Re=30000, phi=0.05,
             max_outer_iter=300):
    """Solve one case end-to-end; returns (mesh, T_field)."""
    print(f"----- Running: {'Parallel' if parallel_flow else 'Counter'} flow "
          f"(model={model}, Re={Re:.0f}, phi={phi}) -----")

    mesh = AxisymmetricMesh()                     # defaults: 15/5/16 radial, Nz=150

    props_inner = MaterialProperties(phi=phi)     # Al2O3-water nanofluid
    props_outer = MaterialProperties(phi=0)       # pure water

    try:
        fd = get_model(model)(mesh, props_inner, props_outer,
                             Re_inner=Re, Re_outer=Re,
                             max_outer_iter=max_outer_iter)
    except TypeError:                             # legacy models have no max_outer_iter
        fd = get_model(model)(mesh, props_inner, props_outer,
                             Re_inner=Re, Re_outer=Re)

    if hasattr(fd, "converged"):
        print(f"Flow convergence: {fd.converged}  iterations: {fd.iterations}")

    solver = ThermalSolver(mesh, fd, parallel_flow=parallel_flow)
    solver.assemble_system()

    T_field = solver.solve()

    print("Simulation completed successfully")
    return mesh, T_field


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Single-case HX simulation")
    parser.add_argument("--model", default="simplec_k_epsilon",
                        choices=["simplec_k_epsilon"])
    parser.add_argument("--Re", type=float, default=30000)
    parser.add_argument("--phi", type=float, default=0.05)
    parser.add_argument("--counter", action="store_true",
                        help="run only the counterflow case (default: both)")
    parser.add_argument("--max-iter", type=int, default=300,
                        help="SIMPLEC outer iterations (bump to 500 at high Re)")
    args = parser.parse_args()

    cases = [False] if args.counter else [True, False]
    for parallel in cases:
        mesh, T = run_case(parallel_flow=parallel, model=args.model,
                           Re=args.Re, phi=args.phi, max_outer_iter=args.max_iter)
        fig, _ = plot_temperature_field(mesh, T, parallel_flow=parallel)
        save_figure(fig, f"{'parallel' if parallel else 'counter'}_contour_{args.model}")
    plt.show()
