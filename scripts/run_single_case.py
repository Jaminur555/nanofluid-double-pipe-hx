"""Run one parallel and one counter case; save temperature contours to results/."""
import matplotlib.pyplot as plt

from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.turbulence import get_model
from nanofluid_hx.plotting import plot_temperature_field, save_figure


def run_case(parallel_flow=True):
    print(f"----- Running: {'Parallel' if parallel_flow else 'Counter'} flow -----")

    mesh = AxisymmetricMesh(Nr_inner=15, Nr_wall=5, Nr_outer=15, Nz=150)

    props_inner = MaterialProperties(phi=0.05)      # Al2O3-water, 5 vol%
    props_outer = MaterialProperties(phi=0)         # pure water

    fd = get_model("mixing_length")(mesh, props_inner, props_outer,
                                    Re_inner=30000, Re_outer=30000)

    solver = ThermalSolver(mesh, fd, parallel_flow=parallel_flow)
    solver.assemble_system()

    T_field = solver.solve()

    print("Simulation completed successfully")
    return mesh, T_field


if __name__ == "__main__":
    
    for parallel in (True, False):
        mesh, T = run_case(parallel_flow=parallel)
        fig, _ = plot_temperature_field(mesh, T, parallel_flow=parallel)
        save_figure(fig, f"{'parallel' if parallel else 'counter'}_contour")
    plt.show()
