"""5x5x2 parameter sweep (Re x phi x arrangement) with the chosen flow model.

The flow provider is ARRANGEMENT-AGNOSTIC (fields always parallel-oriented),
so ONE flow solve per (Re, phi) feeds both the parallel and counter thermal
solves -- half the flow-solve cost of the old script.
"""
import argparse

import matplotlib.pyplot as plt

from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.turbulence import get_model
from nanofluid_hx.postprocessing import evaluate_case
from nanofluid_hx.plotting import save_figure


def make_provider(model, mesh, phi, Re, max_outer_iter=300):
    """One flow solve for one (Re, phi); warn (not fail) on non-convergence."""
    pi = MaterialProperties(phi=phi)
    po = MaterialProperties(phi=0.0)

    try:
        fd = get_model(model)(mesh, pi, po, Re_inner=Re, Re_outer=Re,
                              max_outer_iter=max_outer_iter)
    except TypeError:                     # legacy models have no max_outer_iter
        fd = get_model(model)(mesh, pi, po, Re_inner=Re, Re_outer=Re)

    if hasattr(fd, "converged"):
        for zone, ok in fd.converged.items():
            if not ok:
                print(f"WARNING: flow zone '{zone}' did NOT converge "
                      f"(Re={Re}, phi={phi}, iterations={fd.iterations[zone]})")
    return fd


def analyze_case(fd, parallel_flow):
    """Thermal solve from pre-computed (cached) flow fields; (Nu_avg, eff)."""
    mesh = fd.mesh
    solver = ThermalSolver(mesh, fd, parallel_flow=parallel_flow)
    solver.assemble_system()
    T = solver.solve()

    m = evaluate_case(mesh, fd, T, parallel_flow=parallel_flow,
                      T_hot_in=solver.T_hot_in, T_cold_in=solver.T_cold_in)
    return m["Nu_avg"], m["effectiveness"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re x phi x arrangement sweep")
    parser.add_argument("--model", default="simplec_k_epsilon",
                        choices=["simplec_k_epsilon", "mixing_length"])
    parser.add_argument("--smoke", action="store_true",
                        help="1x1 subset (Re=30000, phi=0.05) for a quick check")
    parser.add_argument("--max-iter", type=int, default=300,
                        help="SIMPLEC outer iterations (bump to 500 at high Re)")
    args = parser.parse_args()

    Re_list  = [30000] if args.smoke else [10000, 20000, 40000, 80000, 100000]
    phi_list = [0.05] if args.smoke else [0.0, 0.025, 0.05, 0.075, 0.1]

    mesh = AxisymmetricMesh()                     # built ONCE, reused everywhere

    # Store results
    result = {
        'parallel': {phi: {'Re': [], 'Nu': [], 'eff': []} for phi in phi_list},
        'counter':  {phi: {'Re': [], 'Nu': [], 'eff': []} for phi in phi_list}
    }

    for phi in phi_list:
        for Re in Re_list:
            fd = make_provider(args.model, mesh, phi, Re,
                               max_outer_iter=args.max_iter)
            for flow_type, parallel_flag in (('parallel', True), ('counter', False)):
                Nu, eff = analyze_case(fd, parallel_flag)
                result[flow_type][phi]['Re'].append(Re)
                result[flow_type][phi]['Nu'].append(Nu)
                result[flow_type][phi]['eff'].append(eff)
            print(f"done: model={args.model} Re={Re} phi={phi} "
                  f"Nu_par={result['parallel'][phi]['Nu'][-1]:.2f} "
                  f"Nu_ctr={result['counter'][phi]['Nu'][-1]:.2f}")

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    # Nu_parallel
    for idx, phi in enumerate(phi_list):
        ax[0, 0].plot(result['parallel'][phi]['Re'], result['parallel'][phi]['Nu'],
                      'o-', color=colors[idx], label=f'phi: {phi * 100:.1f}%')

    ax[0, 0].set_title("Nusselt Number - Parallel Flow")
    ax[0, 0].set_xlabel("Reynolds Number (Re)")
    ax[0, 0].set_ylabel("Average Nusselt Number (Nu)")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend()

    # Efficiency parallel
    for idx, phi in enumerate(phi_list):
        ax[0, 1].plot(result['parallel'][phi]['Re'], result['parallel'][phi]['eff'],
                      '-o', color=colors[idx], label=f'phi = {phi*100:.1f}%')

    ax[0, 1].set_title("Thermal Efficiency - Parallel Flow")
    ax[0, 1].set_xlabel("Reynolds Number (Re)")
    ax[0, 1].set_ylabel("Efficiency (Q/Q_max)")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend()

    # Nu Counter
    for idx, phi in enumerate(phi_list):
        ax[1, 0].plot(result['counter'][phi]['Re'], result['counter'][phi]['Nu'],
                      '-o', color=colors[idx], label=f'phi = {phi*100:.1f}%')

    ax[1, 0].set_title("Nusselt Number - Counter Flow")
    ax[1, 0].set_xlabel("Reynolds Number (Re)")
    ax[1, 0].set_ylabel("Average Nusselt Number (Nu)")
    ax[1, 0].grid(True)
    ax[1, 0].legend()

    # Subplot (1,1): Fig 10 - Efficiency Counter
    for idx, phi in enumerate(phi_list):
        ax[1, 1].plot(result['counter'][phi]['Re'], result['counter'][phi]['eff'],
                      '-o', color=colors[idx], label=f'phi = {phi*100:.1f}%')
    ax[1, 1].set_title("Thermal Efficiency - Counter Flow")
    ax[1, 1].set_xlabel("Reynolds Number (Re)")
    ax[1, 1].set_ylabel("Efficiency (Q/Q_max)")
    ax[1, 1].grid(True)
    ax[1, 1].legend()

    fig.suptitle(f"Al$_2$O$_3$-Water Nanofluid — Effect of Reynolds Number and "
                 f"Volume Fraction (flow model: {args.model})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, f"sweep_nu_effectiveness_{args.model}"
                     + ("_smoke" if args.smoke else ""))
    plt.show()
