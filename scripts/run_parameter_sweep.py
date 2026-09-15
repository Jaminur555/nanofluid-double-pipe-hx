"""6x5x2 parameter sweep (Re x phi x arrangement) with the chosen flow model.

The flow provider is ARRANGEMENT-AGNOSTIC (fields always parallel-oriented),
so ONE flow solve per (Re, phi) feeds both the parallel and counter thermal
solves.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.turbulence import get_model
from nanofluid_hx.postprocessing import evaluate_case
from nanofluid_hx.entropy_generation import (from_case,
                                             integrate_entropy_generation)
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
    """Thermal solve from cached flow fields; (Nu_nf, eff, EGM totals)."""
    mesh = fd.mesh
    solver = ThermalSolver(mesh, fd, parallel_flow=parallel_flow)
    solver.assemble_system()
    T = solver.solve()

    m = evaluate_case(mesh, fd, T, parallel_flow=parallel_flow,
                      T_hot_in=solver.T_hot_in, T_cold_in=solver.T_cold_in)
    s_ht, s_ff = from_case(mesh, fd, T)
    egm = integrate_entropy_generation(mesh, s_ht, s_ff)["total"]
    return (m["Nu_nf"], m["effectiveness"],
            egm["ht"], egm["ff"], egm["S"], egm["Be"])


def flow_diagnostics(fd):
    """(dp_pipe, dp_ann, U_pipe, U_ann) from the SIMPLEC results [Pa, m/s]."""
    def dp_u(res):
        dp = float(np.mean(res["p"][:, 0] - res["p"][:, -1]))
        U = float(np.mean(res["u"][:, 0]))           # uniform inlet column
        return dp, U
    dp_in, u_in = dp_u(fd.result_inner)
    dp_out, u_out = dp_u(fd.result_outer)
    return dp_in, dp_out, u_in, u_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re x phi x arrangement sweep")
    parser.add_argument("--model", default="simplec_k_epsilon",
                        choices=["simplec_k_epsilon"])
    parser.add_argument("--smoke", action="store_true",
                        help="1x1 subset (Re=30000, phi=0.05) for a quick check")
    parser.add_argument("--max-iter", type=int, default=300,
                        help="SIMPLEC outer iterations (bump to 500 at high Re)")
    args = parser.parse_args()

    Re_list  = [30000] if args.smoke else [10000, 20000, 40000, 60000, 80000,
                                            100000]
    phi_list = [0.05] if args.smoke else [0.0, 0.025, 0.05, 0.075, 0.1]

    mesh = AxisymmetricMesh()                     # built ONCE, reused everywhere

    result = {
        'parallel': {phi: {'Re': [], 'Nu': [], 'eff': []} for phi in phi_list},
        'counter':  {phi: {'Re': [], 'Nu': [], 'eff': []} for phi in phi_list}
    }
    rows = []                                     # CSV dump rows (Re, phi, Nu, eff)

    for phi in phi_list:
        for Re in Re_list:
            fd = make_provider(args.model, mesh, phi, Re,
                               max_outer_iter=args.max_iter)
            dp_in, dp_ann, u_in, u_ann = flow_diagnostics(fd)
            row = {'Re': Re, 'phi': phi,
                   'dP_pipe': dp_in, 'dP_ann': dp_ann,
                   'U_pipe': u_in, 'U_ann': u_ann}
            for flow_type, parallel_flag in (('parallel', True), ('counter', False)):
                Nu, eff, S_ht, S_ff, S, Be = analyze_case(fd, parallel_flag)
                sfx = 'par' if parallel_flag else 'ctr'
                result[flow_type][phi]['Re'].append(Re)
                result[flow_type][phi]['Nu'].append(Nu)
                result[flow_type][phi]['eff'].append(eff)
                row.update({f'Nu_{sfx}': Nu, f'eff_{sfx}': eff,
                            f'S_ht_{sfx}': S_ht, f'S_ff_{sfx}': S_ff,
                            f'S_{sfx}': S, f'Be_{sfx}': Be})
            rows.append(row)
            print(f"done: model={args.model} Re={Re} phi={phi} "
                  f"Nu_par={row['Nu_par']:.2f} Nu_ctr={row['Nu_ctr']:.2f} "
                  f"S_par={row['S_par']:.2f} S_ctr={row['S_ctr']:.2f} "
                  f"dP={dp_in:.0f}/{dp_ann:.0f} Pa")

    csv_path = Path("results") / (
        f"sweep_nu_effectiveness_{args.model}" + ("_smoke" if args.smoke else "")
        + ".csv")
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path}")

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for idx, phi in enumerate(phi_list):
        ax[0, 0].plot(result['parallel'][phi]['Re'], result['parallel'][phi]['Nu'],
                      'o-', color=colors[idx], label=f'phi: {phi * 100:.1f}%')

    ax[0, 0].set_title("Nusselt Number - Parallel Flow")
    ax[0, 0].set_xlabel("Reynolds Number (Re)")
    ax[0, 0].set_ylabel("Average Nusselt Number (Nu)")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend()

    for idx, phi in enumerate(phi_list):
        ax[0, 1].plot(result['parallel'][phi]['Re'], result['parallel'][phi]['eff'],
                      '-o', color=colors[idx], label=f'phi = {phi*100:.1f}%')

    ax[0, 1].set_title("Thermal Efficiency - Parallel Flow")
    ax[0, 1].set_xlabel("Reynolds Number (Re)")
    ax[0, 1].set_ylabel("Efficiency (Q/Q_max)")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend()

    for idx, phi in enumerate(phi_list):
        ax[1, 0].plot(result['counter'][phi]['Re'], result['counter'][phi]['Nu'],
                      '-o', color=colors[idx], label=f'phi = {phi*100:.1f}%')

    ax[1, 0].set_title("Nusselt Number - Counter Flow")
    ax[1, 0].set_xlabel("Reynolds Number (Re)")
    ax[1, 0].set_ylabel("Average Nusselt Number (Nu)")
    ax[1, 0].grid(True)
    ax[1, 0].legend()

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
