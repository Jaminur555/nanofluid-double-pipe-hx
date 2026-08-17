import matplotlib.pyplot as plt

from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.turbulence import get_model
from nanofluid_hx.postprocessing import evaluate_case
from nanofluid_hx.plotting import save_figure


def analyze_case(parallel_flow, phi, Re):
    """Solve one case; return (Nu_avg, effectiveness)."""
    mesh = AxisymmetricMesh(Nr_inner=15, Nr_wall=5, Nr_outer=15, Nz=150)
    pi = MaterialProperties(phi=phi)
    po = MaterialProperties(phi=0.0)

    fd = get_model("mixing_length")(mesh, pi, po, Re_inner=Re, Re_outer=Re)

    solver = ThermalSolver(mesh, fd, parallel_flow=parallel_flow)
    solver.assemble_system()
    T = solver.solve()

    m = evaluate_case(mesh, fd, T, parallel_flow=parallel_flow,
                      T_hot_in=solver.T_hot_in, T_cold_in=solver.T_cold_in)
    return m["Nu_avg"], m["effectiveness"]


if __name__ == "__main__":
    Re_list  = [10000, 20000, 40000, 80000, 100000]
    phi_list = [0.0, 0.025, 0.05, 0.075, 0.1]

    # Store results
    result={
        'parallel':{phi:{'Re':[], 'Nu': [], 'eff': []} for phi in phi_list},
        'counter' :{phi:{'Re':[], 'Nu': [], 'eff': []} for phi in phi_list}
    }

    for flow_type in ["parallel", "counter"]:
        parallel_flag = (flow_type == 'parallel')
        for phi in phi_list:
            for Re in Re_list:
                Nu, eff = analyze_case(parallel_flag, phi, Re)
                result[flow_type][phi]['Re'].append(Re)
                result[flow_type][phi]['Nu'].append(Nu)
                result[flow_type][phi]['eff'].append(eff)


    fig, ax = plt.subplots(2, 2, figsize = (14, 10))

    colors  = ['blue', 'green', 'orange', 'red', 'purple']

    # Nu_parallel
    for idx, phi in enumerate(phi_list):
        ax[0,0].plot(result['parallel'][phi]['Re'], result['parallel'][phi]['Nu'],
                     'o-', color = colors[idx], label =f'phi: {phi * 100:.1f}%')

    ax[0, 0].set_title("Nusselt Number - Parallel Flow")
    ax[0, 0].set_xlabel("Reynolds Number (Re)")
    ax[0, 0].set_ylabel("Average Nusselt Number (Nu)")
    ax[0, 0].grid(True, alpha = 0.3)
    ax[0, 0].legend()

    # Efficiency parallel
    for idx, phi in enumerate(phi_list):
        ax[0, 1].plot(result['parallel'][phi]['Re'], result['parallel'][phi]['eff'], 
                      '-o', color=colors[idx], label=f'phi = {phi*100:.1f}%')
        
    ax[0, 1].set_title("Thermal Efficiency - Parallel Flow")
    ax[0, 1].set_xlabel("Reynolds Number (Re)")
    ax[0, 1].set_ylabel("Efficiency (Q/Q_max)")
    ax[0, 1].grid(True, alpha = 0.3)
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

    fig.suptitle("Al$_2$O$_3$-Water Nanofluid — Effect of Reynolds Number and "
                 "Volume Fraction (φ = 0 is the pure-water baseline)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "sweep_nu_effectiveness")
    plt.show()



