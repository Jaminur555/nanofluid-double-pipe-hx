"""
Outer SIMPLEC iteration: momentum predictor -> pressure correction -> field correction 
-> (optional) turbulence update -> repeat untill the mass residual and velocity change 
both fall below tolerance.
"""

import numpy as np

from .momentum import solve_u_momentum, solve_v_momentum
from .pressure_correction import build_and_solve_pressure_correction, correct_fields


def run_simplec(mesh, rho, mu_molecular, U_in, turbulence_model = None,
                alpha_u = 0.7, alpha_v=0.7, alpha_p=1.0,
                max_outer_iter=500, mass_tol=1e-6, vel_tol=1e-6, verbose=False):
    """
    Returns a dict with the converged u, v, p fileds and convergence history.
    """
    Nr, Nz = mesh.Nr, mesh.Nz

    u = np.full((Nr, Nz + 1), U_in)
    v = np.zeros((Nr + 1, Nz))
    p = np.zeros((Nr, Nz))

    history = {"mass_residual": [], "max_du": []}

    for outer in range(max_outer_iter):
        if turbulence_model is None:
            mu_eff_cells = np.full((Nr, Nz), mu_molecular)
        else:
            mu_eff_cells, _ = turbulence_model(u, mesh, rho, mu_molecular)

        u_star, aP_u, sumnd_u = solve_u_momentum(mesh, u, v, p, rho, mu_eff_cells, U_in,
                                                alpha_u=alpha_u, mu_molecular=mu_molecular,
                                                use_wall_function=(turbulence_model is not None))
        v_star, aP_V, sumnd_v = solve_v_momentum(mesh, u, v, p, rho, mu_eff_cells,
                                                alpha_v=alpha_v)

        p_prime, d_e, d_n, mass_res = build_and_solve_pressure_correction(mesh, u_star, v_star,
                                        aP_u, sumnd_u, aP_V, sumnd_v, rho)

        u_new, v_new, p_new = correct_fields(mesh, u_star, v_star, p, p_prime,
                                            d_e, d_n, alpha_p=alpha_p)

        max_du = np.max(np.abs(u_new - u))
        history["mass_residual"].append(mass_res)
        history["max_du"].append(max_du)

        u, v, p = u_new, v_new, p_new

        if verbose and outer % 20 == 0:
            print(f" iter{outer:4d} mass_res={mass_res:.3e} max_du={max_du:.3e}")

        # Normalize mass residual by inlet mass flow for a scale-free check
        mdot_in = rho * U_in * mesh.A_e[:, 0].sum()
        if mass_res / mdot_in < mass_tol and max_du < vel_tol:
            break

    return {
        "u" : u,
        "v" : v,
        "p" : p,
        "iterations" : outer + 1, 
        "history"    : history,
        "converged"  : (mass_res / mdot_in < mass_tol and max_du < vel_tol)
    }





    