"""Outer SIMPLEC loop: momentum predictor -> pressure correction -> field
correction -> turbulence update, until mass and velocity tolerances are met."""
import numpy as np

from .momentum import solve_u_momentum, solve_v_momentum
from .pressure_correction import build_and_solve_pressure_correction, correct_fields
from .k_epsilon import (inlet_turbulence, compute_mu_t, compute_production,
                        solve_k, solve_epsilon)


def run_simplec(mesh, rho, mu_molecular, U_in, turbulence_model=None,
                alpha_u=0.7, alpha_v=0.7, alpha_p=1.0,
                alpha_k=0.6, alpha_eps=0.6,
                max_outer_iter=500, mass_tol=1e-6, vel_tol=1e-6, verbose=False):
    """Solve one fluid zone; returns u, v, p (+ k, eps, mu_t for "k_epsilon").

    turbulence_model: None = laminar; "k_epsilon" = standard k-epsilon; or a
    callable(u, mesh, rho, mu_molecular) -> (mu_eff, mu_t), e.g.
    eddy_viscosity.mixing_length_viscosity.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    Dh = mesh.Dh                     # hydraulic diameter (pipe or annulus)

    u = np.full((Nr, Nz + 1), U_in)
    v = np.zeros((Nr + 1, Nz))
    p = np.zeros((Nr, Nz))

    use_kepsilon = (turbulence_model == "k_epsilon")

    if use_kepsilon:
        k_in, eps_in = inlet_turbulence(U_in, Dh)
        k = np.full((Nr, Nz), k_in)
        eps = np.full((Nr, Nz), eps_in)
        mu_t = compute_mu_t(k, eps, rho)

    history = {"mass_residual": [], "max_du": []}

    for outer in range(max_outer_iter):
        if turbulence_model is None:
            mu_eff_cells = np.full((Nr, Nz), mu_molecular)
        elif use_kepsilon:
            mu_eff_cells = mu_molecular + mu_t
        else:
            mu_eff_cells, _ = turbulence_model(u, mesh, rho, mu_molecular)

        u_star, aP_u, sumnb_u = solve_u_momentum(
            mesh, u, v, p, rho, mu_eff_cells, U_in, alpha_u=alpha_u,
            mu_molecular=mu_molecular, use_wall_function=(turbulence_model is not None))
        v_star, aP_v, sumnb_v = solve_v_momentum(mesh, u, v, p, rho, mu_eff_cells,
                                                 alpha_v=alpha_v)

        p_prime, d_e, d_n, mass_res = build_and_solve_pressure_correction(
            mesh, u_star, v_star, aP_u, sumnb_u, aP_v, sumnb_v, rho)

        u_new, v_new, p_new = correct_fields(mesh, u_star, v_star, p, p_prime,
                                             d_e, d_n, alpha_p=alpha_p)

        if use_kepsilon:
            Gk = compute_production(u_new, mesh, mu_t, rho, mu_molecular)
            k_new = solve_k(mesh, u_new, v_new, rho, mu_molecular, mu_t, k, eps,
                            Gk, k_in, alpha_k=alpha_k)
            eps_new = solve_epsilon(mesh, u_new, v_new, rho, mu_molecular, mu_t,
                                    k_new, eps, Gk, eps_in, alpha_eps=alpha_eps)
            mu_t = compute_mu_t(k_new, eps_new, rho)
            k, eps = k_new, eps_new

        max_du = np.max(np.abs(u_new - u))
        history["mass_residual"].append(mass_res)
        history["max_du"].append(max_du)

        u, v, p = u_new, v_new, p_new

        if verbose and outer % 20 == 0:
            msg = f"  iter {outer:4d}  mass_res={mass_res:.3e}  max_du={max_du:.3e}"
            if use_kepsilon:
                msg += f"  mu_t_max={mu_t.max():.3e}"
            print(msg)

        mdot_in = rho * U_in * mesh.A_e[:, 0].sum()
        if mass_res / mdot_in < mass_tol and max_du < vel_tol:
            break

    result = {"u": u, "v": v, "p": p, "iterations": outer + 1, "history": history,
              "converged": (mass_res / mdot_in < mass_tol and max_du < vel_tol)}
    if use_kepsilon:
        result["k"] = k
        result["eps"] = eps
        result["mu_t"] = mu_t
    return result
