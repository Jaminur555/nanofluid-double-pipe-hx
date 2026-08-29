"""
Standard (Launder-Spalding-style) equilibrium log-law wall function.
"""

import numpy as np

KAPPA  = 0.41
E_WALL = 9.8
YPLUS_VISCOUS = 11.63      # intersection of u+ = Y+ and log law

def wall_shear_stress(u_P, y_P, rho, mu_molecular, tol = 1e-10, max_iter = 50):
    """
    u_P    : near-wall cell velocity magnitude [m/s]
    y_P    : distance from the wall to that cell center [m]

    Returns:
    --------
          tau_w >= 0 (the caller applied it against the flow direction).
    """
    u_mag = abs(u_P)
    if u_mag < 1e-10 or y_P <= 0.0:
        return 0.0
    nu = mu_molecular / rho

    u_tau = max(np.sqrt(nu * u_mag / y_P), 1e-8)    # laminar-sublayer initial guess


    for _ in range(max_iter):
        y_plus  = y_P * u_tau / nu
        if y_plus < YPLUS_VISCOUS:
            f     = u_tau * y_plus - u_mag
            df_du = 2.0 * y_plus                   # u+ = y+  =>  u_tau^2*y_P/nu = u_mag
        else:
            f     = (u_tau / KAPPA) * np.log(E_WALL * y_plus) - u_mag
            df_du = (1.0 / KAPPA) * (np.log(E_WALL * y_plus) + 1.0)
        du    = f / df_du
        u_tau = max(u_tau - du, 1e-10)
        if abs(du) < tol:
            break

    return rho * u_tau ** 2


def wall_coefficient(u_P, y_P, area_wall, rho, mu_molecular):
    """
    Standard lagged-coefficient linearization.
    """
    tau_w = wall_shear_stress(u_P, y_P, rho, mu_molecular)
    u_mag = max(abs(u_P), 1e-8)
    return tau_w * area_wall / u_mag