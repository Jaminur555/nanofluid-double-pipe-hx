"""Standard (Launder-Spalding-style) equilibrium log-law wall function."""

import numpy as np

KAPPA  = 0.41
E_WALL = 9.8

YPLUS_VISCOUS = 11.63      # intersection of u+ = Y+ and log law

CMU = 0.09


def wall_shear_stress(u_P, y_P, rho, mu_molecular, tol = 1e-10, max_iter = 50):
    """Newton solve of the log law for tau_w [Pa] from near-wall velocity u_P [m/s] at distance y_P [m]."""
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


def wall_shear_stress_array(u_P, y_P, rho, mu_molecular, tol=1e-10, max_iter=50):
    """Elementwise wall_shear_stress over an array of u_P (same scalar y_P).

    Identical arithmetic to the scalar version: the Newton update is applied
    only to elements that have not yet hit |du| < tol.
    """
    u_P = np.asarray(u_P, dtype=float)
    u_mag = np.abs(u_P)
    if y_P <= 0.0:
        return np.zeros_like(u_mag)
    nu = mu_molecular / rho

    u_tau = np.maximum(np.sqrt(nu * u_mag / y_P), 1e-8)
    done = u_mag < 1e-10                     # zero-velocity cells return tau_w = 0

    for _ in range(max_iter):
        y_plus = y_P * u_tau / nu
        log_ey = np.log(E_WALL * y_plus)
        visc = y_plus < YPLUS_VISCOUS
        f  = np.where(visc, u_tau * y_plus - u_mag,
                      (u_tau / KAPPA) * log_ey - u_mag)
        df = np.where(visc, 2.0 * y_plus,
                      (1.0 / KAPPA) * (log_ey + 1.0))
        du = f / df
        u_tau = np.where(done, u_tau, np.maximum(u_tau - du, 1e-10))
        done |= np.abs(du) < tol
        if done.all():
            break

    # rho*t**2 via Python floats: CPython's ** uses libm pow, which can differ
    # by 1 ulp from NumPy's array **2 fast path (x*x) -- keep the scalar result.
    squares = np.array([rho * t ** 2 for t in u_tau.tolist()], dtype=float)
    return np.where(u_mag < 1e-10, 0.0, squares)


def wall_coefficient(u_P, y_P, area_wall, rho, mu_molecular):
    """Lagged-coefficient linearization: tau_w * A / |u_P|."""
    tau_w = wall_shear_stress(u_P, y_P, rho, mu_molecular)
    u_mag = max(abs(u_P), 1e-8)
    return tau_w * area_wall / u_mag


def wall_coefficient_row(u_P, y_P, area_wall, rho, mu_molecular):
    """Elementwise wall_coefficient over a row of near-wall cells."""
    tau_w = wall_shear_stress_array(u_P, y_P, rho, mu_molecular)
    u_mag = np.maximum(np.abs(u_P), 1e-8)
    return tau_w * area_wall / u_mag


def wall_k_production(u_P, y_P, rho, mu_molecular):
    """Equilibrium wall-function production of k in the near-wall cell."""
    tau_W = wall_shear_stress(u_P, y_P, rho, mu_molecular)
    u_tau = np.sqrt(tau_W/ rho) if tau_W > 0 else 0.0

    if u_tau < 1e-12 or y_P <= 0.0:
        return 0.0
    return tau_W * u_tau / (KAPPA * y_P)


def wall_k_production_row(u_P, y_P, rho, mu_molecular):
    """Elementwise wall_k_production over a row of near-wall cells."""
    tau_W = wall_shear_stress_array(u_P, y_P, rho, mu_molecular)
    u_tau = np.sqrt(tau_W / rho)
    ok = (u_tau >= 1e-12) & (y_P > 0.0)
    return np.where(ok, tau_W * u_tau / (KAPPA * y_P), 0.0)


def wall_epsilon(k_P, y_P):
    """Equilibrium (prescribed, not transported) epsilon in the near-wall cell."""

    k_P = max(k_P, 1e-12)
    if y_P <= 0.0:
        return 1e-12
    return (CMU ** 0.75) *(k_P ** 1.5) / (KAPPA * y_P)


def wall_epsilon_row(k_P, y_P):
    """Elementwise wall_epsilon over a row of near-wall cells."""
    k_P = np.maximum(k_P, 1e-12)
    if y_P <= 0.0:
        return np.full_like(k_P, 1e-12)
    return (CMU ** 0.75) * (k_P ** 1.5) / (KAPPA * y_P)


def t_plus(y_plus, Pr, Pr_t):
    """Jayatilleke (1969) thermal law of the wall.

    T+ = Pr*y+ in the viscous sublayer (the y+ -> 0 limit is the pure
    molecular-conduction conductance k/y_P); log layer above, with the
    Jayatilleke P function carrying the (Pr/Pr_t)-dependent sublayer
    resistance as an additive term. Same YPLUS_VISCOUS switch as u+.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    P = 9.24 * ((Pr / Pr_t) ** 0.75 - 1.0) * (1.0 + 0.28 * np.exp(-0.007 * Pr / Pr_t))
    sub = y_plus < YPLUS_VISCOUS
    Tp = np.where(sub, Pr * y_plus, (Pr_t / KAPPA) * np.log(y_plus) + P)
    return np.maximum(Tp, 1e-10)


def thermal_wall_h(u_row, y_P, rho, mu, cp, Pr, Pr_t):
    """Wall-function film coefficient h = rho*cp*u_tau/T+ [W/m2 K] per COLUMN.

    u_row is the near-wall staggered velocity (Nz+1 values at the z-faces of
    the node-identical flow sub-mesh); tau_w is taken from the same log-law
    Newton solve the momentum equation used, then averaged onto columns.
    """
    tau = wall_shear_stress_array(u_row, y_P, rho, mu)
    u_tau = np.sqrt(tau / rho)
    u_tau_col = 0.5 * (u_tau[:-1] + u_tau[1:])       # (Nz,) column values
    y_plus = rho * u_tau_col * y_P / mu
    Tp = t_plus(y_plus, Pr, Pr_t)
    return np.maximum(rho * cp * u_tau_col / Tp, 1e-10)
