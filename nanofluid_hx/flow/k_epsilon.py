"""Standard k-epsilon transport equations, solved segregated with lagged sources.

mu_t = Cmu * rho * k^2 / eps. Near-wall rows use equilibrium wall functions:
k gets wall-function production, epsilon is prescribed (wall_epsilon), not
transported.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .wall_function import wall_k_production_row, wall_epsilon_row


CMU = 0.09

SIGMA_K, SIGMA_EPS = 1.0, 1.3
C1EPS, C2EPS       = 1.44, 1.92
K_FLOOR, EPS_FLOOR = 1e-10, 1e-10


def u_at_cell_centers(u):
    return 0.5 * (u[:, :-1] + u[:, 1:])


def inlet_turbulence(U_in, D, intensity=0.05, mixing_length_frac=0.07):
    """Standard turbulence-intensity-based inlet k, epsilon. D = hydraulic diameter."""
    k_in   = 1.5 * (intensity * U_in) ** 2
    l_mix  = mixing_length_frac * D
    eps_in = (CMU ** 0.75) * (k_in ** 1.5) / l_mix
    return max(k_in, K_FLOOR), max(eps_in, EPS_FLOOR)


def compute_mu_t(k, eps, rho):
    k_safe   = np.maximum(k, 0.0)
    eps_safe = np.maximum(eps, EPS_FLOOR)
    return CMU * rho * k_safe ** 2 / eps_safe


def wall_rows(mesh):
    """Radial row indices that sit against a solid wall."""
    rows = [mesh.Nr - 1]                     # north boundary is always a wall
    if mesh.south_is_wall:
        rows.append(0)
    return rows


def wall_distance(mesh, i):
    """Distance from wall-row cell center to its wall."""
    if i == mesh.Nr - 1:
        return mesh.R - mesh.r_center[i]
    return mesh.r_center[i] - mesh.r_faces[0]


def compute_production(u, mesh, mu_t, rho, mu_molecular):
    """Gk at cell centers: mu_t*(du/dr)^2, wall rows use wall-function production instead."""
    Nr, Nz = mesh.Nr, mesh.Nz
    u_c    = u_at_cell_centers(u)

    du_dr = np.zeros((Nr, Nz))
    if mesh.south_is_wall:
        du_dr[0, :] = (u_c[1, :] - 0.0) / (mesh.r_center[1] - mesh.r_faces[0])
    else:
        du_dr[0, :] = 0.0                    # symmetry axis
    du_dr[Nr - 1, :] = (0.0 - u_c[Nr - 2, :]) / (mesh.R - mesh.r_center[Nr - 2])
    du_dr[1:Nr - 1, :] = ((u_c[2:, :] - u_c[:-2, :])
                          / (mesh.r_center[2:] - mesh.r_center[:-2])[:, None])

    Gk = mu_t * du_dr ** 2

    for i_wall in wall_rows(mesh):
        y_P = wall_distance(mesh, i_wall)
        Gk[i_wall, :] = wall_k_production_row(u_c[i_wall, :], y_P, rho, mu_molecular)

    return Gk


def assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit,
                       sink_coefficient, phi_in, dirichlet_rows=None,
                       dirichlet_values=None):
    """Generic cell-centered convection-diffusion-source solve using the staggered
    mesh's exact face velocities (no re-interpolation):

        a_P * phi_P = sum(a_nb * phi_nb) + source_explicit * V
        a_P also includes + sink_coefficient * V (linearized destruction).

    dirichlet_rows / dirichlet_values: fixed-value override for given wall rows.
    """
    Nr, Nz = mesh.Nr, mesh.Nz

    F_w = rho * u[:, :-1] * mesh.A_e
    F_e = rho * u[:, 1:] * mesh.A_e
    F_s = rho * v[:-1, :] * mesh.A_s
    F_n = rho * v[1:, :] * mesh.A_n

    # Upwind convection + central diffusion; gamma averaged onto faces.
    gamma_w = 0.5 * (gamma_cells[:, 1:] + gamma_cells[:, :-1])
    dz_cc = (mesh.z_center[1:] - mesh.z_center[:-1])[None, :]

    a_W = np.zeros((Nr, Nz))
    a_W[:, 1:] = (gamma_w * mesh.A_e[:, 1:] / dz_cc
                  + np.maximum(F_w[:, 1:], 0.0))
    dz_in = mesh.z_center[0] - mesh.z_faces[0]          # half-cell to inlet
    D_w0 = gamma_cells[:, 0] * mesh.A_e[:, 0] / dz_in
    a_W[:, 0] = D_w0 + np.maximum(F_w[:, 0], 0.0)

    a_E = np.zeros((Nr, Nz))
    a_E[:, :-1] = (gamma_w * mesh.A_e[:, :-1] / dz_cc
                   + np.maximum(-F_e[:, :-1], 0.0))

    gamma_sn = 0.5 * (gamma_cells[1:, :] + gamma_cells[:-1, :])
    dr_cc = (mesh.r_center[1:] - mesh.r_center[:-1])[:, None]
    a_S = np.zeros((Nr, Nz))
    a_S[1:, :] = (gamma_sn * mesh.A_s[1:, :] / dr_cc
                  + np.maximum(F_s[1:, :], 0.0))
    a_N = np.zeros((Nr, Nz))
    a_N[:-1, :] = (gamma_sn * mesh.A_n[:-1, :] / dr_cc
                   + np.maximum(-F_n[:-1, :], 0.0))
    # i = 0 south boundary: symmetry axis (pipe) or south wall (annulus) --
    # zero diffusive flux for k; epsilon rows are Dirichlet-overridden.

    a_P = a_W + a_E + a_S + a_N
    a_P += sink_coefficient * mesh.V

    b = source_explicit * mesh.V
    b[:, 0] += a_W[:, 0] * phi_in

    is_dir = np.zeros(Nr, dtype=bool)
    if dirichlet_rows is not None:
        is_dir[list(dirichlet_rows)] = True
        b[is_dir, :] = dirichlet_values[is_dir, :]
    solve = np.broadcast_to(~is_dir[:, None], (Nr, Nz))

    diag = np.where(a_P > 1e-12, a_P, 1.0)
    diag[is_dir, :] = 1.0

    # Sparse system in COO form: row = i*Nz + j
    row = np.arange(Nr)[:, None] * Nz + np.arange(Nz)[None, :]
    r_w = row[:, 1:][solve[:, 1:]]
    r_e = row[:, :-1][solve[:, :-1]]
    r_s = row[1:, :][solve[1:, :]]
    r_n = row[:-1, :][solve[:-1, :]]
    rows = np.concatenate([row.ravel(), r_w, r_e, r_s, r_n])
    cols = np.concatenate([row.ravel(), r_w - 1, r_e + 1, r_s - Nz, r_n + Nz])
    data = np.concatenate([diag.ravel(),
                           -a_W[:, 1:][solve[:, 1:]], -a_E[:, :-1][solve[:, :-1]],
                           -a_S[1:, :][solve[1:, :]], -a_N[:-1, :][solve[:-1, :]]])
    A = coo_matrix((data, (rows, cols)), shape=(Nr * Nz, Nr * Nz)).tocsr()

    phi_flat = spsolve(A, b.ravel())
    return phi_flat.reshape((Nr, Nz))


def solve_k(mesh, u, v, rho, mu_molecular, mu_t, k_old, eps_old, Gk, k_in,
            alpha_k=0.6):
    gamma_cells  = mu_molecular + mu_t / SIGMA_K
    k_old_safe   = np.maximum(k_old, K_FLOOR)
    eps_old_safe = np.maximum(eps_old, EPS_FLOOR)

    sink_coefficient = rho * eps_old_safe / k_old_safe    # linearized -rho*eps
    source_explicit  = Gk

    k_star = assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit,
                                sink_coefficient, k_in)
    k_star = np.maximum(k_star, K_FLOOR)
    return k_old + alpha_k * (k_star - k_old)


def solve_epsilon(mesh, u, v, rho, mu_molecular, mu_t, k_new, eps_old, Gk,
                  eps_in, alpha_eps=0.6):
    Nr, Nz       = mesh.Nr, mesh.Nz
    gamma_cells  = mu_molecular + mu_t / SIGMA_EPS
    k_new_safe   = np.maximum(k_new, K_FLOOR)
    eps_old_safe = np.maximum(eps_old, EPS_FLOOR)

    sink_coefficient = C2EPS * rho * eps_old_safe / k_new_safe
    source_explicit  = C1EPS * (eps_old_safe / k_new_safe) * Gk

    dirichlet_values = np.zeros((Nr, Nz))
    for i_wall in wall_rows(mesh):
        y_P = wall_distance(mesh, i_wall)
        dirichlet_values[i_wall, :] = wall_epsilon_row(k_new[i_wall, :], y_P)

    eps_star = assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit,
                                  sink_coefficient, eps_in,
                                  dirichlet_rows=wall_rows(mesh),
                                  dirichlet_values=dirichlet_values)
    eps_star = np.maximum(eps_star, EPS_FLOOR)
    return eps_old + alpha_eps * (eps_star - eps_old)
