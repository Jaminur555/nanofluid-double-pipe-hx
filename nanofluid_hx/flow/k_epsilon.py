"""
Standard k-epsilon turbulence transport equations (paper's Eqs. 4-6), solved on
the same cell-centered locations as pressure and temperature.

    k:   div(rho*k*u)   = div((mu + mu_t/sigma_k)   grad k)   + Gk - rho*eps
    eps: div(rho*eps*u) = div((mu + mu_t/sigma_eps) grad eps) + C1eps*(eps/k)*Gk
                                                                - C2eps*rho*eps^2/k
    mu_t = Cmu * rho * k^2 / eps

Solved segregated (k then eps, each outer SIMPLEC iteration) with lagged
source-term linearization. Near-wall treatment: standard equilibrium wall
functions paired with wall_function.py's log-law momentum treatment -- k is
transported into the wall row with wall-function production, while epsilon is
NOT solved there: it is prescribed from the current k (wall_epsilon).

Annulus support (Stage 3): every wall row (north always, south when
mesh.south_is_wall) gets the same wall-function treatment.
"""


import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .wall_function import wall_k_production, wall_epsilon


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
    """
    Gk at every cell center: mu_t*(du/dr)^2, EXCEPT wall rows, where the true
    near-wall gradient is not mesh-resolved and the equilibrium wall-function
    production is used instead.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    u_c    = u_at_cell_centers(u)

    du_dr = np.zeros((Nr, Nz))
    for i in range(Nr):
        if i == 0:
            if mesh.south_is_wall:
                dr          = mesh.r_center[1] - mesh.r_faces[0]
                du_dr[i, :] = (u_c[1, :] - 0.0) / dr
            else:
                du_dr[i, :] = 0.0                # symmetry axis
        elif i == Nr - 1:
            dr          = mesh.R - mesh.r_center[i - 1]
            du_dr[i, :] = (0.0 - u_c[i - 1, :]) / dr
        else:
            dr          = mesh.r_center[i + 1] - mesh.r_center[i - 1]
            du_dr[i, :] = (u_c[i + 1, :] - u_c[i - 1, :]) / dr

    Gk = mu_t * du_dr ** 2

    for i_wall in wall_rows(mesh):
        y_P = wall_distance(mesh, i_wall)
        for j in range(Nz):
            Gk[i_wall, j] = wall_k_production(u_c[i_wall, j], y_P, rho, mu_molecular)

    return Gk


def assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit,
                       sink_coefficient, phi_in, dirichlet_rows=None,
                       dirichlet_values=None):
    """
    Generic cell-centered convection-diffusion-source solve using the staggered
    mesh's EXACT face velocities (no re-interpolation):

        a_P * phi_P = sum(a_nb * phi_nb) + source_explicit * V
        a_P also includes + sink_coefficient * V   (linearized destruction)

    dirichlet_rows / dirichlet_values: optional fixed-value override for the
    given wall rows (used for epsilon).
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N      = Nr * Nz

    def idx(i, j):
        return i * Nz + j

    A = lil_matrix((N, N))
    B = np.zeros(N)

    for i in range(Nr):
        for j in range(Nz):
            row = idx(i, j)

            if dirichlet_rows is not None and i in dirichlet_rows:
                A[row, row] = 1.0
                B[row]      = dirichlet_values[i, j]
                continue

            F_w = rho * u[i, j] * mesh.A_e[i, j]
            F_e = rho * u[i, j + 1] * mesh.A_e[i, j]
            F_s = rho * v[i, j] * mesh.A_s[i, j] if i > 0 else 0.0
            F_n = rho * v[i + 1, j] * mesh.A_n[i, j] if i < Nr - 1 else 0.0

            a_W = 0.0
            if j > 0:
                gamma_w = 0.5 * (gamma_cells[i, j] + gamma_cells[i, j - 1])
                dz_w    = mesh.z_center[j] - mesh.z_center[j - 1]
                D_w     = gamma_w * mesh.A_e[i, j] / dz_w
                a_W     = D_w + max(F_w, 0.0)
            else:
                dz_w = mesh.z_center[0] - mesh.z_faces[0]   # half-cell to inlet
                D_w  = gamma_cells[i, 0] * mesh.A_e[i, 0] / dz_w
                a_W  = D_w + max(F_w, 0.0)

            a_E = 0.0
            if j < Nz - 1:
                gamma_e = 0.5 * (gamma_cells[i, j] + gamma_cells[i, j + 1])
                dz_e    = mesh.z_center[j + 1] - mesh.z_center[j]
                D_e     = gamma_e * mesh.A_e[i, j] / dz_e
                a_E     = D_e + max(-F_e, 0.0)

            a_S = 0.0
            if i > 0:
                gamma_s = 0.5 * (gamma_cells[i, j] + gamma_cells[i - 1, j])
                dr_s    = mesh.r_center[i] - mesh.r_center[i - 1]
                D_s     = gamma_s * mesh.A_s[i, j] / dr_s
                a_S     = D_s + max(F_s, 0.0)
            # else: symmetry axis (pipe) or south wall (annulus):
            # zero diffusive flux for k; epsilon row is Dirichlet-overridden.

            a_N = 0.0
            if i < Nr - 1:
                gamma_n = 0.5 * (gamma_cells[i, j] + gamma_cells[i + 1, j])
                dr_n    = mesh.r_center[i + 1] - mesh.r_center[i]
                D_n     = gamma_n * mesh.A_n[i, j] / dr_n
                a_N     = D_n + max(-F_n, 0.0)
            # else: north wall -- zero flux (k) / Dirichlet (eps)

            a_P = a_W + a_E + a_S + a_N
            a_P += sink_coefficient[i, j] * mesh.V[i, j]

            b_p = source_explicit[i, j] * mesh.V[i, j]
            if j == 0:
                b_p += a_W * phi_in

            A[row, row] = a_P if a_P > 1e-12 else 1.0
            if j > 0:
                A[row, idx(i, j - 1)] = -a_W
            if j < Nz - 1:
                A[row, idx(i, j + 1)] = -a_E
            if i > 0:
                A[row, idx(i - 1, j)] = -a_S
            if i < Nr - 1:
                A[row, idx(i + 1, j)] = -a_N

            B[row] = b_p

    phi_flat = spsolve(A.tocsr(), B)
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

    # Prescribed (not transported) epsilon at every wall row
    dirichlet_values = np.zeros((Nr, Nz))
    for i_wall in wall_rows(mesh):
        y_P = wall_distance(mesh, i_wall)
        for j in range(Nz):
            dirichlet_values[i_wall, j] = wall_epsilon(k_new[i_wall, j], y_P)

    eps_star = assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit,
                                  sink_coefficient, eps_in,
                                  dirichlet_rows=wall_rows(mesh),
                                  dirichlet_values=dirichlet_values)
    eps_star = np.maximum(eps_star, EPS_FLOOR)
    return eps_old + alpha_eps * (eps_star - eps_old)
