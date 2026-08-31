"""
Standard k-epsilon turbulence transport equation, solved on the same cell-centered
locations as pressure and temperature.
    k:   div(rho*k*u)   = div((mu + mu_t/ sigma_k) + Gk - rho*eps)
    eps: div(rho*eps*u) = div((mu + mu_t/sigma_eps) grad eps) + C1eps*(eps/k)*Gk
                                                             - C2eps*rho*eps^2/k
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
    """Standard turbulence-intensity based inlet k, epsilon."""
    k_in  = 1.5 * (intensity * U_in) ** 2

    l_mix  = mixing_length_frac * D
    eps_in = (CMU ** 0.75) * (k_in ** 1.5) / l_mix

    return max(k_in, K_FLOOR), max(eps_in, EPS_FLOOR) 


def compute_mu_t(k, eps, rho):
    k_safe   = np.maximum(k, 0.0)         # to ensure numerical safety
    eps_safe = np.maximum(eps, EPS_FLOOR)
    return CMU * rho * k_safe ** 2 / eps_safe 


def compute_production(u, mesh, mu_t, rho, mu_molecular):
    """Gk at every cell center, Except the wall row."""
    Nr, Nz = mesh.Nr, mesh.Nz

    u_c = u_at_cell_centers(u)

    du_dr = np.zeros((Nr, Nz))
    for i in range(Nr):
        if i == 0:
            du_dr[i,:] = 0.0
        elif i == Nr -1:
            dr = mesh.R - mesh.r_center[i - 1]
            du_dr[i, :] = (0.0 - u_c[i - 1, :]) / dr
        else:
            dr = mesh.r_center[i + 1] - mesh.r_center[i - 1]
            du_dr[i, :] = (u_c[i + 1, :] - u_c[i - 1, :]) / dr
    Gk = mu_t * du_dr ** 2

    i_wall = Nr - 1
    y_P    = mesh.R - mesh.r_center[i_wall]
    for j in range(Nz):
        Gk[i_wall, j] = wall_k_production(u_c[i_wall, j], y_P, rho, mu_molecular)

    return Gk


def assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit,
                       sink_coefficient, phi_in, wall_dirichlet=None):
    Nr, Nz = mesh.Nr, mesh.Nz
    N      = Nr * Nz

    def idx(i, j):          # flatten 2D grid into 1D
        return i * Nz + j

    A = lil_matrix((N, N))
    B = np.zeros(N)

    for i in range(Nr):
        for j in range(Nz):
            row = idx(i, j)

            if wall_dirichlet is not None and i == Nr -1:
                A[row, row] = 1.0
                B[row]      = wall_dirichlet[j]
                continue

            F_w = rho * u[i, j] * mesh.A_e[i, j]
            F_e = rho * u[i, j + 1] * mesh.A_e[i, j]
            F_s = rho * v[i, j] * mesh.A_s[i, j] if i > 0 else 0.0
            F_n = rho * v[i + 1, j] * mesh.A_n[i, j] if i < Nr - 1 else 0.0

            a_W = 0.0
            if j > 0:
                gamma_w = 0.5 * (gamma_cells[i, j] + gamma_cells[i, j - 1])

                dz_w = mesh.z_center[j] - mesh.z_center[j - 1]
                D_w  = gamma_w * mesh.A_e[i, j] / dz_w
                a_W  = D_w + max(F_w, 0.0) 
            else:
                dz_w = mesh.z_center[0] - mesh.z_faces[0]
                D_w  = gamma_cells[i, 0] * mesh.A_e[i, 0] / dz_w
                a_W  = D_w + max(F_w, 0.0)

            a_E = 0.0
            if j < Nz - 1:
                gamma_e = 0.5 * (gamma_cells[i, j] + gamma_cells[i, j + 1])

                dz_e = mesh.z_center[j + 1] - mesh.z_center[j]
                D_e  = gamma_e * mesh.A_e[i, j] / dz_e
                a_E  = D_e + max(-F_e, 0.0) 

            a_S = 0.0
            if i > 0:
                gamma_s = 0.5 * (gamma_cells[i, j] + gamma_cells[i - 1, j])

                dr_s = mesh.r_center[i] - mesh.r_center[i - 1]
                D_s  = gamma_s * mesh.A_s[i, j] /dr_s
                a_S  = D_s + max(F_s, 0.0) 

            a_N = 0.0
            if i < Nr - 1:
                gamma_n = 0.5 * (gamma_cells[i, j] + gamma_cells[i + 1, j])

                dr_n = mesh.r_center[i + 1] - mesh.r_center[i]
                D_n  = gamma_n * mesh.A_n[i, j] /dr_n
                a_N  = D_n + max(-F_n, 0.0) 

            a_P = a_W + a_E + a_N + a_S
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


def solve_k(mesh, u, v, rho, mu_molecular, mu_t, k_old, eps_old, 
            Gk, k_in, alpha_k=0.6):
    gamma_cells  = mu_molecular + mu_t / SIGMA_K
    k_old_safe   = np.maximum(k_old, K_FLOOR)
    eps_old_safe = np.maximum(eps_old, EPS_FLOOR)

    sink_coefficient = rho * eps_old_safe / k_old_safe
    source_explicit  = Gk

    _k_star = assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit, 
                                sink_coefficient, k_in, wall_dirichlet=None)
    k_star = np.maximum(_k_star, K_FLOOR)

    return k_old + alpha_k * (k_star - k_old)


def solve_epsilon(mesh, u, v, rho, mu_molecular, mu_t, k_new, eps_old, Gk,
                  eps_in , alpha_eps=0.6):
    Nr, Nz,      = mesh.Nr, mesh.Nz
    gamma_cells  = mu_molecular + mu_t / SIGMA_EPS
    k_new_safe   = np.maximum(k_new, K_FLOOR)
    eps_old_safe = np.maximum(eps_old, EPS_FLOOR)

    sink_coefficient = C2EPS * rho * eps_old_safe / k_new_safe
    source_explicit = C1EPS * (eps_old_safe / k_new_safe) * Gk 

    wall_dirichlet = np.zeros(Nz)

    i_wall = Nr - 1
    y_P    = mesh.R - mesh.r_center[i_wall]
    for j in range(Nz):
        wall_dirichlet[j] = wall_epsilon(k_new[i_wall, j], y_P)


    _eps_star = assemble_and_solve(mesh, u, v, rho, gamma_cells, source_explicit,
                                  sink_coefficient, eps_in, wall_dirichlet=wall_dirichlet)
    eps_star = np.maximum(_eps_star, EPS_FLOOR)
    return eps_old + alpha_eps * (eps_star - eps_old)
    


    


    

            



    
