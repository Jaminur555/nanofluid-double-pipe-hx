"""
SIMPLEC pressure-correction step.

Given predicted (u*, v*) that not yet satisfy continuity, build and solve the pressure-correction 
equation, then corrected u, v, p.

SIMPLEC vs. SIMPLE differs only in the 'd' coefficient relating a velocity correction to a press-
ure-correction gradient:
                        SIMPLE:   d = A / a_P
                        SIMPLEC:  d = A / (a_P - sum(a_nb))

a_P must be IMPLICITLY under-relaxed momentum diagonal; sum(a_nb) is the raw (unrelaxed) neighvor-
coefficient sum from that same equation. Both are returned by solve_u_momentum / solve_v_momentum
"""

import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def build_and_solve_pressure_correction(mesh, u_star, v_star, aP_u, sumnb_u,
                                        aP_v, sumnb_v, rho):
    Nr, Nz = mesh.Nr, mesh.Nz
    N_eq   = Nr * Nz

    def idx(i, j):
        return i * Nz + j


    # SIMPLEX face coefficents
    # d_e[i, j]: east face of pressure cell(i, j), used u[i, j+1]
    # d_n[i, j]: north face of pressure cell (i, j), used v[i+1, j]
    d_e = np.zeros((Nr, Nz))
    d_n = np.zeros((Nr, Nz))

    for i in range(Nr):
        for j in range(Nz - 1):
            denom = aP_u[i, j + 1] - sumnb_u[i, j + 1]
            d_e[i, j] = mesh.A_e[i, j] / denom if denom > 1e-12 else 0.0


    for i in range(Nr - 1):
        for j in range(Nz):
            denom = aP_v[i + 1, j] - sumnb_v[i + 1, j]
            d_n[i, j] = mesh.A_n[i, j] / denom if denom > 1e-12 else 0.0

    A = lil_matrix((N_eq, N_eq))
    B = np.zeros(N_eq)

    mass_res = np.zeros((Nr, Nz))

    for i in range(Nr):
        for j in range(Nz):
            row = idx(i,j)

            F_w = rho * u_star[i, j] * mesh.A_e[i, j]
            F_e = rho * u_star[i, j + 1] * mesh.A_e[i, j]
            F_s = rho * v_star[i, j] * mesh.A_s[i, j]
            F_n = rho * v_star[i + 1, j] * mesh.A_n[i, j]

            residual       = (F_e - F_w) + (F_n - F_s) 
            mass_res[i, j] =  residual

            if j == Nz - 1:
                # Outlet cell: p' pinned to 0 (reference pressure boundary).
                A[row, row] = 1.0
                B[row]      = 0.0
                continue

            a_E = rho * d_e[i, j] * mesh.A_e[i, j]
            a_W = rho * d_e[i, j - 1] * mesh.A_e[i, j - 1] if j > 0 else 0.0
            a_N = rho * d_n[i, j] * mesh.A_n[i, j] if i < Nr - 1 else 0.0
            a_S = rho * d_n[i - 1, j] * mesh.A_n[i - 1, j] if i > 0 else 0.0

            a_P = a_E + a_W + a_N + a_S
            A[row, row] = a_P if a_P > 1e-12 else 1.0
            if j > 0:
                A[row, idx(i, j - 1)] = -a_W
            A[row, idx(i, j + 1)] = -a_E
            if i > 0:
                A[row, idx(i - 1, j)] = -a_S
            if i < Nr - 1:
                A[row, idx(i + 1, j)] = -a_N

            B[row] = -residual       # drive mass imbalance to zero

    p_flat = spsolve(A.tocsr(), B)
    p_prime = p_flat.reshape((Nr, Nz))

    return p_prime, d_e, d_n, np.linalg.norm(mass_res)


def correct_fields(mesh, u_star, v_star, p, p_prime, d_e, d_n, alpha_p = 1.0):
    """
    Apply SIMPLEC corrections: p += alpha_p * p', u /v corrected dia d.
    """
    Nr, Nz = mesh.Nr, mesh.Nz

    u_new = u_star.copy()
    for i in range(Nr):
        for j in range (1, Nz - 1):
            u_new[i, j] += d_e[i, j - 1] * (p_prime[i, j - 1] - p_prime[i, j])

    v_new = v_star.copy()
    for i in range(1, Nr - 1):
        for j in range (Nz):
            v_new[i, j] += d_n[i - 1, j] * (p_prime[i - 1, j] - p_prime[i, j])

    p_new = p + alpha_p * p_prime
    p_new -= p_new[:, -1:].mean()    # keep outle column ~0 as the reference

    return u_new, v_new, p_new