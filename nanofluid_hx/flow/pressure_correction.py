"""SIMPLEC pressure-correction step: build/solve the p' equation and correct u, v, p.

SIMPLEC vs. SIMPLE is only the 'd' coefficient: d = A / (a_P - sum a_nb), with
a_P the implicitly under-relaxed momentum diagonal and sum a_nb the raw
(unrelaxed) neighbor sum from the same equation.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def build_and_solve_pressure_correction(mesh, u_star, v_star, aP_u, sumnb_u,
                                        aP_v, sumnb_v, rho):
    """Solve the pressure-correction equation for p'; returns (p_prime, d_e, d_n, mass_res)."""
    Nr, Nz = mesh.Nr, mesh.Nz

    # d_e[i, j]: east face of pressure cell (i, j), drives u[i, j+1].
    # d_n[i, j]: north face of pressure cell (i, j), drives v[i+1, j].
    d_e = np.zeros((Nr, Nz))
    denom = aP_u[:, 1:Nz] - sumnb_u[:, 1:Nz]
    np.divide(mesh.A_e[:, :-1], denom, out=d_e[:, :-1], where=denom > 1e-12)

    d_n = np.zeros((Nr, Nz))
    denom = aP_v[1:Nr, :] - sumnb_v[1:Nr, :]
    np.divide(mesh.A_n[:-1, :], denom, out=d_n[:-1, :], where=denom > 1e-12)

    F_w = rho * u_star[:, :-1] * mesh.A_e
    F_e = rho * u_star[:, 1:] * mesh.A_e
    F_s = rho * v_star[:-1, :] * mesh.A_s
    F_n = rho * v_star[1:, :] * mesh.A_n
    residual = (F_e - F_w) + (F_n - F_s)
    mass_res = np.linalg.norm(residual)

    # Interior cells (j = 0..Nz-2); the outlet column pins p' = 0.
    a_E = rho * d_e[:, :-1] * mesh.A_e[:, :-1]
    a_W = np.zeros((Nr, Nz - 1))
    a_W[:, 1:] = rho * d_e[:, :-2] * mesh.A_e[:, :-2]
    a_N = np.zeros((Nr, Nz - 1))
    a_N[:-1, :] = rho * d_n[:-1, :-1] * mesh.A_n[:-1, :-1]
    a_S = np.zeros((Nr, Nz - 1))
    a_S[1:, :] = rho * d_n[:-1, :-1] * mesh.A_n[:-1, :-1]
    a_P = a_E + a_W + a_N + a_S

    # Sparse system in COO form: row = i*Nz + j
    row = np.arange(Nr)[:, None] * Nz + np.arange(Nz)[None, :]
    r_d = row.ravel()
    r_i = row[:, :-1].ravel()                          # interior cells
    r_w = row[:, 1:-1].ravel()                         # interior, j > 0
    r_n = row[:-1, :-1].ravel()                        # interior, i < Nr-1
    r_s = row[1:, :-1].ravel()                         # interior, i > 0
    rows = np.concatenate([r_d, r_i, r_w, r_n, r_s])
    cols = np.concatenate([r_d, r_i + 1, r_w - 1, r_n + Nz, r_s - Nz])
    diag = np.ones((Nr, Nz))
    diag[:, :-1] = np.where(a_P > 1e-12, a_P, 1.0)
    data = np.concatenate([diag.ravel(),
                           -a_E.ravel(), -a_W[:, 1:].ravel(),
                           -a_N[:-1, :].ravel(), -a_S[1:, :].ravel()])
    A = coo_matrix((data, (rows, cols)), shape=(Nr * Nz, Nr * Nz)).tocsr()

    B = np.zeros((Nr, Nz))
    B[:, :-1] = -residual[:, :-1]                      # drive mass imbalance to zero

    p_prime = spsolve(A, B.ravel()).reshape((Nr, Nz))
    return p_prime, d_e, d_n, mass_res


def correct_fields(mesh, u_star, v_star, p, p_prime, d_e, d_n, alpha_p = 1.0):
    """Apply SIMPLEC corrections: p += alpha_p*p', u/v corrected via d."""
    Nr, Nz = mesh.Nr, mesh.Nz
    u_new = u_star.copy()
    u_new[:, 1:Nz - 1] += d_e[:, :Nz - 2] * (p_prime[:, :Nz - 2] - p_prime[:, 1:Nz - 1])

    v_new = v_star.copy()
    v_new[1:Nr - 1, :] += d_n[:Nr - 2, :] * (p_prime[:Nr - 2, :] - p_prime[1:Nr - 1, :])

    p_new = p + alpha_p * p_prime
    p_new -= p_new[:, -1:].mean()    # keep outlet column ~0 as the reference

    return u_new, v_new, p_new
