"""Axial (u) and radial (v) momentum predictors on the staggered mesh.

Upwind convection, central diffusion, implicit (Patankar) under-relaxation.
Returns the solved velocity field plus the relaxed diagonal (aP) and raw
neighbor-sum arrays that pressure_correction.py needs for SIMPLEC's 'd'.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .wall_function import wall_coefficient_row


def solve_u_momentum(mesh, u, v, p, rho, mu_eff_cells, U_in, alpha_u=0.7,
                     mu_molecular=None, use_wall_function=False):
    """Solve the axial-momentum predictor u* for j = 1..Nz-1; u[:, 0] = U_in, outlet zero-gradient.

    use_wall_function=True gives wall rows a log-law resistance instead of a
    literal no-slip diffusive flux. Returns (u_star, aP_u, sumnb_u).
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    dz_p = mesh.z_faces[1:] - mesh.z_faces[:-1]        # pressure-cell widths

    # Coefficients over the unknowns (i, j = 1..Nz-1), arrays of shape (Nr, Nz-1)
    A_ax = mesh.A_e_u[:, None]
    mu_w = mu_eff_cells[:, :-1]                        # mu_eff[i, j-1]
    mu_e = mu_eff_cells[:, 1:]                         # mu_eff[i, j]
    dz_w = dz_p[:-1][None, :]                          # dz_p[j-1]
    dz_e = dz_p[1:][None, :]                           # dz_p[j]
    u_col = u[:, 1:Nz]                                 # u[i, j]

    D_w = mu_w * A_ax / dz_w
    F_w = rho * (0.5 * (u[:, :Nz - 1] + u_col)) * A_ax
    a_W = D_w + np.maximum(F_w, 0.0)

    D_e = mu_e * A_ax / dz_e
    F_e = rho * (0.5 * (u_col + u[:, 2:])) * A_ax
    a_E = np.zeros((Nr, Nz - 1))
    a_E[:, :-1] = D_e[:, :-1] + np.maximum(-F_e[:, :-1], 0.0)  # j = Nz-1: outlet, no resistance

    dz_u_cv = 0.5 * (dz_w + dz_e)

    a_N = np.zeros((Nr, Nz - 1))
    mu_n = 0.5 * (mu_eff_cells[:-1, 1:] + mu_eff_cells[1:, 1:])
    area_n = mesh.An_u_perlen[:-1, None] * dz_u_cv
    dr_n = (mesh.r_center[1:] - mesh.r_center[:-1])[:, None]
    a_N[:-1, :] = mu_n * area_n / dr_n                 # i = 0 pipe: symmetry axis, zero flux

    a_S = np.zeros((Nr, Nz - 1))
    mu_s = 0.5 * (mu_eff_cells[1:, 1:] + mu_eff_cells[:-1, 1:])
    area_s = mesh.As_u_perlen[1:, None] * dz_u_cv
    dr_s = (mesh.r_center[1:] - mesh.r_center[:-1])[:, None]
    a_S[1:, :] = mu_s * area_s / dr_s

    a_P = a_W + a_E + a_N + a_S
    sumnb_u = np.zeros((Nr, Nz + 1))
    sumnb_u[:, 1:Nz] = a_P                             # raw sum, for SIMPLEC's d_u

    def wall_add(perlen, i_wall, u_row):
        # Wall rows target u = 0, so the resistance only enters the diagonal.
        area_wall = perlen[i_wall] * dz_u_cv[0, :]
        dr_wall = (mesh.R - mesh.r_center[i_wall] if i_wall == Nr - 1
                   else mesh.r_center[i_wall] - mesh.r_faces[0])
        if use_wall_function:
            return wall_coefficient_row(u_row, dr_wall, area_wall,
                                        rho, mu_molecular)
        return mu_eff_cells[i_wall, 1:] * area_wall / dr_wall

    add = np.zeros((Nr, Nz - 1))
    add[Nr - 1, :] = wall_add(mesh.An_u_perlen, Nr - 1, u[Nr - 1, 1:Nz])
    if mesh.south_is_wall:
        add[0, :] = wall_add(mesh.As_u_perlen, 0, u[0, 1:Nz])
    a_P = a_P + add

    b = (p[:, :-1] - p[:, 1:]) * A_ax                  # pressure-gradient source
    a_P_relaxed = a_P / alpha_u
    b += (1.0 - alpha_u) * a_P_relaxed * u_col

    # Sparse system in COO form: row = i*(Nz-1) + (j-1)
    N = Nr * (Nz - 1)
    row = np.arange(Nr)[:, None] * (Nz - 1) + np.arange(Nz - 1)[None, :]
    r_d = row.ravel()
    r_w = row[:, 1:].ravel()                           # j >= 2
    r_e = row[:, :-1].ravel()                          # j <= Nz-2
    r_n = row[:-1, :].ravel()
    r_s = row[1:, :].ravel()
    rows = np.concatenate([r_d, r_w, r_e, r_n, r_s])
    cols = np.concatenate([r_d, r_w - 1, r_e + 1, r_n + (Nz - 1), r_s - (Nz - 1)])
    data = np.concatenate([a_P_relaxed.ravel(),
                           -a_W[:, 1:].ravel(), -a_E[:, :-1].ravel(),
                           -a_N[:-1, :].ravel(), -a_S[1:, :].ravel()])
    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    b[:, 0] += a_W[:, 0] * U_in                        # inlet Dirichlet folded into RHS
    u_flat = spsolve(A, b.ravel())

    u_new = u.copy()
    u_new[:, 1:Nz] = u_flat.reshape((Nr, Nz - 1))
    u_new[:, 0] = U_in
    u_new[:, Nz] = u_new[:, Nz - 1]                    # outlet zero-gradient

    aP_u = np.zeros((Nr, Nz + 1))
    aP_u[:, 1:Nz] = a_P_relaxed
    return u_new, aP_u, sumnb_u


def solve_v_momentum(mesh, u, v, p, rho, mu_eff_cells, alpha_v=0.7):
    """Solve the radial-momentum predictor v* for i = 1..Nr-1; v = 0 at both r-boundaries.

    Returns (v_star, aP_v, sumnb_v).
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    dr_p = mesh.r_faces[1:] - mesh.r_faces[:-1]
    dz_p = mesh.z_faces[1:] - mesh.z_faces[:-1]

    # Coefficients over the unknowns (i = 1..Nr-1, j), arrays of shape (Nr-1, Nz)
    r_here = mesh.r_faces[1:Nr][:, None]
    area_r = 2.0 * np.pi * r_here * dz_p[None, :]
    mu_lo = mu_eff_cells[:-1, :]                       # mu_eff[i-1, j]
    mu_hi = mu_eff_cells[1:, :]                        # mu_eff[i, j]
    dr_s = dr_p[:-1][:, None]                          # dr_p[i-1]
    dr_n = dr_p[1:][:, None]                           # dr_p[i]
    v_col = v[1:Nr, :]                                 # v[i, j]

    D_s = mu_lo * area_r / dr_s
    F_s = rho * (0.5 * (v[:Nr - 1, :] + v_col)) * area_r
    a_S = D_s + np.maximum(F_s, 0.0)

    a_N = np.zeros((Nr - 1, Nz))
    D_n = mu_hi[:-1, :] * area_r[:-1, :] / dr_n[:-1]
    F_n = rho * (0.5 * (v[1:Nr - 1, :] + v[2:Nr, :])) * area_r[:-1, :]
    a_N[:-1, :] = D_n + np.maximum(-F_n, 0.0)          # i = Nr-1: north wall, v = 0 fixed

    A_ax = 0.5 * (mesh.A_e[:-1, :] + mesh.A_e[1:, :])  # face area at r_faces[i]
    mu_ax = 0.5 * (mu_lo + mu_hi)                      # mu_eff averaged onto r_faces[i]
    dz_cc = dz_p[:-1][None, :]
    u_bulk = 0.5 * (u[:-1, 1:Nz] + u[1:Nr, 1:Nz])      # u averaged onto r_faces[i], z-faces 1..Nz-1

    D_w = mu_ax[:, :-1] * A_ax[:, 1:] / dz_cc
    F_w = rho * u_bulk * A_ax[:, 1:]
    a_W = np.zeros((Nr - 1, Nz))
    a_W[:, 1:] = D_w + np.maximum(F_w, 0.0)

    D_e = mu_ax[:, :-1] * A_ax[:, :-1] / dz_cc
    F_e = rho * u_bulk * A_ax[:, :-1]
    a_E = np.zeros((Nr - 1, Nz))
    a_E[:, :-1] = D_e + np.maximum(-F_e, 0.0)

    a_P = a_S + a_N + a_W + a_E
    sumnb_v = np.zeros((Nr + 1, Nz))
    sumnb_v[1:Nr, :] = a_P

    # No-slip resistance of the v-CVs touching a wall (target v = 0)
    add = np.zeros((Nr - 1, Nz))
    add[Nr - 2, :] = (mu_eff_cells[Nr - 1, :] * area_r[Nr - 2, :]
                      / (mesh.r_faces[Nr] - mesh.r_center[Nr - 1]))
    if mesh.south_is_wall:
        add[0, :] = (mu_eff_cells[0, :] * area_r[0, :]
                     / (mesh.r_center[0] - mesh.r_faces[0]))
    a_P = a_P + add

    dz_left = np.concatenate(([dz_p[0]], dz_p[:-1]))   # dz_p[j-1] (dz_p[0] at j = 0)
    vol_v = area_r * 0.5 * (dz_left[None, :] + dz_p[None, :])
    a_P += 2.0 * mu_ax * vol_v / np.maximum(r_here ** 2, 1e-12)   # implicit curvature source

    b = (p[:-1, :] - p[1:, :]) * area_r                # pressure-gradient source
    a_P_relaxed = a_P / alpha_v
    b += (1.0 - alpha_v) * a_P_relaxed * v_col

    # Sparse system in COO form: row = (i-1)*Nz + j
    N = (Nr - 1) * Nz
    row = np.arange(Nr - 1)[:, None] * Nz + np.arange(Nz)[None, :]
    r_d = row.ravel()
    r_s = row[1:, :].ravel()                           # i >= 2
    r_n = row[:-1, :].ravel()                          # i <= Nr-2
    r_w = row[:, 1:].ravel()
    r_e = row[:, :-1].ravel()
    rows = np.concatenate([r_d, r_s, r_n, r_w, r_e])
    cols = np.concatenate([r_d, r_s - Nz, r_n + Nz, r_w - 1, r_e + 1])
    data = np.concatenate([a_P_relaxed.ravel(),
                           -a_S[1:, :].ravel(), -a_N[:-1, :].ravel(),
                           -a_W[:, 1:].ravel(), -a_E[:, :-1].ravel()])
    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    v_flat = spsolve(A, b.ravel())

    v_new = v.copy()
    v_new[1:Nr, :] = v_flat.reshape((Nr - 1, Nz))
    v_new[0, :] = 0.0
    v_new[Nr, :] = 0.0

    aP_v = np.zeros((Nr + 1, Nz))
    aP_v[1:Nr, :] = a_P_relaxed
    return v_new, aP_v, sumnb_v
