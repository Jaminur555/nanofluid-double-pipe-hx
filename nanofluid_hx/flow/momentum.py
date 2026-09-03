"""
Axial (u) and radial (v) momentum predictor equations, upwind convection,
central diffusion, on the staggered axisymmetric mesh. Steady RANS with an
eddy-viscosity closure: mu_eff = mu_molecular + mu_t (mu_t supplied per cell).

Returns both the solved velocity field AND the diagonal coefficient arrays
(a_P minus the pressure term) that pressure_correction.py needs to build the
SIMPLEC 'd' coefficients.

Annulus support (Stage 3): when mesh.south_is_wall is True the south boundary
row (i = 0) gets the same wall treatment as the north row.
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def _mu_eff_at(mu_eff_cells, i, j, Nr, Nz):
    """Clamp-safe lookup into the (Nr, Nz) cell-centered effective viscosity."""
    i = min(max(i, 0), Nr - 1)
    j = min(max(j, 0), Nz - 1)
    return mu_eff_cells[i, j]


def solve_u_momentum(mesh, u, v, p, rho, mu_eff_cells, U_in, alpha_u=0.7,
                     mu_molecular=None, use_wall_function=False):
    """
    Solve the axial-momentum predictor u* for interior nodes j = 1..Nz-1.
    u[:, 0] = U_in (inlet, fixed); u[:, Nz] set by zero-gradient after solving.
    Returns (u_star, aP_u, sumnb_u).

    use_wall_function: if True, wall rows get their resistance from a log-law
    wall function instead of a literal no-slip diffusive flux (required once
    the near-wall mesh does not resolve the viscous sublayer).
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    dz_p   = mesh.z_faces[1:] - mesh.z_faces[:-1]           # pressure-cell widths

    N_unknown = Nr * (Nz - 1)                              # interior columns

    def idx(i, jj):  # jj is 0-based index into interior columns (j = jj+1)
        return i * (Nz - 1) + jj

    A = lil_matrix((N_unknown, N_unknown))
    B = np.zeros(N_unknown)

    aP_u    = np.zeros((Nr, Nz + 1))
    sumnb_u = np.zeros((Nr, Nz + 1))

    for i in range(Nr):
        for j in range(1, Nz):
            jj = j - 1
            row = idx(i, jj)

            # --- axial (east/west) neighbors ---
            mu_w = _mu_eff_at(mu_eff_cells, i, j - 1, Nr, Nz)
            mu_e = _mu_eff_at(mu_eff_cells, i, j, Nr, Nz)
            A_ax = mesh.A_e_u[i]

            dz_w = dz_p[j - 1]
            dz_e = dz_p[j] if j < Nz else dz_p[j - 1]

            D_w      = mu_w * A_ax / dz_w
            u_face_w = 0.5 * (u[i, j - 1] + u[i, j])

            F_w = rho * u_face_w * A_ax
            a_W = D_w + max(F_w, 0.0)

            a_E = 0.0
            if j < Nz - 1:
                D_e      = mu_e * A_ax / dz_e
                u_face_e = 0.5 * (u[i, j] + u[i, j + 1])
                F_e      = rho * u_face_e * A_ax
                a_E      = D_e + max(-F_e, 0.0)
            # else: outlet zero-gradient, no extra resistance

            # --- radial (north/south) neighbors ---
            dz_u_cv = 0.5 * (dz_p[j - 1] + (dz_p[j] if j < Nz else dz_p[j - 1]))

            a_N = 0.0
            if i < Nr - 1:
                mu_n   = 0.5 * (mu_eff_cells[i, min(j, Nz - 1)] + mu_eff_cells[i + 1, min(j, Nz - 1)])
                area_n = mesh.An_u_perlen[i] * dz_u_cv
                dr_n   = mesh.r_center[i + 1] - mesh.r_center[i]
                a_N    = mu_n * area_n / dr_n
            # else: north wall handled below

            a_S = 0.0
            if i > 0:
                mu_s   = 0.5 * (mu_eff_cells[i, min(j, Nz - 1)] + mu_eff_cells[i - 1, min(j, Nz - 1)])
                area_s = mesh.As_u_perlen[i] * dz_u_cv
                dr_s   = mesh.r_center[i] - mesh.r_center[i - 1]
                a_S    = mu_s * area_s / dr_s
            # else: symmetry axis (pipe) -> zero flux; annulus wall handled below

            a_P = a_W + a_E + a_N + a_S
            sumnb_u[i, j] = a_P   # raw neighbor-coefficient sum, for SIMPLEC's d_u
            b_p = 0.0

            # North wall row (i = Nr-1): log-law wall function (turbulent) or
            # literal diffusive flux to u = 0 (laminar / resolved mesh).
            if i == Nr - 1:
                area_wall = mesh.An_u_perlen[i] * dz_u_cv
                dr_wall   = mesh.R - mesh.r_center[i]
                if use_wall_function:
                    from .wall_function import wall_coefficient
                    a_P_wall = wall_coefficient(u[i, j], dr_wall, area_wall,
                                                rho, mu_molecular)
                else:
                    mu_wall  = mu_eff_cells[i, min(j, Nz - 1)]
                    a_P_wall = mu_wall * area_wall / dr_wall
                a_P += a_P_wall
                # target u = 0 -> no contribution to b_p

            # South wall row (i = 0, annulus only): same treatment at r = r_min.
            if i == 0 and mesh.south_is_wall:
                area_wall = mesh.As_u_perlen[i] * dz_u_cv
                dr_wall   = mesh.r_center[i] - mesh.r_faces[0]
                if use_wall_function:
                    from .wall_function import wall_coefficient
                    a_P_wall = wall_coefficient(u[i, j], dr_wall, area_wall,
                                                rho, mu_molecular)
                else:
                    mu_wall  = mu_eff_cells[i, min(j, Nz - 1)]
                    a_P_wall = mu_wall * area_wall / dr_wall
                a_P += a_P_wall

            # Pressure-gradient source: (p_west - p_east) * A
            b_p += (p[i, j - 1] - p[i, j]) * A_ax

            # IMPLICIT under-relaxation (Patankar)
            a_P_relaxed = a_P / alpha_u
            b_p += (1.0 - alpha_u) * a_P_relaxed * u[i, j]

            A[row, row] = a_P_relaxed
            if j - 1 >= 1:
                A[row, idx(i, jj - 1)] = -a_W
            else:
                b_p += a_W * U_in  # inlet Dirichlet folded into RHS
            if j + 1 <= Nz - 1:
                A[row, idx(i, jj + 1)] = -a_E
            if i < Nr - 1:
                A[row, idx(i + 1, jj)] = -a_N
            if i > 0:
                A[row, idx(i - 1, jj)] = -a_S

            B[row] = b_p
            aP_u[i, j] = a_P_relaxed

    u_flat = spsolve(A.tocsr(), B)
    u_new = u.copy()
    for i in range(Nr):
        for j in range(1, Nz):
            u_new[i, j] = u_flat[idx(i, j - 1)]

    u_new[:, 0] = U_in
    u_new[:, Nz] = u_new[:, Nz - 1]  # outlet zero-gradient

    return u_new, aP_u, sumnb_u


def solve_v_momentum(mesh, u, v, p, rho, mu_eff_cells, alpha_v=0.7):
    """
    Solve the radial-momentum predictor v* for interior nodes i = 1..Nr-1.
    v[0, :] = 0 (symmetry axis for a pipe, no-slip south wall for an annulus),
    v[Nr, :] = 0 (north wall no-slip). Both are physically v = 0.
    Returns (v_star, aP_v, sumnb_v).
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    dr_p = mesh.r_faces[1:] - mesh.r_faces[:-1]
    dz_p = mesh.z_faces[1:] - mesh.z_faces[:-1]

    N_unknown = (Nr - 1) * Nz

    def idx(ii, j):  # ii is 0-based index into interior rows (i = ii + 1)
        return ii * Nz + j

    A = lil_matrix((N_unknown, N_unknown))
    B = np.zeros(N_unknown)
    aP_v = np.zeros((Nr + 1, Nz))
    sumnb_v = np.zeros((Nr + 1, Nz))

    for i in range(1, Nr):
        ii = i - 1
        r_here = mesh.r_faces[i]
        for j in range(Nz):
            row = idx(ii, j)

            # --- radial (south/north) neighbors ---
            area_r = 2.0 * np.pi * r_here * dz_p[j]
            dr_s = dr_p[i - 1]
            dr_n = dr_p[i] if i < Nr else dr_p[i - 1]

            mu_s = mu_eff_cells[i - 1, j]
            mu_n = mu_eff_cells[min(i, Nr - 1), j]

            D_s = mu_s * area_r / dr_s
            v_face_s = 0.5 * (v[i - 1, j] + v[i, j])
            F_s = rho * v_face_s * area_r
            a_S = D_s + max(F_s, 0.0)

            a_N = 0.0
            if i < Nr - 1:
                D_n = mu_n * area_r / dr_n
                v_face_n = 0.5 * (v[i, j] + v[i + 1, j])
                F_n = rho * v_face_n * area_r
                a_N = D_n + max(-F_n, 0.0)
            # else: north wall, v = 0 fixed

            # --- axial (west/east) neighbors ---
            A_ax = 0.5 * (mesh.A_e[i - 1, j] + mesh.A_e[min(i, Nr - 1), j])

            a_W = 0.0
            if j > 0:
                mu_w = 0.5 * (mu_eff_cells[i - 1, j - 1] + mu_eff_cells[min(i, Nr - 1), j - 1])
                D_w = mu_w * A_ax / dz_p[j - 1]
                u_bulk_w = 0.5 * (u[i - 1, j] + u[min(i, Nr - 1), j])
                F_w = rho * u_bulk_w * A_ax
                a_W = D_w + max(F_w, 0.0)

            a_E = 0.0
            if j < Nz - 1:
                mu_e = 0.5 * (mu_eff_cells[i - 1, j] + mu_eff_cells[min(i, Nr - 1), j])
                D_e = mu_e * A_ax / dz_p[j]
                u_bulk_e = 0.5 * (u[i - 1, j + 1] + u[min(i, Nr - 1), j + 1])
                F_e = rho * u_bulk_e * A_ax
                a_E = D_e + max(-F_e, 0.0)

            a_P = a_S + a_N + a_W + a_E
            sumnb_v[i, j] = a_P
            b_p = 0.0

            # North wall no-slip for the v-CV touching it
            if i == Nr - 1:
                mu_wall = mu_eff_cells[i, j]
                dr_wall = mesh.r_faces[Nr] - mesh.r_center[i]
                D_wall = mu_wall * area_r / dr_wall
                a_P += D_wall  # target v = 0, no RHS contribution

            # South wall no-slip (annulus) for the v-CV touching it
            if i == 1 and mesh.south_is_wall:
                mu_wall = mu_eff_cells[i - 1, j]
                dr_wall = mesh.r_center[i - 1] - mesh.r_faces[0]
                D_wall = mu_wall * area_r / dr_wall
                a_P += D_wall

            # Curvature source: -2*mu_eff*v/r^2 * Volume (implicit)
            vol_v = area_r * 0.5 * ((dz_p[j - 1] if j > 0 else dz_p[j]) +
                                    (dz_p[j] if j < Nz - 1 else dz_p[j]))
            mu_here = 0.5 * (mu_eff_cells[i - 1, j] + mu_eff_cells[min(i, Nr - 1), j])
            a_P += 2.0 * mu_here * vol_v / max(r_here ** 2, 1e-12)

            # Pressure-gradient source: (p_south - p_north) * area_r
            p_s = p[i - 1, j]
            p_n = p[min(i, Nr - 1), j]
            b_p += (p_s - p_n) * area_r

            a_P_relaxed = a_P / alpha_v
            b_p += (1.0 - alpha_v) * a_P_relaxed * v[i, j]

            A[row, row] = a_P_relaxed
            if i - 1 >= 1:
                A[row, idx(ii - 1, j)] = -a_S
            if i + 1 <= Nr - 1:
                A[row, idx(ii + 1, j)] = -a_N
            if j > 0:
                A[row, idx(ii, j - 1)] = -a_W
            if j < Nz - 1:
                A[row, idx(ii, j + 1)] = -a_E

            B[row] = b_p
            aP_v[i, j] = a_P_relaxed

    v_flat = spsolve(A.tocsr(), B)
    v_new = v.copy()
    for i in range(1, Nr):
        for j in range(Nz):
            v_new[i, j] = v_flat[idx(i - 1, j)]

    v_new[0, :] = 0.0
    v_new[Nr, :] = 0.0

    return v_new, aP_v, sumnb_v
