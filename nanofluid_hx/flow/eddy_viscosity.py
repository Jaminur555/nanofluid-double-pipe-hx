"""
Prandtl mixing-length eddy viscosity from the SOLVED axial-velocity field's
radial gradient (Stage 1's placeholder closure -- the real k-eps model is
k_epsilon.py). Annulus-aware: distances to both walls.
"""
import numpy as np

KAPPA  = 0.41
LM_CAP = 0.085      # l_m <= LM_CAP * (zone hydraulic radius)


def u_at_cell_centers(u, mesh):
    """Average the two flanking u-nodes onto each pressure cell center."""
    return 0.5 * (u[:, :-1] + u[:, 1:])


def mixing_length_viscosity(u, mesh, rho, mu_molecular):
    """Returns (mu_eff, mu_t), shape (Nr, Nz), cell-centered."""
    Nr, Nz = mesh.Nr, mesh.Nz
    u_c    = u_at_cell_centers(u, mesh)

    du_dr = np.zeros((Nr, Nz))
    for i in range(Nr):
        if i == 0:
            if mesh.south_is_wall:
                dr          = mesh.r_center[1] - mesh.r_faces[0]
                du_dr[i, :] = (u_c[1, :] - 0.0) / dr
            else:
                du_dr[i, :] = 0.0     # symmetry axis
        elif i == Nr - 1:
            dr          = mesh.R - mesh.r_center[i - 1]
            du_dr[i, :] = (0.0 - u_c[i - 1, :]) / dr
        else:
            dr          = mesh.r_center[i + 1] - mesh.r_center[i - 1]
            du_dr[i, :] = (u_c[i + 1, :] - u_c[i - 1, :]) / dr

    half_gap = mesh.R - mesh.r_faces[0]      # pipe: R; annulus: (R - r_min)

    mu_t = np.zeros((Nr, Nz))
    for i in range(Nr):
        y_north = mesh.R - mesh.r_center[i]
        y_south = (mesh.r_center[i] - mesh.r_faces[0]) if mesh.south_is_wall else np.inf

        y   = min(y_north, y_south)
        l_m = min(KAPPA * y, LM_CAP * half_gap)

        mu_t[i, :] = rho * (l_m ** 2) * np.abs(du_dr[i, :])

    return mu_molecular + mu_t, mu_t
