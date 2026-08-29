"""
Prandtl mixing-length eddy viscosity, evaluated from actual solved axial-velociy field's radial 
gradient.
"""

import numpy as np

KAPPA  = 0.41        # Von-Karman constant
LM_CAP = 0.085       # l_m <= LM_CAP * R (Nikuradse-type outer-region cap)


def u_at_cell_centers(u, mesh):
    """
    Average the two flanking u-nodes (west/east faces) onto each pressure cell center.
    Shape (Nr, Nz).
    """
    return 0.5 * (u[:, : -1] + u[:, 1:])


def mixing_length_viscosity(U, mesh, rho, mu_molecular):
    """
    Returns mu_eff (mu_molecular + mu_t), shape (Nr, Nz), cell-centered
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    u_c    = u_at_cell_centers(U, mesh)

    du_dr = np.zeros(Nr, Nz)
    for i in range(Nr):
        if i == 0:
            # symetry axis: du/dr = 0 there; use one-sided estimate inward
            du_dr[i, :] = (u_c[1, :] - u_c[0, :]) / (mesh.r_center[1] - mesh.r_center[0]) \
                if Nr > 1 else 0.0
            du_dr[i, :] *= 0.0     # enforce symmetry exactly at the axis cell
        elif i == Nr - 1:
        # wall: u = 0 there, one-sided difference to the wall
            dr = mesh.R -  mesh.r_center[i - 1]
            du_dr[i, :] = (0.0 - u_c[i - 1, :]) / dr
        else:
            dr = mesh.r_center[i + 1] -  mesh.r_center[i - 1]
            du_dr[i, :] = (u_c[i + 1, :] - u_c[i - 1, :]) / dr


    mu_t = np.zeros((Nr, Nz))
    for i in range(Nr):
        y   = mesh.R - mesh.r_center[i]            # distance from the wall
        l_m = min(KAPPA * y, LM_CAP * mesh.R)

        mu_t [i, :] = rho * (l_m ** 2) * np.abs(du_dr[i, :])

    return mu_molecular + mu_t, mu_t
