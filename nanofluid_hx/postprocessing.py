"""Post-processing: bulk temperatures, duty, LMTD, U, Nusselt, effectiveness."""
import numpy as np


def mass_flow_rate(rho, u, area):
    """mdot = sum(rho * u * A) over a stream's radial cells [kg/s]."""
    return np.sum(rho * u * area)


def bulk_outlet_temperature(rho, cp, u, area, T_column):
    """Energy-weighted bulk temperature of a stream at one axial station [K]."""
    flux = rho * cp * u * area
    return np.sum(flux * T_column) / np.sum(flux)


def LMTD(dT1, dT2):
    """Log-mean temperature difference [K]."""
    if abs(dT1 - dT2) < 1e-5:
        return dT1
    return (dT1 - dT2) / np.log(dT1 / dT2)


def wall_nusselt(mesh, fd, T):
    """Paper-definition Nusselt (Bahmani et al. Eqs. 14-16).

    Local hot-side h(x) = q''(x)/(T_b(x) - T_w(x)) at the inner wall r = r1,
    averaged as hbar = (1/L) integral h dx; Nu_nf = hbar*(2 r1)/k_nf. The wall
    flux q'' is the same harmonic-mean face flux the energy equation solved,
    and T_w is one-sided extrapolated to r = r1 from the nanofluid-side cell.
    Arrangement-independent: fd fields are always parallel-oriented and the
    hot zone is rows [0:Nr_inner].
    """
    pi = fd.pi
    inner = slice(0, mesh.Nr_inner)
    i = mesh.Nr_inner                        # radial face at r = r1 (hot wall)

    wall_h = getattr(fd, "wall_h", None)
    if wall_h is not None:
        # Thermal wall function: same series conductance the energy equation
        # used at this face (must match ThermalSolver.assemble_system).
        h_film = wall_h["r1"]
        d_s = mesh.r_center[i] - mesh.r1             # steel cell centre to wall
        g = 1.0 / (1.0 / h_film + d_s / fd.k_eff[i, :])
        q = g * (T[i - 1, :] - T[i, :])              # W/m^2, outward through r = r1
        T_w = T[i - 1, :] - q / h_film               # wall temp from the film
    else:
        k1, k2 = fd.k_eff[i - 1, :], fd.k_eff[i, :]
        kf = 2.0 * k1 * k2 / (k1 + k2)               # harmonic face conductivity
        dr = mesh.r_center[i] - mesh.r_center[i - 1]
        q = kf * (T[i - 1, :] - T[i, :]) / dr        # W/m^2, outward through r = r1
        T_w = T[i - 1, :] - q * (mesh.r1 - mesh.r_center[i - 1]) / k1

    T_b = np.array([bulk_outlet_temperature(pi.rho_nf, pi.cp_nf,
                                            np.abs(fd.u_face[inner, j]),
                                            mesh.A_e[inner, j], T[inner, j])
                    for j in range(mesh.Nz)])
    h = q / (T_b - T_w)                      # W/m^2 K
    hbar = float(np.mean(h))                 # uniform dz: mean = (1/L) int h dx
    return hbar * (2.0 * mesh.r1) / pi.k_nf


def evaluate_case(mesh, fd, T, parallel_flow=True, T_hot_in=350.0, T_cold_in=285.0):
    """All performance metrics from a solved temperature field. Returns a dict."""
    pi, po = fd.pi, fd.po
    inner = slice(0, mesh.Nr_inner)
    outer = slice(mesh.Nr_inner + mesh.Nr_wall, mesh.Nr)

    # Weights from the outlet face column of the (parallel-oriented) face
    # velocities: the staggered u-nodes land exactly on the thermal z-faces.
    u_in  = np.abs(fd.u_face[inner, -1])
    u_out = np.abs(fd.u_face[outer, -1])
    A_in, A_out = mesh.A_e[inner, -1], mesh.A_e[outer, -1]

    m_nf = mass_flow_rate(pi.rho_nf, u_in, A_in)
    m_f  = mass_flow_rate(po.rho_f,  u_out, A_out)

    T_nf_out = bulk_outlet_temperature(pi.rho_nf, pi.cp_nf, u_in, A_in, T[inner, -1])
    T_col    = T[outer, 0] if not parallel_flow else T[outer, -1]
    T_f_out  = bulk_outlet_temperature(po.rho_f, po.cp_f, u_out, A_out, T_col)

    Q_hot  = m_nf * pi.cp_nf * (T_hot_in - T_nf_out)    # W, lost by nanofluid
    Q_cold = m_f  * po.cp_f * (T_f_out - T_cold_in)     # W,  gained by water

    if parallel_flow:
        dT1, dT2 = T_hot_in - T_cold_in, T_nf_out - T_f_out
    else:
        dT1, dT2 = T_hot_in - T_f_out, T_nf_out - T_cold_in

    dT_lm = LMTD(dT1, dT2)

    A_wall = 2.0 * np.pi * mesh.r1 * mesh.L
    U_val  = Q_hot / (A_wall * dT_lm)
    Nu_avg = U_val * (2.0 * mesh.r1) / pi.k_nf     # overall-coefficient Nu (legacy)
    Nu_nf  = wall_nusselt(mesh, fd, T)             # paper Eq. 14-16 definition

    C_min = min(m_nf * pi.cp_nf, m_f * po.cp_f)
    eps   = Q_hot / (C_min * (T_hot_in - T_cold_in))

    return {"m_dot_nf": m_nf, "m_dot_f": m_f,
            "T_nf_out": T_nf_out, "T_f_out": T_f_out,
            "Q_hot": Q_hot, "Q_cold": Q_cold,
            "LMTD": dT_lm, "U": U_val, "Nu_avg": Nu_avg,
            "Nu_nf": Nu_nf, "effectiveness": eps}
