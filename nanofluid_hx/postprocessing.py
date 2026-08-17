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


def evaluate_case(mesh, fd, T, parallel_flow=True, T_hot_in=350.0, T_cold_in=285.0):
    """All performance metrics from a solved temperature field. Returns a dict."""
    pi, po = fd.pi, fd.po
    inner = slice(0, mesh.Nr_inner)
    outer = slice(mesh.Nr_inner + mesh.Nr_wall, mesh.Nr)

    A_in,  u_in  = mesh.A_e[inner, 0], fd.u[inner]
    A_out, u_out = mesh.A_e[outer, 0], fd.u[outer]

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
    Nu_avg = U_val * (2.0 * mesh.r1) / pi.k_nf

    C_min = min(m_nf * pi.cp_nf, m_f * po.cp_f)
    eps   = Q_hot / (C_min * (T_hot_in - T_cold_in))

    return {"m_dot_nf": m_nf, "m_dot_f": m_f,
            "T_nf_out": T_nf_out, "T_f_out": T_f_out,
            "Q_hot": Q_hot, "Q_cold": Q_cold,
            "LMTD": dT_lm, "U": U_val, "Nu_avg": Nu_avg, "effectiveness": eps}