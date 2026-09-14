"""Entropy generation analysis (EGM).

Equations (all T absolute, kelvin):
    (EGM-1) s''' = k_eff/T^2 [(dT/dr)^2 + (dT/dz)^2] + mu_eff/T (du/dr)^2
            Bejan, "Entropy Generation Through Heat and Fluid Flow", Wiley 1982.
            Dominant-term friction: full Phi (White, "Viscous Fluid Flow")
            reduces to (du/dr)^2 in developed internal flow; no friction in
            the solid wall.
    (EGM-3) Be = s_ht/(s_ht + s_ff)          Bejan, "EGM", CRC 1996.
    (EGM-4) S = Integral s''' dV, dV = mesh.V.

Gradients: np.gradient on cell centers (u_face averaged onto cell centers
first); same effective fields the solver used (k_eff, mu_mol + mu_t).
"""

import numpy as np

ZONE_PIPE, ZONE_WALL, ZONE_ANNULUS = 0, 1, 2


def local_entropy_generation(mesh, T, u_face, mu_eff, k_eff):
    """Per-cell (s_ht, s_ff) [W/(K m^3)], Eq. (EGM-1)."""
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("T must be absolute temperature in kelvin")

    dT_dr, dT_dz = np.gradient(T, mesh.r_center, mesh.z_center)
    T_safe = np.maximum(T, 1e-10)
    s_ht = k_eff * (dT_dr**2 + dT_dz**2) / T_safe**2

    u_cell = 0.5 * (u_face[:, :-1] + u_face[:, 1:])          # (Nr, Nz)
    du_dr = np.gradient(u_cell, mesh.r_center, mesh.z_center)[0]
    s_ff = mu_eff * du_dr**2 / T_safe
    s_ff = np.where((mesh.zone_map == ZONE_WALL)[:, None], 0.0, s_ff)
    return s_ht, s_ff


def from_case(mesh, fd, T):
    """Assemble mu_eff (zone mu + mu_t) from a SimplecFlow provider."""
    mu_eff = np.zeros((mesh.Nr, mesh.Nz))
    mu_eff[mesh.zone_map == ZONE_PIPE] = fd.pi.mu_nf
    mu_eff[mesh.zone_map == ZONE_ANNULUS] = fd.po.mu_f
    mu_eff += fd.mu_t
    return local_entropy_generation(mesh, T, fd.u_face, mu_eff, fd.k_eff)


def integrate_entropy_generation(mesh, s_ht, s_ff):
    """Zone-wise volume integrals (EGM-4): {'pipe','wall','annulus','total'} ->
    {'ht', 'ff', 'S'} [W/K] plus integrated 'Be' (EGM-3)."""
    out = {}
    for name, zone in (("pipe", ZONE_PIPE), ("wall", ZONE_WALL),
                       ("annulus", ZONE_ANNULUS)):
        mask = mesh.zone_map == zone
        ht = float(np.sum((s_ht * mesh.V)[mask]))
        ff = float(np.sum((s_ff * mesh.V)[mask]))
        S = ht + ff
        out[name] = {"ht": ht, "ff": ff, "S": S,
                     "Be": ht / S if S > 0.0 else np.nan}
    total_ht = sum(out[k]["ht"] for k in ("pipe", "wall", "annulus"))
    total_ff = sum(out[k]["ff"] for k in ("pipe", "wall", "annulus"))
    out["total"] = {"ht": total_ht, "ff": total_ff,
                    "S": total_ht + total_ff,
                    "Be": total_ht / (total_ht + total_ff)}
    return out
