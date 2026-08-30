"""
Staggered axisymetric mesh for single round pipe(r in [0, R] , z in [0, L])

Layout (Patankar convention):
    - Pressure p, and any scalar (T, k, eps) line at CELL CENTERS: shape (Nr, Nz)
    - Axial velocity u lives on constant-z FACEs, same radial index as pressure
      cells: shape(Nr, Nz + 1). u[i, j] sits at (r_center[i], z_faces[j]).
    - Radial velocity v lives on constant-r FACES, same axial index as pressure
      cells: shape (Nr + 1, Nz). v[i, j] sits at (r_faces[i], z_center[j])

This avoids the pressure checkboarding that a collected grid suffers under SIMPLE-family
algorithms.
"""

import numpy as np


class StaggeredPipeMesh:
    def __init__(self, R: float, L: float, Nr: int, Nz: int, r_stretch: float = 1.4):
        """
        R         : Pipe radius [m],
        L         : Pipe length [m],
        Nr        : Number of pressure-cell rows
        Nz        : Number of pressure-cell collumns
        r_stretch : >1 clusters radial cells toward the wall (r = R * s ** r_strtch)
        """
        self.R, self.L, self.Nr, self.Nz = R, L, Nr, Nz

        s = np.linspace(0.0, 1, Nr + 1)

        self.r_faces = R * s ** r_stretch
        self.z_faces = np.linspace(0.0, L, Nz + 1)

        self.r_center = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        self.z_center = 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

        # Pressure-cell Geometry
        self.v   = np.zeros((Nr, Nz))   # Cell volume
        self.A_e = np.zeros((Nr, Nz))   # constant-z axial face area, annular disk
        self.A_n = np.zeros((Nr, Nz))   # constant-r radial face area at r_faces[i+1]
        self.A_s = np.zeros((Nr, Nz))   # constant-s axial face area at r_faces[i]

        for i in range(Nr):
            r_s, r_n = self.r_faces[i], self.r_faces[i + 1]
            for j in range(Nz):
                dz = self.z_faces[j+1] - self.z_faces[j]

                self.v[i, j]   = np.pi * (r_n **2 - r_s**2) * dz
                self.A_e[i, j] = np.pi * (r_n **2 - r_s**2)
                self.A_n[i, j] = 2.0 * np.pi *r_n * dz
                self.A_s[i, j] = 2.0 * np.pi *r_s * dz


        # u-CV faces areas: aixal (east/west) faces the same annular area as the underlying pressure row
        # (constant along z for a straight pipe). Radial (north/south) faces of u-Cv: same as A_n/ A_s of
        # the pressure row, but with the CV's own dz_u width.

        self.A_e_u = self.A_e[:, 0].copy()       # (Nr,) same for every j (straight pipe)

        r_n = self.r_faces[1:]
        r_s = self.r_faces[:-1]

        self.An_u_perlen = 2.0 * np.pi * r_n    # per unit dz -> multiply by dz_u
        self.As_u_perlen = 2.0 * np.pi * r_s

        # v-node  (radial velocity, at r_faces) geometry
        # v[i, j] for i = 1..Nr-1 sits between p-cells (i-1, j) and (i, j). (dr spacing and CV face areas for 
        # for v are computed inline in momentum.py, since they need per-column dz which varies with j.)
        self.dr_v = np.zeros(Nr + 1)

        self.dr_v[1:Nr] = self.r_center[1:] - self.r_center[:-1]
        self.dr_v[0]    = self.r_center[0] - self.r_faces[0]
        self.dr_v[Nr]   = self.r_faces[Nr] - self.r_center[Nr -1]