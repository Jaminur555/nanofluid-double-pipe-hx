"""
Staggered axisymmetric mesh for ONE fluid zone: r in [r_min, R], z in [0, L].

Layout (Patankar convention):
    - Pressure p, and any scalar (T, k, eps), live at CELL CENTERS: (Nr, Nz)
    - Axial velocity u lives on constant-z FACES, same radial index as
      pressure cells: shape (Nr, Nz + 1). u[i, j] sits at (r_center[i], z_faces[j]).
    - Radial velocity v lives on constant-r FACES, same axial index as
      pressure cells: shape (Nr + 1, Nz). v[i, j] sits at (r_faces[i], z_center[j]).

r_min = 0  -> inner pipe: the south boundary (i = 0) is the SYMMETRY AXIS
              (zero radial flux, v = 0).
r_min > 0  -> annulus: the south boundary is a solid WALL (no-slip), exactly
              like the north boundary; mesh.south_is_wall = True.

from_faces() builds a mesh that SHARES its face coordinates with the collocated
thermal mesh (AxisymmetricMesh) -- then the staggered u-nodes land exactly on
the thermal mesh's z-faces and cell-centered fields land on thermal cell
centers, so coupling flow to energy (Stage 3) needs NO interpolation.
"""
import numpy as np


class StaggeredPipeMesh:
    def __init__(self, R: float, L: float, Nr: int, Nz: int, r_stretch: float = 1.4,
                 r_min: float = 0.0, south_is_wall: bool = None):
        """
        R, L          : outer radius and length [m]
        Nr, Nz        : number of pressure-cell rows/columns
        r_stretch     : >1 clusters radial cells toward r = R (power grading)
        r_min         : inner radius (0 for the pipe, r2 = 0.015 for the annulus)
        south_is_wall : True if the r = r_min boundary is a wall
                        (default: True iff r_min > 0)
        """
        if south_is_wall is None:
            south_is_wall = (r_min > 0)
        self.r_min = r_min
        self.south_is_wall = south_is_wall

        s = np.linspace(0.0, 1.0, Nr + 1)
        r_faces = r_min + (R - r_min) * s ** r_stretch
        z_faces = np.linspace(0.0, L, Nz + 1)
        self._build(r_faces, z_faces, R, L, Nr, Nz)

    @classmethod
    def from_faces(cls, r_faces, z_faces, south_is_wall=False):
        """Build from explicit face coordinates (use the thermal mesh's own
        faces in Stage 3 so the two grids are node-identical)."""
        mesh = cls.__new__(cls)
        mesh.south_is_wall = south_is_wall
        mesh.r_min = float(r_faces[0])
        mesh._build(np.asarray(r_faces, dtype=float),
                    np.asarray(z_faces, dtype=float),
                    float(r_faces[-1]),
                    float(z_faces[-1]) - float(z_faces[0]),
                    len(r_faces) - 1, len(z_faces) - 1)
        return mesh

    def _build(self, r_faces, z_faces, R, L, Nr, Nz):
        self.R, self.L, self.Nr, self.Nz = R, L, Nr, Nz

        self.r_faces = r_faces
        self.z_faces = z_faces

        self.r_center = 0.5 * (r_faces[:-1] + r_faces[1:])
        self.z_center = 0.5 * (z_faces[:-1] + z_faces[1:])

        # Hydraulic diameter: pipe -> 2R, annulus -> 2*(R - r_min)
        self.Dh = 2.0 * (R - self.r_min) if self.r_min > 0 else 2.0 * R

        # ---- Pressure-cell geometry ----
        self.V   = np.zeros((Nr, Nz))   # cell volume
        self.A_e = np.zeros((Nr, Nz))   # constant-z (axial) face area, annular disc
        self.A_n = np.zeros((Nr, Nz))   # constant-r face area at r_faces[i+1]
        self.A_s = np.zeros((Nr, Nz))   # constant-r face area at r_faces[i]

        for i in range(Nr):
            r_s, r_n = r_faces[i], r_faces[i + 1]
            for j in range(Nz):
                dz = z_faces[j + 1] - z_faces[j]
                self.V[i, j]   = np.pi * (r_n ** 2 - r_s ** 2) * dz
                self.A_e[i, j] = np.pi * (r_n ** 2 - r_s ** 2)
                self.A_n[i, j] = 2.0 * np.pi * r_n * dz
                self.A_s[i, j] = 2.0 * np.pi * r_s * dz

        # ---- u-node geometry (axial velocity, at z_faces) ----
        self.A_e_u = self.A_e[:, 0].copy()          # (Nr,) same for every j
        r_n = r_faces[1:]
        r_s = r_faces[:-1]
        self.An_u_perlen = 2.0 * np.pi * r_n         # per unit dz -> * dz_u
        self.As_u_perlen = 2.0 * np.pi * r_s

        # ---- v-node geometry (radial velocity, at r_faces) ----
        self.dr_v = np.zeros(Nr + 1)
        self.dr_v[1:Nr] = self.r_center[1:] - self.r_center[:-1]
        self.dr_v[0]    = self.r_center[0] - r_faces[0]
        self.dr_v[Nr]   = r_faces[Nr] - self.r_center[Nr - 1]
