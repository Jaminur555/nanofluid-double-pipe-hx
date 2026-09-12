"""Staggered axisymmetric mesh for ONE fluid zone: r in [r_min, R], z in [0, L].

Layout (Patankar): p and scalars at cell centers (Nr, Nz); u on constant-z
faces, shape (Nr, Nz+1); v on constant-r faces, shape (Nr+1, Nz). r_min = 0
-> south boundary is the symmetry axis; r_min > 0 -> solid wall
(south_is_wall = True). from_faces() shares coordinates with the thermal
mesh so coupled fields need no interpolation.
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
        """Build from explicit face coordinates (e.g. the thermal mesh's own faces)."""
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
        r_s, r_n = r_faces[:-1], r_faces[1:]
        dz = z_faces[1:] - z_faces[:-1]
        annular = np.pi * (r_n ** 2 - r_s ** 2)            # (Nr,)
        self.V   = annular[:, None] * dz[None, :]          # (Nr, Nz)
        self.A_e = annular[:, None] * np.ones_like(dz)[None, :]
        self.A_n = (2.0 * np.pi * r_n[:, None] * dz[None, :])
        self.A_s = (2.0 * np.pi * r_s[:, None] * dz[None, :])

        # ---- u-node geometry (axial velocity, at z_faces) ----
        self.A_e_u = self.A_e[:, 0].copy()          # (Nr,) same for every j
        self.An_u_perlen = 2.0 * np.pi * r_n        # per unit dz -> * dz_u
        self.As_u_perlen = 2.0 * np.pi * r_s

        # ---- v-node geometry (radial velocity, at r_faces) ----
        self.dr_v = np.zeros(Nr + 1)
        self.dr_v[1:Nr] = self.r_center[1:] - self.r_center[:-1]
        self.dr_v[0]    = self.r_center[0] - r_faces[0]
        self.dr_v[Nr]   = r_faces[Nr] - self.r_center[Nr - 1]
