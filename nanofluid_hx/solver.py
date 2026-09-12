import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .mesh import AxisymmetricMesh
from .properties import MaterialProperties


class ThermalSolver:
    def __init__(self, mesh: AxisymmetricMesh, fd,
                parallel_flow: bool = True):
        self.mesh = mesh
        self.fd   = fd
        self.pi   = fd.pi
        self.po   = fd.po

        self.parallel_flow = parallel_flow

        # Temperatures from the paper
        self.T_hot_in  = 350.0 # K
        self.T_cold_in = 285.0 # K

        self.N_eq = mesh.Nr * mesh.Nz

        self.A = lil_matrix((self.N_eq, self.N_eq))
        self.B = np.zeros(self.N_eq)


    def get_index(self, i, j) -> int:
        """Maps 2D cell indices (i, j) to 1D system matrix index"""
        return i * self.mesh.Nz + j


    def assemble_system(self):
        """Assemble the global FVM matrix coefficients for every control volume."""

        self.A = lil_matrix((self.N_eq, self.N_eq))
        self.B = np.zeros(self.N_eq)

        # Orient the flow fields once: providers are arrangement-agnostic
        # (always parallel). Counterflow reverses the annulus z-columns here;
        # u negates, scalars do not.
        u_use = self.fd.u_face.copy()
        k_use = self.fd.k_eff.copy()
        if not self.parallel_flow:
            i2 = self.mesh.Nr_inner + self.mesh.Nr_wall
            u_use[i2:, :] = -self.fd.u_face[i2:, ::-1]
            k_use[i2:, :] = self.fd.k_eff[i2:, ::-1]

        # Thermal wall functions at the fluid-solid faces: the fluid-side
        # conductance of the harmonic face coupling is replaced by the
        # Jayatilleke film resistance (viscous sublayer), in series with the
        # solid half-cell. Turbulent providers only (fd.wall_h is None for
        # laminar/mixing-length). Both cells sharing a face get the SAME
        # conductance, so the flux stays conservative.
        wall_h = getattr(self.fd, "wall_h", None)
        i1 = self.mesh.Nr_inner
        i2 = i1 + self.mesh.Nr_wall
        G_r1 = G_r2 = None
        if wall_h is not None:
            # r1 face (nanofluid cell i1-1 | steel cell i1)
            d_s1 = self.mesh.r_center[i1] - self.mesh.r_faces[i1]
            G_r1 = self.mesh.A_n[i1 - 1, :] / (1.0 / wall_h["r1"] + d_s1 / k_use[i1, :])
            # r2 face (steel cell i2-1 | annulus cell i2)
            d_s2 = self.mesh.r_faces[i2] - self.mesh.r_center[i2 - 1]
            G_r2 = self.mesh.A_n[i2 - 1, :] / (d_s2 / k_use[i2 - 1, :] + 1.0 / wall_h["r2"])

        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]

            if zone == 0:
                rho, cp = self.pi.rho_nf, self.pi.cp_nf
            elif zone == 1:
                rho, cp = self.pi.rho_s, self.pi.cp_s
            else:
                rho, cp = self.po.rho_f, self.po.cp_f

            for j in range(self.mesh.Nz):
                idx_p = self.get_index(i, j)

                is_west_bound = (j == 0)
                is_east_bound = (j == self.mesh.Nz - 1)
                is_south_bound = (i == 0)
                is_north_bound = (i == self.mesh.Nr - 1)

                a_W = a_E = a_S = a_N = 0.0
                b_p = 0

                # Radial diffusion; zero flux at the outer wall and the axis
                if not is_north_bound:
                    k_n = 2.0/ (1.0 / k_use[i, j] + 1.0 / k_use[i+1, j])
                    dr  = self.mesh.r_center[i+1] - self.mesh.r_center[i]
                    a_N = (self.mesh.A_n[i, j] * k_n) / dr

                if not is_south_bound:
                    k_s = 2.0/ (1.0 / k_use[i, j] + 1.0 / k_use[i-1, j])
                    dr  = self.mesh.r_center[i] - self.mesh.r_center[i-1]
                    a_S = (self.mesh.A_s[i, j] * k_s) / dr

                dz = self.mesh.z_faces[j+1] - self.mesh.z_faces[j]

                if is_west_bound:
                    if zone == 0:                          # Inner fluid: hot inlet
                        D_w_bound = (self.mesh.A_w[i, j] * k_use[i, j]) / (0.5 * dz)
                        F_w       = rho * cp * u_use[i, 0] * self.mesh.A_w[i, j]

                        a_W = 0.0                          # no neighbor cell to the west

                        self.A[idx_p, idx_p] += D_w_bound + F_w
                        b_p += (D_w_bound + F_w) * self.T_hot_in

                    elif zone == 2 and self.parallel_flow: # Outer fluid: cold inlet (parallel)

                        D_w_bound = (self.mesh.A_w[i, j] * k_use[i, j]) / (0.5 * dz)
                        F_w       = rho * cp * u_use[i, 0] * self.mesh.A_w[i, j]

                        a_W = 0.0

                        self.A[idx_p, idx_p] += D_w_bound + F_w
                        b_p += (D_w_bound + F_w) * self.T_cold_in
                    else:
                        # Solid wall, or counter-flow annulus outlet at west:
                        # pure OUTFLOW. The upwind face value is T_P itself and
                        # that contribution is already carried by a_E (|F_e|);
                        # adding another |F_w| here would double-count it.
                        a_W = 0.0
                else:
                    k_w  = 2.0 / (1.0 / k_use[i, j-1] + 1.0 / k_use[i, j])
                    dz_c = self.mesh.z_center[j] - self.mesh.z_center[j-1]
                    D_w  = (self.mesh.A_w[i, j] * k_w) / dz_c

                    u_w = u_use[i, j]
                    F_w = rho * cp * u_w * self.mesh.A_w[i,j]

                    if u_w >= 0:
                        a_W = D_w + F_w
                    else:
                        a_W = D_w

                if is_east_bound:
                    if zone == 2 and not self.parallel_flow:  # Outer fluid: cold inlet (counter)
                        D_e_bound = (self.mesh.A_e[i,j] * k_use[i, j]) / ( 0.5 * dz)
                        F_e = rho * cp * abs(u_use[i, self.mesh.Nz] * self.mesh.A_e[i,j])

                        a_E = 0.0

                        self.A[idx_p, idx_p] += D_e_bound + F_e
                        b_p += (D_e_bound + F_e) * self.T_cold_in
                    else:
                        # Solid wall / outlets: insulated east boundary
                        a_E = 0.0
                else:
                    k_e  = 2.0 / (1.0 / k_use[i, j] + 1.0 / k_use[i, j+1])
                    dz_c = self.mesh.z_center[j+1] - self.mesh.z_center[j]
                    D_e  = (self.mesh.A_e[i, j] * k_e) / dz_c
                    u_e  = u_use[i, j+1]
                    F_e  = rho * cp * u_e * self.mesh.A_e[i, j]

                    if u_e < 0:
                        a_E = D_e + abs(F_e)
                    else:
                        a_E = D_e


                # Wall-function faces override the harmonic radial coupling
                if G_r1 is not None:
                    if i == i1 - 1:
                        a_N = G_r1[j]
                    elif i == i1:
                        a_S = G_r1[j]
                if G_r2 is not None:
                    if i == i2 - 1:
                        a_N = G_r2[j]
                    elif i == i2:
                        a_S = G_r2[j]

                self.A[idx_p, idx_p] += a_W + a_E + a_N + a_S    # preliminary a_P

                if not is_west_bound:
                    self.A[idx_p, self.get_index(i, j-1)]   = -a_W
                if not is_east_bound:
                    self.A[idx_p, self.get_index(i, j+1)]   = - a_E
                if not is_north_bound:
                     self.A[idx_p, self.get_index(i + 1, j)] = -a_N
                if not is_south_bound:
                    self.A[idx_p, self.get_index(i - 1, j)] = -a_S

                self.B[idx_p] = b_p


    def solve(self):
        """Solve the sparse linear system A T = B; returns T of shape (Nr, Nz)."""
        A_csr  = self.A.tocsr()
        T_flat = spsolve(A_csr, self.B)
        return T_flat.reshape((self.mesh.Nr, self.mesh.Nz))
