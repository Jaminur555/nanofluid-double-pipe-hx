import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .mesh import AxisymmetricMesh
from .properties import MaterialProperties
from .turbulence.mixing_length import FluidDynamics


class ThermalSolver:
    def __init__(self, mesh: AxisymmetricMesh, fd: FluidDynamics,
                parallel_flow: bool = True):
        self.mesh = mesh
        self.fd   = fd
        self.pi   = fd.pi
        self.po   = fd.po

        self.parallel_flow = parallel_flow

        # Temperatures from the paper
        self.T_hot_in  = 350.0 # K
        self.T_cold_in = 285.0 # K

        # Total Number of equation nuknows
        self.N_eq = mesh.Nr * mesh.Nz

        # Sparse Matrix A and Right Hand Side vector B
        self.A = lil_matrix((self.N_eq, self.N_eq))
        self.B = np.zeros(self.N_eq)


    def get_index(self, i, j) -> int:
        """Maps 2D cell indices (i, j) to 1D system matrix index"""
        return i * self.mesh.Nz + j   


    def assemble_system(self):
        """ 
        Assembles the global FVM matrix coefficients for every control volume.
        """

        self.A = lil_matrix((self.N_eq, self.N_eq))
        self.B = np.zeros(self.N_eq)

        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]

            #select proper thermal propertied based on cell location
            if zone == 0:
                rho, cp = self.pi.rho_nf, self.pi.cp_nf
            elif zone == 1:
                rho, cp = self.pi.rho_s, self.pi.cp_s
            else:
                rho, cp = self.po.rho_f, self.po.cp_f

            # Force negative velocity for outer fluid in counter_flow
            u_cell = self.fd.u[i]
            if zone == 2 and not self.parallel_flow:
                u_cell = -1 * u_cell

            for j in range(self.mesh.Nz):
                idx_p = self.get_index(i, j)

                # check boundary locations
                is_west_bound = (j == 0)
                is_east_bound = (j == self.mesh.Nz - 1)
                is_south_bound = (i == 0)
                is_north_bound = (i == self.mesh.Nr - 1)

                # Initializa local FVM coefficients
                a_W = a_E = a_S = a_N = 0.0
                b_p = 0

                # Radial Diffusion (North & South)
                if not is_north_bound:
                    k_n = 2.0/ (1.0 / self.fd.k_eff[i] + 1.0 / self.fd.k_eff[i+1])
                    dr  = self.mesh.r_center[i+1] - self.mesh.r_center[i]
                    D_n = (self.mesh.A_n[i, j] * k_n) / dr
                    a_N = D_n
                # Insulated Outer Wall: if is_north_bound, a_N stays 0.0 (Prefectly Adiabatic)

                if not is_south_bound:
                    k_s = 2.0/ (1.0 / self.fd.k_eff[i] + 1.0 / self.fd.k_eff[i-1])
                    dr  = self.mesh.r_center[i] - self.mesh.r_center[i-1]
                    D_s = (self.mesh.A_s[i, j] * k_s) / dr
                    a_S = D_s
                # Symmetry Axis (r = 0): if is_south_bound, a_S stays 0.0 (Perfectly Adiabatic)

                # Axial Convection & Diffusion (west & East)
                dz = self.mesh.z_faces[j+1] - self.mesh.z_faces[j]
                
                if is_west_bound:
                    if zone == 0:                          # Inner Fluid (Hot Inlet)
                        # Boundary diffusion (distance to face is half cell size: dz / 2)

                        D_w_bound = (self.mesh.A_w[i, j] * self.fd.k_eff[i]) / (0.5 * dz)
                        F_w       = rho * cp * self.fd.u[i] * self.mesh.A_w[i, j]

                        # Add to diagonal coefficient and the RHS vector

                        a_W = 0.0                          # No Neibhour Cell to the west

                        self.A[idx_p, idx_p] += D_w_bound + F_w
                        b_p += (D_w_bound + F_w) * self.T_hot_in

                    elif zone == 2 and self.parallel_flow: # Outer Fluid (Cold Inlet - Parallel)

                        D_w_bound = (self.mesh.A_w[i, j] * self.fd.k_eff[i]) / (0.5 * dz)

                        # Outer fluid velocity in parallel flow is positive (moving West to East)
                        F_w       = rho * cp * self.fd.u[i] * self.mesh.A_w[i, j]

                        a_W = 0.0                         

                        self.A[idx_p, idx_p] += D_w_bound + F_w
                        b_p += (D_w_bound + F_w) * self.T_cold_in 
                    else:
                        # Solid wall, or counter-flow annulus outlet at west.
                        # Pure OUTFLOW: the upwind face value is T_P itself, and
                        # that contribution is already carried by a_E (|F_e|),
                        # so a_P = a_W + a_E + ... below closes the balance.
                        # Adding another |F_w| here would double-count it.
                        a_W = 0.0
                else:               
                    # Inner Cell: Link to West neighbor
                    k_w = self.fd.k_eff[i]
                    D_w = (self.mesh.A_w[i, j] * k_w) / dz

                    # upwind advection
                    u_w = u_cell
                    F_w = rho * cp * u_w * self.mesh.A_w[i,j]

                    # If flow is moving from West to East (u_w > 0)
                    if u_w >= 0:
                        a_W = D_w + F_w
                    else:
                        a_W = D_w

                if is_east_bound:
                    if zone == 2 and not self.parallel_flow:  # Outer Fluid (Cold Inlet - Counter)
                        D_e_bound = (self.mesh.A_e[i,j] * self.fd.k_eff[i]) / ( 0.5 * dz)

                        # In counter flow, cold fluid flows East-to-West (negative velocity)
                        F_e = rho * cp * abs(self.fd.u[i] * self.mesh.A_e[i,j])

                        a_E = 0.0  # No Neighbour cell to the east

                        self.A[idx_p, idx_p] += D_e_bound + F_e
                        b_p += (D_e_bound + F_e) * self.T_cold_in
                    else:
                        # Solid wall, hot fluid outlet, or parallel cold fluid outlet: Insulated East boundary
                        a_E = 0.0
                else:
                    k_e = self.fd.k_eff[i]
                    D_e = (self.mesh.A_e[i, j] * k_e) / dz
                    u_e = u_cell
                    F_e = rho * cp * u_e * self.mesh.A_e[i, j]    

                    # If flow is mocing from East to West (u_e < 0)
                    if u_e < 0:
                        a_E = D_e + abs(F_e)
                    else:
                        a_E = D_e                


                self.A[idx_p, idx_p] += a_W + a_E + a_N + a_S    # Preliminary a_P

                if not is_west_bound:
                    self.A[idx_p, self.get_index(i, j-1)]   = -a_W
                if not is_east_bound:
                    self.A[idx_p, self.get_index(i, j+1)]   = - a_E
                if not is_north_bound:
                     self.A[idx_p, self.get_index(i + 1, j)] = - a_N
                if not is_south_bound:
                    self.A[idx_p, self.get_index(i - 1, j)] = - a_S

                self.B[idx_p] = b_p


    def solve(self):
        """Solves the sparse matrix linear system  AT = B"""
        # Convers A to CSR format for fast calculation
        A_csr  = self.A.tocsr()
        T_flat = spsolve(A_csr, self.B)

        # Reshape flat 1D temperature array back to 2D grid shape (Nr, Nz)
        return T_flat.reshape((self.mesh.Nr, self.mesh.Nz))
