"""Not implemented yet"""
import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .base import TurbulenceModel


class KEpsilonModel(TurbulenceModel):
    name = "kepsilon"

    def __init__(self, mesh, props_inner, props_outer,
                 Re_inner: float, Re_outer: float):
        self.mesh = mesh

        self.pi   = props_inner
        self.po   = props_outer

        self.Re_in  = Re_inner
        self.Re_out = Re_outer

        self.Pr_t = 0.85       # Turbulent Prandtl Number

        # Standared k-epsilon constants
        self.C_mu   = 0.09
        self.C1_eps = 1.44
        self.C2_eps = 1.92

        self.sigma_k   = 1.0
        self.sigma_eps = 1.3

        self.N_eq = mesh.Nr * mesh.Nz

        # Initializaing velociy, turbulent variables, and effective conductivity
        self.u     = np.zeros(mesh.Nr)
        self.k     = np.ones((mesh.Nr, mesh.Nz)) * 1e-5
        self.eps   = np.ones((mesh.Nr, mesh. Nz)) * 1e-5
        self.mu_t  = np.ones((mesh.Nr, mesh. Nz))
        self.k_eff = np.ones((mesh.Nr, mesh. Nz))

        self.compute(mesh, props_inner, props_outer, Re_inner, Re_outer)


    def compute(self, mesh, props_inner, props_outer,
                Re_inner: float, Re_outer: float) -> None:
        
        """
        Solves velocity, k, epsilon, and updates effective thermal conductivity
        """
        # compute velocity profile (using 1/7th power law as baseline flow)
        self.compute_velocity_profile()

        # Iteratively solve k and epsilon equatin untill convergence
        max_iter  = 50
        tolerance = 1e-4

        for interation in range(max_iter):
            mu_t_old = self.mu_t.copy()

            G_k = self.compute_shear_production()  # Calculate shear production 

            self.solve_k_equation(G_k)
            self.solve_eps_equation(G_k)

            self.update_turbulent_viscosity()

            # Check Convergence
            change = np.max(np.abs(self.mu_t - mu_t_old)) / (np.max(np.abs(self.mu_t))) + 1e-10
            
            if change < tolerance: break

            self.compute_effective_conductivity()

    
    def compute_velocity_profile(self):
        """
        Compute fully develop turbulent velocity profiles (1/7th power law)
        """
        Din = 2.0 * self.mesh.r1

        U_in_mean = (self.Re_in * self.pi.mu_nf) / (self.pi.rho_nf * Din)
        U_in_max  = 1.22 * U_in_mean

        Dh = 2.0 * (self.mesh.r3 - self.mesh.r2)

        U_out_mean = (self.Re_out * self.pi.mu_f) / (self.pi.rho_f * Dh)
        U_out_max  = 1.15 * U_out_mean

        for i in range(self.mesh.Nr):
            r    = self.mesh.r_center[i]
            zone = self.mesh.zone_map[i]

            if zone == 0:        # inner fluid
                ratio = r / self.mesh.r1
                ratio = min(max(ratio, 0.0), 1.0)

                self.u[i] = U_in_max * (1.0 - ratio) ** ( 1.0 / 7.0)
            elif zone == 1:      # Steel Wall
                self.u[i] = 0.0
            elif zone == 2:
                r_mid      = 0.5 * (self.mesh.r3 + self.mesh.r2)
                half_width = 0.5 * (self.mesh.r3 - self.mesh.r2)

                dist_normalized = abs(r - r_mid) / half_width
                dist_normalized = min(max(dist_normalized, 0.0), 1.0)

                self.u[i] = U_out_max * ( 1.0 - dist_normalized) ** (1.0 / 7.0)


    def compute_shear_production(self)-> np.ndarray:
        """
        Calculate G_k (Turbulent Shear Production) form radial velocity gradiants
        """
        G_k    = np.zeros((self.mesh.Nr, self.mesh.Nz))
        duz_dr = np.zeros(self.mesh.Nr)

        # Radial central differences
        for i in range(1, self.mesh.Nr - 1):
            dr        = self.mesh.r_center[i+1] - self.mesh.r_center[i-1]
            duz_dr[i] = (self.u[i+1] - self.u[i-1]) / dr

        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]
            if zone != 1:               # Fluid Zone Only
                strain    = duz_dr[i] ** 2
                G_k[i, :] = self.mu_t[i, :] * strain
        return G_k


    def get_inlet_values(self, Re, zone):
        """
        Calculate turbulent inlet boundary conditions
        """
        I = 0.05     # 5% Turbulent intensity

        if zone == 0:
            Din    = 2.0 * self.mesh.r1 
            U_mean = (Re * self.pi.mu_nf) / (self.po.rho_nf * Din)

            Dh = Din

        else:
            Dh     = 2.0 * (self.mesh.r3 - self.mesh.r2)
            U_mean = (Re * self.po.mu_f) / (self.po.rho_f * Dh)

        k_in   = 1.5 * (U_mean * I) ** 2
        eps_in = (self.C_mu ** 0.75) * (k_in ** 1.5) / (0.07 * Dh)

        return k_in, eps_in


    def solve_k_equation(self, G_k):
        """
        Assemble and solves the FVM matrix for Turbulent Kinetic Energy (k)
        """
        A = lil_matrix((self.N_eq, self.N_eq))
        B = lil_matrix(self.N_eq)


        k_in_inner, _ = self.get_inlet_values(self.Re_in, zone = 0)
        k_in_outer, _ = self.get_inlet_values(self.Re_out, zone = 2)


        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]

            if zone == 0:
                rho, mu = self.pi.rho_nf, self.pi.mu_nf
            elif zone == 1:              # solid wall: k = 1e-8
                for j in range(self.mesh.Nz):
                    idx = i * self.mesh.Nz + j

                    A[idx, idx] = 1.0
                    B[idx]      = 1e-8
                continue
            else:
                rho, mu = self.po.rho_f, self.po.mu_f

            for j in range(self.mesh.Nz):
                idx_p = i * self.mesh.Nz + j

                is_west = (j == 0)
                is_east = (j == self.mesh.Nz -1)

                gamma = mu + self.mu_t[i, j] / self.sigma_k
                dz    = self.mesh.z_faces[j+1] - self.mesh.z_faces[j]

                a_W = a_E = a_N = a_S = 0.0
                b_p = 0.0

                # Convection-Diffusion links
                if not is_west:
                    D_w = (self.mesh.A_w[i,j] * gamma) / dz
                    F_w = rho * self.u[i] * self.mesh.A_w[i,j]
                    a_W = D_w + max(F_w, 0.0)
                if not is_east:
                    D_e = (self.mesh.A_e[i,j] * gamma) / dz
                    F_e = rho * self.u[i] * self.mesh.A_e[i,j]
                    a_E = D_e + max(F_e, 0.0)

                if i < self.mesh.Nr - 1:
                    gamma_n = 0.5 * (gamma + (mu + self.mu_t[i+1, j] / self.sigma_k))

                    dr  = self.mesh.r_center[i] - self.mesh.r_center[i-1]
                    a_N = (self.mesh.A_s[i, j] * gamma_n) / dr
                if i > 0.0:
                    gamma_s = 0.5 * (gamma + (mu + self.mu_t[i-1, j] / self.sigma_k))

                    dr  = self.mesh.r_center[i] - self.mesh.r_center[i-1]
                    a_S = (self.mesh.A_n[i, j] * gamma_s) / dr

                 # Sorce Term Linearization
                vol = self.mesh.V[i, j]
                S_C = G_k[i, j] * vol
                S_P = -(rho * self.eps[i, j] / max(self.k[i, j], 1e-8)) * vol

                # Inlets Boundary Conditions
                if is_west:
                    if zone == 0:
                        D_bound = (self.mesh.A_w[i,j] * gamma) / (0.5 * dz)
                        F_w     = rho * self.u[i] * self.mesh.A_w[i,j]

                        A[idx_p, idx_p] += D_bound + F_w
                            
                        b_p += (D_bound + F_w) * k_in_inner
                    elif zone == 2:                       # Parallel Inlet
                        D_bound = (self.mesh.A_w[i, j] * gamma) / (0.5 * dz)
                        F_w     = rho * self.u[i] * self.mesh.A_w[i,j]

                        A[idx_p, idx_p] += D_bound + F_w

                        b_p += (D_bound + F_w) * k_in_outer

                if is_east and zone == 2:               # Counter-flow cold inlet(j = Nz - 1)
                        D_bound = (self.mesh.A_e[i, j] * gamma) / (0.5 * dz)
                        F_e     = rho * self.u[i] * self.mesh.A_e[i,j]

                        A[idx_p, idx_p] += D_bound + F_e

                        b_p += (D_bound + F_e) * k_in_outer

                A[idx_p, idx_p] += a_W + a_E + a_N + a_S - S_P
                B[idx_p] = b_p + S_C

                if not is_west and not (zone == 0 and is_west) and not (zone == 2 and is_west):
                    A[idx_p, idx_p - 1] = -a_W
                if not is_east and not (zone == 2 and is_east):
                    A[idx_p, idx_p + 1] = -a_E
                if i < self.mesh.Nr - 1:
                    A[idx_p, idx_p + self.mesh.Nz] = -a_N
                if i > 0:
                    A[idx_p, idx_p - self.mesh.Nz] = -a_S

        self.k = spsolve(A.tocsr(), B).reshape((self.mesh.Nr, self.mesh.Nz))
        self.k = np.clip(self.k, 1e-8, None)


    def solve_eps_equation(self, G_k):
        """
        Assemble and solves the FVM matrix for Dissipation Rate (epsilon)
        """
        A = lil_matrix((self.N_eq, self.N_eq))
        B = np.zeros(self.N_eq)

        _, eps_in_inner = self.get_inlet_values(self.Re_in, zone = 0)
        _, eps_in_outer = self.get_inlet_values(self.Re_out, zone = 2)

        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]

            if zone == 0:
                rho, mu = self.pi.rho_nf, self.pi.mu_nf
            elif zone == 1:
                for j in range(self.mesh.Nz):
                    idx = i * self.mesh.Nz + j

                    A[idx, idx] = 1.0
                    B[idx] = 1e-8
                continue
            else:
                rho, mu = self.po.rho_f, self.po.mu_f

            for j in range(self.mesh.Nz):
                idx_p = i * self.mesh.Nz + j

                is_west = (j == 0)
                is_east = (j == self.mesh.Nz -1)

                gamma = mu + self.mu_t[i, j] / self.sigma_eps
                dz    = self.mesh.z_faces[j+1] - self.mesh.z_faces[j]

                a_W = a_E = a_S = a_N = 0.0
                b_p = 0.0

                if not is_west:
                    D_w = (self.mesh.A_w[i, j] * gamma) / dz
                    F_w = rho * self.u[i] * self.mesh.A_w[i, j]
                    a_W = D_w + max(F_w, 0.0)
                if not is_east:
                    D_e = (self.mesh.A_e[i, j] * gamma) / dz
                    F_e = rho * self.u[i] * self.mesh.A_e[i, j]
                    a_E = D_e + max(-F_e, 0.0)

                if i < self.mesh.Nr - 1:
                    gamma_n = 0.5 * (gamma + (mu + self.mu_t[i+1, j] / self.sigma_eps))

                    dr  = self.mesh.r_center[i+1] - self.mesh.r_center[i]
                    a_N = (self.mesh.A_n[i, j] * gamma_n) / dr
                if i > 0:
                    gamma_s = 0.5 * (gamma + (mu + self.mu_t[i-1, j] / self.sigma_eps))

                    dr  = self.mesh.r_center[i] - self.mesh.r_center[i-1]
                    a_S = (self.mesh.A_s[i, j] * gamma_s) / dr

                # Source Term Linearization
                vol = self.mesh.V[i, j]
                S_C = self.C1_eps * (self.eps[i, j] / max(self.k[i, j], 1e-8)) * G_k[i, j] * vol
                S_P = - self.C2_eps * (rho * self.eps[i, j] / max(self.k[i, j], 1e-8)) * vol

                # Inlets Boundary Conditions
                if is_west:
                    if zone == 0:
                        D_bound = (self.mesh.A_w[i, j] * gamma) / (0.5 * dz)

                        F_w = rho * self.u[i] * self.mesh.A_w[i, j]

                        A[idx_p, idx_p] += D_bound + F_w
                        b_p += (D_bound + F_w) * eps_in_inner
                    elif zone == 2:
                        D_bound = (self.mesh.A_w[i, j] * gamma) / (0.5 * dz)

                        F_w = rho * self.u[i] * self.mesh.A_w[i, j]

                        A[idx_p, idx_p] += D_bound + F_w
                        b_p += (D_bound + F_w) * eps_in_outer

                if is_east and zone == 2:
                    D_bound = (self.mesh.A_e[i, j] * gamma) / (0.5 * dz)

                    F_e = rho * abs(self.u[i]) * self.mesh.A_e[i, j]

                    A[idx_p, idx_p] += D_bound + F_e
                    b_p += (D_bound + F_e) * eps_in_outer

                A[idx_p, idx_p] += a_W + a_E + a_N + a_S - S_P
                B[idx_p] = b_p + S_C

                if not is_west and not (zone == 0 and is_west) and not (zone == 2 and is_west):
                    A[idx_p, idx_p - 1] = -a_W
                if not is_east and not (zone == 2 and is_east):
                    A[idx_p, idx_p + 1] = -a_E
                if i < self.mesh.Nr - 1:
                    A[idx_p, idx_p + self.mesh.Nz] = -a_N
                if i > 0:
                    A[idx_p, idx_p - self.mesh.Nz] = -a_S

        self.eps = spsolve(A.tocsr(), B).reshape((self.mesh.Nr, self.mesh.Nz))
        self.eps = np.clip(self.eps, 1e-8, None)               


    def update_turbulent_viscosity(self):
        """
        Updates mu_t with molecular and resolved k_epsilon eddy contributions
        """
        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]
            if zone == 1:
                self.mu_t[i, :] = 0.0
                continue
            rho = self.pi.rho_nf if zone == 0 else self.po.rho_f
            for j in range(self.mesh.Nz):
                self.mu_t[i, j] = rho * self.C_mu * (self.k[i, j] ** 2) / max(self.eps[i, j], 1e-8)


    def compute_effective_conductivity(self):
        """
        Updates k_eff with molecular and resolved k_epsilon eddy contributions
        """
        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]

            if zone == 0:
                self.k_eff[i, :] = self.pi.k_nf + (self.pi.rho_nf * self.pi.cp_nf *
                                                   (self.mu_t[i, :] / self.pi.rho_nf)) / self.Pr_t
            elif zone == 1:
                self.k_eff[i, :] = self.pi.k_s
            else:
                self.k_eff[i, :] = self.po.k_f + (self.po.rho_f * self.po.cp_f *
                                                   (self.mu_t[i, :] / self.po.rho_f)) / self.Pr_t




                        