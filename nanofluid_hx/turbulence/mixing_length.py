import numpy as np

from ..properties import MaterialProperties
from ..mesh import AxisymmetricMesh
from .base import TurbulenceModel


class FluidDynamics(TurbulenceModel):
    """
    Simplified closure: 1/7-power-law velocity + mixing-length eddy diffusivity.
    Stand-in for the future k-epsilon model (see kepsilon.py).
    """
    name = "mixing_length"

    def __init__(self, mesh: AxisymmetricMesh,
                props_inner: MaterialProperties, props_outer: MaterialProperties,
                Re_inner: float, Re_outer: float):
        
        self.Pr_t = 0.85                # Turbulent Prandtl Number

        # Initialize profiles
        self.u     = np.zeros(mesh.Nr)  # Axial velocity profile (r-denpendent) 
        self.k_eff = np.zeros(mesh.Nr)  # Effective thermal conductivity profile (r-dependent)

        self.compute(mesh, props_inner, props_outer, Re_inner, Re_outer)


    def compute(self, mesh, props_inner, props_outer,
                Re_inner: float, Re_outer: float) -> None:

        self.mesh = mesh

        self.pi = props_inner
        self.po = props_outer

        self.Re_in  = Re_inner
        self.Re_out = Re_outer


        self.compute_velocity_profile()
        self.compute_turbulent_conductivity()

    def compute_velocity_profile(self):
        """Compute Turbulent velocity profile using the 1/7th power law"""

        # Inner Pipe flow
        Din = 2.0 * self.mesh.r1

        U_in_mean = (self.Re_in * self.pi.mu_nf) / (self.pi.rho_nf  * Din)
        U_in_max  = 1.22 * U_in_mean

        # Annulus Flow
        Dh = 2.0 * (self.mesh.r3 - self.mesh.r2)

        U_out_mean = (self.Re_out * self.po.mu_f) / (self.po.rho_f * Dh)


        for i in range(self.mesh.Nr):
            r    = self.mesh.r_center[i]
            zone = self.mesh.zone_map[i]

            if zone == 0:    # Inner fluid
                ratio = r / self.mesh.r1
                ratio = min(max(ratio, 0.0), 1.0)

                self.u[i] = U_in_max * (1 - ratio) ** (1.0 / 7.0)

            elif zone == 1:   # Stell Wall
                self.u[i] = 0.0

            elif zone == 2:   # Outer Fluid (Annulus)
                # Distance from annulus center line
                r_mid      = 0.5 * (self.mesh.r2 + self.mesh.r3)
                half_width = 0.5 * (self.mesh.r3 - self.mesh.r2)

                dist_normalized = abs(r - r_mid) / half_width
                dist_normalized = min(max(dist_normalized, 0.0), 1.0)

                # Max velocity at the centerline of the annulus
                U_out_max = 1.15 * U_out_mean
                self.u[i] = U_out_max * (1.0 - dist_normalized) ** (1.0 / 7.0)


    def compute_turbulent_conductivity(self):
        """ Computes the effective radial heat conductivity including molecular and eddy effects"""

        # Simple velocity gradiant calculation
        du_dr = np.zeros(self.mesh.Nr)
        for i in range(1, self.mesh.Nr - 1):
            dr    = self.mesh.r_center[i+1] - self.mesh.r_center[i-1]
            du_dr[i] = (self.u[i+1] - self.u[i-1]) / dr

        for i in range(self.mesh.Nr):
            zone = self.mesh.zone_map[i]
            r    = self.mesh.r_center[i]

            if zone == 0:             # Inner Fluid
                # Wall distance (distance to r1)
                y    = self.mesh.r1 - r
                l_m  = min(0.41 * y, 0.085 * self.mesh.r1)
                nu_t = l_m ** 2 * abs(du_dr[i]) 
                k_t  = (self.pi.rho_nf * self.pi.cp_nf * nu_t) / self.Pr_t

                self.k_eff[i] = self.pi.k_nf + k_t

            elif zone == 1:            # Steel Wall
                self.k_eff[i] = self.pi.k_s

            elif zone == 2:            # Outer Fluid
                # Wall distance (distance to nearest wall, re or r3)
                y = min(r - self.mesh.r2, self.mesh.r3 - r)
                half_width = 0.5 * (self.mesh.r3 - self.mesh.r2)

                l_m  = min(0.41 * y, 0.085 * half_width)
                nu_t = l_m ** 2 * abs(du_dr[i]) 
                k_t  = (self.po.rho_f * self.po.cp_f * nu_t) / self.Pr_t

                self.k_eff[i] = self.po.k_f + k_t   


# Verification test script
if __name__ == "__main__":
    m  = AxisymmetricMesh()
    pi = MaterialProperties(phi = 0.05)
    po = MaterialProperties(phi = 0)      # Pure Water
    fd = FluidDynamics(m, pi, po, Re_inner = 30000, Re_outer = 30000)

    print("Hydrodynamics profile calculated sucessfully!")
    print(f"Max inner velocity: {np.max(fd.u): .3f} m/s")
    print(f"Molecular Inner conductivity: {pi.k_nf: .4f} W/mK")
    print(f"Max effective Inner conductivity due to turbulence: {np.max(fd.k_eff[:15]): 0.4f} W/mk")
