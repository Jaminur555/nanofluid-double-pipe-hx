class MaterialProperties:
    # Base Fluid: Water
    rho_f = 997.1        # kg/m^3
    cp_f  = 4179.0        # J/(kg*K)
    k_f   = 0.613          # W/(m*K)
    mu_f  = 8.91e-4       # N*s/m^2 (Pa*s)

    # Nanoparticles: Al2O3
    rho_p = 3970.0       # kg/m^3
    cp_p  = 765.0         # J/(kg*K)
    k_p   = 40.0           # W/(m*K)

    # Tube Wall: Steel
    rho_s = 8030.0       # kg/m^3
    cp_s  = 502.48        # J/(kg*K)
    k_s   = 16.27          # W/(m*K)


    def __init__(self, phi: float):
        """ Phi: Volume fraction of Nano-particles (0.0 - 0.10)"""
        if not(0.0 <= phi <= 0.10):
            raise ValueError("Volume fraction must be between 0.0 and 0.10")

        self.phi = phi

        # Calculate nanofluid properties
        self.rho_nf = self.calc_density()
        self.cp_nf  = self.calc_specific_heat()
        self.mu_nf     = self.calc_viscosity()
        self.k_nf   = self.calc_conductivity()


    def calc_density(self) -> float:
         return (1.0 - self.phi) * self.rho_f + self.phi * self.rho_p


    def calc_specific_heat(self) -> float:
        return (1.0 - self.phi) * self.cp_f + self.phi * self.cp_p


    def calc_viscosity (self) -> float:
        mu_r = 123.0 * (self.phi ** 2) + 7.3 * self.phi + 1.0
        return mu_r * self.mu_f


    def calc_conductivity(self) -> float:
        k_r = 4.97 * (self.phi ** 2) + 2.72 * self.phi + 1.0
        return k_r * self.k_f
        

if __name__ == "__main__":
    phi = 0.05
    test_fluid = MaterialProperties(phi)
    print(f"Nanofulid properties at phi = {phi}%:")
    print(f"Denstiy: {test_fluid.rho_nf: .2f} kg/ m^3")
    print(f"Sepecific heat: {test_fluid.cp_nf: .2f} j/kg. k ")
    print(f"Denstiy: {test_fluid.mu_nf: .6f} Pas")
    print(f"Denstiy: {test_fluid.k_nf: .2f} W/(m.k)")
