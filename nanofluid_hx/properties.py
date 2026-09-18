"""Nanofluid mixture properties with selectable particle and property models.

Defaults reproduce the original Bahmani et al. (2018) Al2O3-water fits
exactly; particle/mu_model/k_model/cp_model select alternative models for the
uncertainty study.
"""
import math

# Nanoparticle material constants (300 K literature values)
PARTICLES = {
    "al2o3": {"rho": 3970.0, "cp": 765.0, "k": 40.0},
    "cuo"  : {"rho": 6350.0, "cp": 535.6, "k": 20.0},
    "cu"   : {"rho": 8933.0, "cp": 385.0, "k": 400.0},
}

MU_MODELS = ("bahmani", "brinkman", "batchelor", "corcione")
K_MODELS  = ("bahmani", "maxwell", "corcione")
CP_MODELS = ("volume", "mixture")


class MaterialProperties:
    # Base Fluid: Water (300 K)
    rho_f = 997.1          # kg/m^3
    cp_f  = 4179.0         # J/(kg*K)
    k_f   = 0.613          # W/(m*K)
    mu_f  = 8.91e-4        # N*s/m^2 (Pa*s)

    # Tube Wall: Steel
    rho_s = 8030.0         # kg/m^3
    cp_s  = 502.48         # J/(kg*K)
    k_s   = 16.27          # W/(m*K)

    # Reference constants (Corcione / Xuan-Li models)
    T_REF = 300.0          # K, property reference temperature
    T_FR  = 273.15         # K, water freezing point (Corcione k)
    D_P   = 47e-9          # m, nanoparticle diameter
    M_F   = 0.01802        # kg/mol, water molar mass (Corcione d_f)
    K_B   = 1.380649e-23   # J/K, Boltzmann constant
    N_A   = 6.02214076e23  # 1/mol, Avogadro number

    def __init__(self, phi: float, particle: str = "al2o3",
                 mu_model: str = "bahmani", k_model: str = "bahmani",
                 cp_model: str = "volume"):
        """
        Phi: volume fraction (0.0 - 0.10); selectors choose the models.
        """
        if not (0.0 <= phi <= 0.10):
            raise ValueError("Volume fraction must be between 0.0 and 0.10")
        if particle not in PARTICLES:
            raise ValueError(f"Unknown particle '{particle}', "
                             f"use {sorted(PARTICLES)}")
        if mu_model not in MU_MODELS:
            raise ValueError(f"Unknown mu_model '{mu_model}', use {MU_MODELS}")
        if k_model not in K_MODELS:
            raise ValueError(f"Unknown k_model '{k_model}', use {K_MODELS}")
        if cp_model not in CP_MODELS:
            raise ValueError(f"Unknown cp_model '{cp_model}', use {CP_MODELS}")
        if particle != "al2o3" and "bahmani" in (mu_model, k_model):
            raise ValueError("Bahmani fits are Al2O3-water specific")

        self.phi = phi

        self.particle = particle
        self.mu_model = mu_model
        self.k_model  = k_model
        self.cp_model = cp_model

        self.rho_p = PARTICLES[particle]["rho"]
        self.cp_p  = PARTICLES[particle]["cp"]
        self.k_p   = PARTICLES[particle]["k"]

        self.rho_nf = self.calc_density()
        self.cp_nf  = self.calc_specific_heat()
        self.mu_nf  = self.calc_viscosity()
        self.k_nf   = self.calc_conductivity()


    def calc_density(self) -> float:
         return (1.0 - self.phi) * self.rho_f + self.phi * self.rho_p


    def calc_specific_heat(self) -> float:
        if self.cp_model == "mixture":
            return ((1.0 - self.phi) * self.rho_f * self.cp_f
                    + self.phi * self.rho_p * self.cp_p) / self.rho_nf
        return (1.0 - self.phi) * self.cp_f + self.phi * self.cp_p


    def calc_viscosity(self) -> float:
        # mu_nf/mu_f: Brinkman (1-phi)^-2.5; Batchelor 1+2.5phi+6.5phi^2;
        # Corcione 1/(1-34.87 (d_p/d_f)^-0.3 phi^1.03)
        if self.mu_model == "brinkman":
            mu_r = (1.0 - self.phi) ** -2.5
        elif self.mu_model == "batchelor":
            mu_r = 1.0 + 2.5 * self.phi + 6.5 * self.phi ** 2
        elif self.mu_model == "corcione":
            mu_r = 1.0 / (1.0 - 34.87 * (self.D_P / self._d_f) ** -0.3
                          * self.phi ** 1.03)
        else:
            mu_r = 123.0 * self.phi ** 2 + 7.3 * self.phi + 1.0
        return mu_r * self.mu_f


    def calc_conductivity(self) -> float:
        # k_nf/k_f: Maxwell two-phase; Corcione 1+4.4 Re_p^0.4 Pr^0.66
        # (T/T_fr)^10 (k_p/k_f)^0.03 phi^0.66 (Brownian Re_p)
        if self.k_model == "maxwell":
            k_r = ((self.k_p + 2.0 * self.k_f
                    - 2.0 * self.phi * (self.k_f - self.k_p))
                   / (self.k_p + 2.0 * self.k_f
                      + self.phi * (self.k_f - self.k_p)))
        elif self.k_model == "corcione":
            k_r = 1.0 + 4.4 * self._re_p ** 0.4 * self._pr_f ** 0.66 \
                * (self.T_REF / self.T_FR) ** 10 \
                * (self.k_p / self.k_f) ** 0.03 * self.phi ** 0.66
        else:
            k_r = 4.97 * self.phi ** 2 + 2.72 * self.phi + 1.0
        return k_r * self.k_f


    @property
    def _d_f(self) -> float:
        # Equivalent water-molecule diameter, Corcione (2011)
        return 0.1 * (6.0 * self.M_F / (self.N_A * math.pi * self.rho_f)) ** (1 / 3)


    @property
    def _re_p(self) -> float:
        # Brownian-motion nanoparticle Reynolds number, Corcione (2011)
        return 2.0 * self.rho_f * self.K_B * self.T_REF \
            / (math.pi * self.mu_f ** 2 * self.D_P)


    @property
    def _pr_f(self) -> float:
        return self.mu_f * self.cp_f / self.k_f


if __name__ == "__main__":
    phi = 0.05
    for kw in ({}, {"mu_model": "brinkman", "k_model": "maxwell"},
               {"mu_model": "corcione", "k_model": "corcione"}):
        p = MaterialProperties(phi, **kw)
        print(f"phi={phi} {kw or 'bahmani'}: rho={p.rho_nf:7.1f} "
              f"cp={p.cp_nf:7.1f} mu={p.mu_nf:.6f} k={p.k_nf:.4f}")
