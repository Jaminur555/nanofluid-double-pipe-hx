"""Experimental Nusselt correlations for Al2O3-water nanofluid validation.

All correlations take the SAME Reynolds number convention as the solver:
Re = rho_nf * U * D / mu_nf with D = 2*r1 (pipe hydraulic diameter), and the
nanofluid Prandtl number Pr_nf = mu_nf * cp_nf / k_nf from MaterialProperties
(at phi = 0 these collapse to pure-water values).

References:
- Pak, B.-C., Cho, Y.-I. (1998), Exp. Heat Transfer 11(2), 151-170:
  Nu = 0.021 Re^0.8 Pr^0.5 (turbulent, fitted for phi <= 3 vol%)
- Xuan, Y., Li, Q. (2003), J. Heat Transfer 125(1), 151-155:
  Nu = 0.0059 (1 + 7.6286 phi^0.6886 Pe_d^0.001) Re^0.9238 Pr^0.4
  (experimental, fitted to Cu-water d_p = 100 nm; secondary reference)
- Dittus-Boelter (heating): Nu = 0.023 Re^0.8 Pr^0.4 (pure-water baseline)

Maiga et al. (2006) is deliberately EXCLUDED from validation: it is derived
from CFD simulations (Maiga's own), so it is model-vs-model, not experimental
validation (user decision 2026-09-14).
"""


def prandtl(props) -> float:
    """Prandtl number of the nanofluid (or base fluid at phi = 0)."""
    return props.mu_nf * props.cp_nf / props.k_nf


def dittus_boelter(Re: float, props) -> float:
    """Dittus-Boelter (fluid heated): Nu = 0.023 Re^0.8 Pr^0.4."""
    Pr = prandtl(props)
    return 0.023 * Re**0.8 * Pr**0.4


def pak_cho(Re: float, props) -> float:
    """Pak-Cho (1998) turbulent nanofluid correlation: Nu = 0.021 Re^0.8 Pr^0.5.

    Fitted to experimental data for phi <= 3 vol%; using it at phi up to 10%
    (as Bahmani et al. also did) is an extrapolation.
    """
    Pr = prandtl(props)
    return 0.021 * Re**0.8 * Pr**0.5


# Xuan-Li reference constants: pipe hydraulic diameter (solver 2*r1) and
# their Cu particle diameter
D_PIPE = 0.026
D_P_XUAN_LI = 100e-9


def xuan_li(Re: float, props) -> float:
    """Xuan-Li (2003): Nu = 0.0059(1+7.6286 phi^0.6886 Pe_d^0.001) Re^0.9238 Pr^0.4.

    Experimental Cu-water fit (d_p = 100 nm); the Pe_d^0.001 term is nearly
    inert, so deviations mainly reflect the particle-material mismatch.
    """
    Pr = prandtl(props)
    alpha = props.k_nf / (props.rho_nf * props.cp_nf)
    u_m = Re * props.mu_nf / (props.rho_nf * D_PIPE)
    pe_d = u_m * D_P_XUAN_LI / alpha
    return (0.0059 * (1.0 + 7.6286 * props.phi**0.6886 * pe_d**0.001)
            * Re**0.9238 * Pr**0.4)
