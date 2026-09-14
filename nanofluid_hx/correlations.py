"""Experimental Nusselt correlations for Al2O3-water nanofluid validation.

All correlations take the SAME Reynolds number convention as the solver:
Re = rho_nf * U * D / mu_nf with D = 2*r1 (pipe hydraulic diameter), and the
nanofluid Prandtl number Pr_nf = mu_nf * cp_nf / k_nf from MaterialProperties
(at phi = 0 these collapse to pure-water values).

References:
- Pak, B.-C., Cho, Y.-I. (1998), Exp. Heat Transfer 11(2), 151-170:
  Nu = 0.021 Re^0.8 Pr^0.5 (turbulent, fitted for phi <= 3 vol%)
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
