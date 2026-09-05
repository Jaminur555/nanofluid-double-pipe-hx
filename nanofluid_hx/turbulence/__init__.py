"""Turbulence closures feeding the thermal solver (the _MODELS registry).

Available models:
    "mixing_length"      : 1/7-power-law velocity + mixing-length eddy
                           diffusivity (fast approximate mode, not the
                           paper's methodology)
    "simplec_k_epsilon"  : the paper's methodology -- SIMPLEC + standard
                           k-epsilon on node-identical staggered sub-meshes
                           (nanofluid_hx.flow.coupling), zero interpolation
"""

from .mixing_length import FluidDynamics
from ..flow.coupling import SimplecFlow


_MODELS = {
    "mixing_length": FluidDynamics,
    "simplec_k_epsilon": SimplecFlow,
}


def get_model(name: str):
    """Return a turbulence model class by name."""
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown turbulence model '{name}'. Available: {', '.join(_MODELS)}"
        )
