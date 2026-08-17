"""Turbulence closures.

Available models:
    "mixing_length" : 1/7-power-law velocity + mixing-length eddy diffusivity (implemented)
    "kepsilon"      : two-equation k-epsilon model (reserved, not yet implemented)
"""

from .base import TurbulenceModel
from .mixing_length import FluidDynamics
from .kepsilon import KEpsilonModel


_MODELS = {
    "mixing_length": FluidDynamics,
    "kepsilon": KEpsilonModel,
}


def get_model(name: str):
    """Return a turbulence model class by name."""
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown turbulence model '{name}'. Available: {', '.join(_MODELS)}"
        )
    