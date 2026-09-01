"""Legacy turbulence closures for the prescribed-velocity thermal solver.

Available models:
    "mixing_length" : 1/7-power-law velocity + mixing-length eddy diffusivity
                      (fast approximate mode, not the paper's methodology)

The real solver (SIMPLEC + standard k-epsilon on a staggered grid) lives in
nanofluid_hx.flow.
"""

from .mixing_length import FluidDynamics


_MODELS = {
    "mixing_length": FluidDynamics,
}


def get_model(name: str):
    """Return a turbulence model class by name."""
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown turbulence model '{name}'. Available: {', '.join(_MODELS)}"
        )
