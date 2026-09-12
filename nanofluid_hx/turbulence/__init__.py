"""Turbulence closures feeding the thermal solver (the _MODELS registry)."""

from ..flow.coupling import SimplecFlow


_MODELS = {
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
