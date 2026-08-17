"""Not implemented yet"""

from .base import TurbulenceModel


class KEpsilonModel(TurbulenceModel):
    name = "kepsilon"

    def compute(self, mesh, props_inner, props_outer,
                Re_inner: float, Re_outer: float) -> None:
        raise NotImplementedError(
            'k-epsilon model is not implemented yet; use get_model("mixing_length").'
        )