from abc import ABC, abstractmethod


class TurbulenceModel(ABC):

    name: str = "base"


    @abstractmethod
    def compute(self, mesh, props_inner, props_outer, 
                Re_inner:float, Re_outer:float) -> None:
        """Fill self.u and self.k_eff for the given operating point"""
        pass

        