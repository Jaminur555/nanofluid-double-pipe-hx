"""
2D axisymmetric FVM thermal solver for nanofluid flow in a double-pipe heat exchanger.
"""

from .properties import MaterialProperties
from .mesh import AxisymmetricMesh
from .solver import ThermalSolver
from .turbulence import get_model

__version__ = "1.0.0"
__all__ = ['MaterialProperties', 'AxisymmetricMesh', 'ThermalSolver', 'get_model']
