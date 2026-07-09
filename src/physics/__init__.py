"""Physics layer: initial conditions, boundary conditions, PDE weak form.

These modules touch the FEniCSx stack; importing them requires dolfinx.
"""

from src.physics.initial_conditions import InitialConditionFactory
from src.physics.boundary_conditions import BoundaryConditionBuilder
from src.physics.pde import BurgersPDE

__all__ = [
    "InitialConditionFactory",
    "BoundaryConditionBuilder",
    "BurgersPDE",
]
