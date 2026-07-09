"""
src — modular finite-element Burgers solver + PINN dataset generator.

Public entry points are re-exported here for convenience::

    from src import BurgersConfig, BurgersSolver

Run the reference experiment with::

    python -m src.run_burger
"""

from src.config.config import BurgersConfig
from src.config.enums import ElementType, TimeIntegrator, ICKind, BCType, BCSpec

# NOTE: BurgersSolver (and anything else touching dolfinx) is intentionally NOT
# imported here, so that pure-Python consumers of the config layer can import
# `src` without the FEniCSx stack installed. Import it explicitly when needed:
#     from src.solver.burgers_solver import BurgersSolver

__all__ = [
    "BurgersConfig",
    "ElementType",
    "TimeIntegrator",
    "ICKind",
    "BCType",
    "BCSpec",
]
