from __future__ import annotations
from src.config.config import BurgersConfig
from src.config.enums import ElementType, TimeIntegrator, ICKind, BCType, BCSpec
from src.utils.logging_utils import build_logger, LOGGER
from src.fenics_backend import HAS_DOLFINX, DOLFINX_IMPORT_ERROR
try:
    from src.physics.initial_conditions import InitialConditionFactory
    from src.physics.boundary_conditions import BoundaryConditionBuilder
    from src.physics.pde import BurgersPDE
    from src.solver.burgers_solver import BurgersSolver
except Exception:  
    InitialConditionFactory = None 
    BoundaryConditionBuilder = None     
    BurgersPDE = None                
    BurgersSolver = None                


_HAS_DOLFINX = HAS_DOLFINX
_DOLFINX_IMPORT_ERROR = DOLFINX_IMPORT_ERROR
_build_logger = build_logger

__all__ = [
    "BurgersConfig",
    "ElementType",
    "TimeIntegrator",
    "ICKind",
    "BCType",
    "BCSpec",
    "InitialConditionFactory",
    "BoundaryConditionBuilder",
    "BurgersPDE",
    "BurgersSolver",
    "build_logger",
    "LOGGER",
    "HAS_DOLFINX",
    "DOLFINX_IMPORT_ERROR",
]


if __name__ == "__main__": 
    demo = BurgersConfig(experiment_name="burgers_demo")
    if HAS_DOLFINX and BurgersSolver is not None:
        BurgersSolver(demo).solve()
    else:
        LOGGER.warning(
            "dolfinx not importable in this environment; printing config only."
        )
        print(demo.to_json())
