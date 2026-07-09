"""
src/burgers_equation.py
    src/config/       enums + BurgersConfig
    src/physics/      InitialConditionFactory, BoundaryConditionBuilder, BurgersPDE
    src/solver/       BurgersSolver (the time loop)
    src/diagnostics/  compute_diagnostics, check_stability
    src/io/           result writers + PINN-dataset export
    src/viz/          plotting helpers
    src/utils/        rank-0-safe logging
    src/fenics_backend.py   centralized FEniCSx/PETSc/MPI import

This module remains only so that existing code and notebooks importing

    from src.burgers_equation import BurgersConfig, BurgersSolver

keep working. New code should import from the packages above (or from
``src`` / ``src.solver.burgers_solver``) directly. Prefer running the
experiment with::

    python -m src.run_burger
"""

from __future__ import annotations

# Pure-Python layer (always importable, no dolfinx required).
from src.config.config import BurgersConfig
from src.config.enums import ElementType, TimeIntegrator, ICKind, BCType, BCSpec
from src.utils.logging_utils import build_logger, LOGGER
from src.fenics_backend import HAS_DOLFINX, DOLFINX_IMPORT_ERROR

# dolfinx-backed layer. Guarded so importing this shim never hard-fails in an
# environment without the FEniCSx stack (matching the original behaviour, where
# the module could be imported for its config even without dolfinx).
try:
    from src.physics.initial_conditions import InitialConditionFactory
    from src.physics.boundary_conditions import BoundaryConditionBuilder
    from src.physics.pde import BurgersPDE
    from src.solver.burgers_solver import BurgersSolver
except Exception:  # pragma: no cover - environment dependent
    InitialConditionFactory = None      # type: ignore[assignment]
    BoundaryConditionBuilder = None     # type: ignore[assignment]
    BurgersPDE = None                   # type: ignore[assignment]
    BurgersSolver = None                # type: ignore[assignment]

# Backward-compatible private aliases some callers may have relied on.
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


if __name__ == "__main__":  # pragma: no cover
    # Minimal self-test / demo. The intended entry point is `python -m src.run_burger`.
    demo = BurgersConfig(experiment_name="burgers_demo")
    if HAS_DOLFINX and BurgersSolver is not None:
        BurgersSolver(demo).solve()
    else:
        LOGGER.warning(
            "dolfinx not importable in this environment; printing config only."
        )
        print(demo.to_json())
