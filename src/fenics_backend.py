from __future__ import annotations

from typing import Optional

try:
    from mpi4py import MPI
    from petsc4py import PETSc

    import ufl
    from basix.ufl import element as ufl_element       
    from dolfinx import fem, mesh as dmesh, io, default_scalar_type
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver

    HAS_DOLFINX: bool = True
    DOLFINX_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc: 
    MPI = None                   
    PETSc = None             
    ufl = None                 
    ufl_element = None            
    fem = None                    
    dmesh = None                 
    io = None                     
    default_scalar_type = None    
    NonlinearProblem = None       
    NewtonSolver = None        

    HAS_DOLFINX = False
    DOLFINX_IMPORT_ERROR = exc


def require_dolfinx() -> None:
    """Raise a clear, actionable error if the FEniCSx stack is unavailable."""
    if not HAS_DOLFINX:
        raise ImportError(
            "dolfinx / mpi4py / petsc4py are required to run the solver but "
            "could not be imported. Run inside the FEniCSx Docker image.\n"
            f"Original import error: {DOLFINX_IMPORT_ERROR!r}"
        )


__all__ = [
    "MPI",
    "PETSc",
    "ufl",
    "ufl_element",
    "fem",
    "dmesh",
    "io",
    "default_scalar_type",
    "NonlinearProblem",
    "NewtonSolver",
    "HAS_DOLFINX",
    "DOLFINX_IMPORT_ERROR",
    "require_dolfinx",
]
