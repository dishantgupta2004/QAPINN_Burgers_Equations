"""
src/fenics_backend.py
=====================

Single, centralized import point for the FEniCSx / PETSc / MPI stack.

Historically each module that needed dolfinx wrapped the import in its own
``try/except``. That duplicated the guard and made it easy for the "is dolfinx
available?" answer to drift between modules. Instead, every module that touches
the finite-element backend imports the symbols it needs from *here*.

The imports are wrapped so that documentation tooling, linters, and unit tests
of the pure-Python layers (config, enums) can import the package even where
dolfinx is not installed (e.g. outside the FEniCSx Docker image). Any attempt to
actually *solve* without dolfinx raises a clear, actionable error via
:func:`require_dolfinx`.

Bug fix (B3)
------------
The original monolith did a bare ``import basix`` and then referenced
``basix.ufl.element``. ``basix.ufl`` is a submodule that is not guaranteed to be
bound by ``import basix`` on every version, which can raise ``AttributeError``.
We import the ``element`` factory explicitly here.
"""

from __future__ import annotations

from typing import Optional

try:
    from mpi4py import MPI
    from petsc4py import PETSc

    import ufl
    from basix.ufl import element as ufl_element        # B3: explicit submodule import
    from dolfinx import fem, mesh as dmesh, io, default_scalar_type
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver

    HAS_DOLFINX: bool = True
    DOLFINX_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    MPI = None                    # type: ignore[assignment]
    PETSc = None                  # type: ignore[assignment]
    ufl = None                    # type: ignore[assignment]
    ufl_element = None            # type: ignore[assignment]
    fem = None                    # type: ignore[assignment]
    dmesh = None                  # type: ignore[assignment]
    io = None                     # type: ignore[assignment]
    default_scalar_type = None    # type: ignore[assignment]
    NonlinearProblem = None       # type: ignore[assignment]
    NewtonSolver = None           # type: ignore[assignment]

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
