"""
src/diagnostics/diagnostics.py
==============================

Integral / stability diagnostics, extracted as free functions that operate on a
dolfinx ``Function`` and MPI communicator. Keeping them out of the solver makes
them independently callable and testable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.config.config import BurgersConfig
from src.fenics_backend import fem, ufl, MPI


def compute_diagnostics(u, comm, t: float) -> Dict[str, float]:
    """Compute integral/stability diagnostics at the current time.

    Quantities
    ----------
    umax, umin : pointwise extrema (monotonicity / overshoot indicator)
    l2         : ||u||_{L2(Ω)}       = sqrt(∫ u^2 dx)
    energy     : (1/2) ∫ u^2 dx      (kinetic-energy-like functional)
    mass       : ∫ u dx              (conserved for periodic/no-flux)
    """
    # L2 norm and mass via UFL assembly (exact quadrature of the FE field).
    l2_form = fem.form(u * u * ufl.dx)
    mass_form = fem.form(u * ufl.dx)
    l2_sq = comm.allreduce(fem.assemble_scalar(l2_form), op=MPI.SUM)
    mass = comm.allreduce(fem.assemble_scalar(mass_form), op=MPI.SUM)

    arr = u.x.array.real
    umax = comm.allreduce(arr.max(), op=MPI.MAX)
    umin = comm.allreduce(arr.min(), op=MPI.MIN)

    return {
        "t": float(t),
        "umax": float(umax),
        "umin": float(umin),
        "l2": float(np.sqrt(max(l2_sq, 0.0))),
        "energy": float(0.5 * l2_sq),
        "mass": float(mass),
    }


def check_stability(diag: Dict[str, float], max_abs: float) -> None:
    """Abort early on NaN/Inf or blow-up beyond the configured threshold."""
    if not np.isfinite(diag["umax"]) or not np.isfinite(diag["umin"]):
        raise FloatingPointError(f"Non-finite solution at t={diag['t']:.4g}.")
    if max(abs(diag["umax"]), abs(diag["umin"])) > max_abs:
        raise FloatingPointError(
            f"Solution blew up (|u|>{max_abs:g}) at t={diag['t']:.4g}."
        )


def check_stability_cfg(diag: Dict[str, float], cfg: BurgersConfig) -> None:
    """Convenience wrapper pulling the threshold from a config object."""
    check_stability(diag, cfg.stability_max_abs)
