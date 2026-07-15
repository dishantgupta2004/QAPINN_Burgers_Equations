from __future__ import annotations
from typing import Dict
import numpy as np
from src.config.config import BurgersConfig
from src.fenics_backend import fem, ufl, MPI


def compute_diagnostics(u, comm, t: float) -> Dict[str, float]:
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
    if not np.isfinite(diag["umax"]) or not np.isfinite(diag["umin"]):
        raise FloatingPointError(f"Non-finite solution at t={diag['t']:.4g}.")
    if max(abs(diag["umax"]), abs(diag["umin"])) > max_abs:
        raise FloatingPointError(
            f"Solution blew up (|u|>{max_abs:g}) at t={diag['t']:.4g}."
        )


def check_stability_cfg(diag: Dict[str, float], cfg: BurgersConfig) -> None:
    check_stability(diag, cfg.stability_max_abs)
