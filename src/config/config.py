"""
src/config/config.py
=====================

The declarative description of one experiment, generalized across 1D/2D/3D.

A single :class:`BurgersConfig` object fully specifies *what* to solve — the
spatial dimension, the box domain, the mesh resolution per axis, the time
stepping, the viscosity, the initial and boundary conditions, the solver knobs
and the output flags. It contains **no numerics**; the solver reads it and does
the work. Keeping the dimension here (a single ``dimension`` field) is what lets
every downstream module branch cleanly on 1D vs 2D vs 3D.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from src.config.enums import ICKind, BCType, BCSpec


# ---------------------------------------------------------------------------
# Face vocabulary. A structured box domain has 2 faces per spatial dimension.
# The names are the single source of truth used by the boundary-condition
# builder and by validation. ``FACE_AXIS`` maps each face to the coordinate
# axis (0=x, 1=y, 2=z) it is normal to and whether it sits at the min or max.
# ---------------------------------------------------------------------------
FACE_NAMES: Dict[int, Tuple[str, ...]] = {
    1: ("xmin", "xmax"),
    2: ("xmin", "xmax", "ymin", "ymax"),
    3: ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
}

# face -> (axis index, "min"|"max")
FACE_AXIS: Dict[str, Tuple[int, str]] = {
    "xmin": (0, "min"), "xmax": (0, "max"),
    "ymin": (1, "min"), "ymax": (1, "max"),
    "zmin": (2, "min"), "zmax": (2, "max"),
}


@dataclass(frozen=True)
class BurgersConfig:
    """Immutable, fully-declarative description of a single experiment."""

    # ---- 1. dimension / domain ----------------------------------------------
    # dimension selects 1D interval, 2D rectangle, or 3D box. All the other
    # per-axis fields (ymin/ymax/ny, zmin/zmax/nz) are simply ignored for the
    # axes that do not exist at the chosen dimension.
    dimension: int = 1
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    zmin: float = 0.0
    zmax: float = 1.0

    # ---- 2/3. mesh ----------------------------------------------------------
    nx: int = 300                            # number of cells along x
    ny: int = 64                             # number of cells along y (2D/3D)
    nz: int = 64                             # number of cells along z (3D)
    element_type: str = "Lagrange"
    polynomial_degree: int = 1
    # Simplex cells (interval/triangle/tetrahedron) are used by default; the
    # cell family is chosen automatically per dimension by the mesh builder.

    # ---- 4. time integration ------------------------------------------------
    T: float = 1.0
    dt: float = 0.002
    time_integrator: str = "backward_euler"
    adaptive: bool = False                   # reserved; not yet active

    # ---- 5. physics ---------------------------------------------------------
    nu: float = 0.01                         # kinematic viscosity, must be > 0

    # ---- 6. Initial condition -----------------------------------------------
    # Either a preset name (see ICKind) or "custom" together with ic_function.
    # Presets are dimension-aware (see InitialConditionFactory).
    initial_condition: str = "sin"
    ic_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # Preset shaping parameters (used only by the relevant preset):
    ic_amplitude: float = 1.0
    ic_gaussian_center: float = 0.5          # per-axis center (same on each axis)
    ic_gaussian_width: float = 0.1
    ic_square_lo: float = 0.25
    ic_square_hi: float = 0.75
    ic_random_seed: int = 0
    ic_random_modes: int = 8

    # ---- 7. Boundary conditions ---------------------------------------------
    # 1D-friendly aliases (kept for backward compatibility): these map to the
    # xmin / xmax faces respectively when ``boundary_conditions`` is not given.
    left_bc: BCSpec = ("Dirichlet", 0.0)
    right_bc: BCSpec = ("Dirichlet", 0.0)
    # General, dimension-agnostic form: a mapping from face name to spec, e.g.
    #   boundary_conditions={
    #       "xmin": ("Dirichlet", 0.0), "xmax": ("Dirichlet", 0.0),
    #       "ymin": ("Neumann", 0.0),  "ymax": ("Neumann", 0.0),
    #   }
    # Any face omitted from the mapping defaults to homogeneous Dirichlet u=0.
    boundary_conditions: Optional[Dict[str, BCSpec]] = None

    # ---- 8. Numerical method / solver ---------------------------------------
    newton_rtol: float = 1.0e-8
    newton_atol: float = 1.0e-10
    newton_max_it: int = 50
    linear_solver: str = "preonly"           # PETSc KSP type
    preconditioner: str = "lu"               # PETSc PC type
    petsc_options: Dict[str, object] = field(default_factory=dict)

    # ---- 9. Output -----------------------------------------------------------
    output_dir: str = "output"
    save_numpy: bool = True
    save_csv: bool = False
    save_hdf5: bool = False
    save_xdmf: bool = True
    save_vtk: bool = False
    save_every: int = 1                      # store solution every k steps

    # ---- 10. PINN dataset ----------------------------------------------------
    generate_pinn_dataset: bool = True
    pinn_dataset_basename: str = "burgers_dataset"
    pinn_export_npy: bool = True
    pinn_export_csv: bool = True

    # ---- 11/12/13. Diagnostics & logging ------------------------------------
    compute_diagnostics: bool = True
    log_level: str = "INFO"
    stability_max_abs: float = 1.0e6         # blow-up guard threshold

    # ---- Metadata ------------------------------------------------------------
    experiment_name: str = "burgers_run"

    # --- derived helpers ------------------------------------------------------
    @property
    def gdim(self) -> int:
        """Geometric (spatial) dimension — an alias for ``dimension``."""
        return self.dimension

    @property
    def spatial_columns(self) -> Tuple[str, ...]:
        """Names of the spatial coordinate columns, e.g. ('x', 'y')."""
        return ("x", "y", "z")[: self.dimension]

    def axis_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """Return ((min, max), ...) for each active spatial axis."""
        all_bounds = (
            (self.xmin, self.xmax),
            (self.ymin, self.ymax),
            (self.zmin, self.zmax),
        )
        return all_bounds[: self.dimension]

    def face_axis_target(self, face: str) -> Tuple[int, float]:
        """Map a face name to its ``(axis, coordinate value)``.

        e.g. ``"ymax"`` -> ``(1, self.ymax)``. Used by the boundary-condition
        builder to locate the DOFs lying on that face.
        """
        axis, which = FACE_AXIS[face]
        lo, hi = self.axis_bounds()[axis]
        return axis, (lo if which == "min" else hi)

    def resolved_boundary_conditions(self) -> Dict[str, BCSpec]:
        """Return the concrete ``{face: spec}`` map for this dimension.

        Resolution order:
          1. If ``boundary_conditions`` is given, use it (validated to contain
             only faces valid for the dimension); faces left unspecified fall
             back to homogeneous Dirichlet ``u=0``.
          2. Otherwise, in 1D use the ``left_bc``/``right_bc`` aliases.
          3. Otherwise (2D/3D with nothing specified) default every face to
             homogeneous Dirichlet ``u=0``.
        """
        faces = FACE_NAMES[self.dimension]
        if self.boundary_conditions is not None:
            resolved: Dict[str, BCSpec] = {f: ("Dirichlet", 0.0) for f in faces}
            for face, spec in self.boundary_conditions.items():
                if face not in faces:
                    raise ValueError(
                        f"Boundary face '{face}' is not valid for dimension "
                        f"{self.dimension}. Valid faces: {faces}."
                    )
                resolved[face] = spec
            return resolved
        if self.dimension == 1:
            return {"xmin": self.left_bc, "xmax": self.right_bc}
        return {f: ("Dirichlet", 0.0) for f in faces}

    # --- validation ----------------------------------------------------------
    def __post_init__(self) -> None:
        """Fail fast on inconsistent configuration.

        Because the dataclass is frozen we validate here and raise immediately;
        it is far cheaper to reject a bad config than to debug a NaN blow-up an
        hour into a run.
        """
        if self.dimension not in (1, 2, 3):
            raise ValueError(
                f"dimension={self.dimension} is invalid; must be 1, 2, or 3."
            )

        # Per-axis domain + resolution checks, only for the active axes.
        axis_specs = (
            ("x", self.xmin, self.xmax, self.nx),
            ("y", self.ymin, self.ymax, self.ny),
            ("z", self.zmin, self.zmax, self.nz),
        )
        for name, lo, hi, n in axis_specs[: self.dimension]:
            if hi <= lo:
                raise ValueError(f"Require {name}max > {name}min.")
            if n < 1:
                raise ValueError(f"Require n{name} >= 1.")

        if self.polynomial_degree < 1:
            raise ValueError("Require polynomial_degree >= 1.")
        if self.nu <= 0.0:
            raise ValueError("Viscosity nu must be strictly positive.")
        if self.T <= 0.0 or self.dt <= 0.0:
            raise ValueError("Require T > 0 and dt > 0.")
        if self.dt > self.T:
            raise ValueError("Require dt <= T.")
        if self.initial_condition == ICKind.CUSTOM.value and self.ic_function is None:
            raise ValueError("initial_condition='custom' requires ic_function.")

        # Boundary-condition specs must be (type, value) pairs. Validate both
        # the 1D aliases and the general mapping (whichever is in play).
        specs = list(self.resolved_boundary_conditions().items())
        valid_types = {t.value for t in BCType}
        for face, spec in specs:
            if (not isinstance(spec, (tuple, list))) or len(spec) != 2:
                raise ValueError(f"{face} BC must be a (type, value) pair.")
            if spec[0] not in valid_types:
                raise ValueError(
                    f"{face} BC type '{spec[0]}' is unknown; "
                    f"valid: {sorted(valid_types)}."
                )

    # --- serialization -------------------------------------------------------
    def to_json(self) -> str:
        """Serialize to JSON (callables are noted, not pickled)."""
        d = asdict(self)
        d["ic_function"] = (
            None if self.ic_function is None
            else f"<callable {getattr(self.ic_function, '__name__', 'lambda')}>"
        )
        # Record the resolved BC map so output/config.json is self-contained
        # for whatever dimension was run.
        d["resolved_boundary_conditions"] = {
            f: list(s) for f, s in self.resolved_boundary_conditions().items()
        }
        return json.dumps(d, indent=2, default=str)
