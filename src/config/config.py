"""
src/config/config.py
The declarative description of one experiment, generalized across 1D/2D/3D.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from src.config.enums import ICKind, BCType, BCSpec

FACE_NAMES: Dict[int, Tuple[str, ...]] = {
    1: ("xmin", "xmax"),
    2: ("xmin", "xmax", "ymin", "ymax"),
    3: ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
}

FACE_AXIS: Dict[str, Tuple[int, str]] = {
    "xmin": (0, "min"), "xmax": (0, "max"),
    "ymin": (1, "min"), "ymax": (1, "max"),
    "zmin": (2, "min"), "zmax": (2, "max"),
}


@dataclass(frozen=True)
class BurgersConfig:
    dimension: int = 1
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    zmin: float = 0.0
    zmax: float = 1.0

    
    nx: int = 300                         
    ny: int = 64                             
    nz: int = 64                          
    element_type: str = "Lagrange"
    polynomial_degree: int = 1
    T: float = 1.0
    dt: float = 0.002
    time_integrator: str = "backward_euler"
    adaptive: bool = False                

    
    nu: float = 0.01                      
    initial_condition: str = "sin"
    ic_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
    
    ic_amplitude: float = 1.0
    ic_gaussian_center: float = 0.5         
    ic_gaussian_width: float = 0.1
    ic_square_lo: float = 0.25
    ic_square_hi: float = 0.75
    ic_random_seed: int = 0
    ic_random_modes: int = 8

    
    left_bc: BCSpec = ("Dirichlet", 0.0)
    right_bc: BCSpec = ("Dirichlet", 0.0)
    boundary_conditions: Optional[Dict[str, BCSpec]] = None

    
    newton_rtol: float = 1.0e-8
    newton_atol: float = 1.0e-10
    newton_max_it: int = 50
    linear_solver: str = "preonly"           # PETSc KSP type
    preconditioner: str = "lu"               # PETSc PC type
    petsc_options: Dict[str, object] = field(default_factory=dict)

    
    output_dir: str = "output"
    save_numpy: bool = True
    save_csv: bool = False
    save_hdf5: bool = False
    save_xdmf: bool = True
    save_vtk: bool = False
    save_every: int = 1                      # store solution every k steps

    
    generate_pinn_dataset: bool = True
    pinn_dataset_basename: str = "burgers_dataset"
    pinn_export_npy: bool = True
    pinn_export_csv: bool = True

    
    compute_diagnostics: bool = True
    log_level: str = "INFO"
    stability_max_abs: float = 1.0e6         # blow-up guard threshold

   
    experiment_name: str = "burgers_run"

    
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

    
    def __post_init__(self) -> None:
        if self.dimension not in (1, 2, 3):
            raise ValueError(
                f"dimension={self.dimension} is invalid; must be 1, 2, or 3."
            )

        
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

    
    def to_json(self) -> str:
        """Serialize to JSON (callables are noted, not pickled)."""
        d = asdict(self)
        d["ic_function"] = (
            None if self.ic_function is None
            else f"<callable {getattr(self.ic_function, '__name__', 'lambda')}>"
        )
        d["resolved_boundary_conditions"] = {
            f: list(s) for f, s in self.resolved_boundary_conditions().items()
        }
        return json.dumps(d, indent=2, default=str)
