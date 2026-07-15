from __future__ import annotations
import logging
from typing import List
import numpy as np
from src.config.config import BurgersConfig
from src.config.enums import BCType
from src.fenics_backend import fem, default_scalar_type


class BoundaryConditionBuilder:
    """Translate a config's boundary specs into dolfinx ``DirichletBC`` objects."""

    def __init__(
        self,
        config: BurgersConfig,
        mesh,
        V,
        log: logging.Logger,
    ) -> None:
        self.cfg = config
        self.mesh = mesh
        self.V = V
        self.log = log

    def _locate_face_dofs(self, axis: int, target: float) -> np.ndarray:
        """Return the DOFs lying on the hyperplane ``x_axis == target``."""
        lo, hi = self.cfg.axis_bounds()[axis]
        atol = 1e-12 + 1e-9 * abs(hi - lo)

        def on_face(x: np.ndarray) -> np.ndarray:
            return np.isclose(x[axis], target, atol=atol)

        return fem.locate_dofs_geometrical(self.V, on_face)

    def build(self) -> List["fem.DirichletBC"]:
        """Return the list of Dirichlet BC objects for this configuration.

        Iterates over every face valid for the configured dimension (2 faces per
        axis) using the config's resolved ``{face: spec}`` map.
        """
        bcs: List["fem.DirichletBC"] = []
        for face, spec in self.cfg.resolved_boundary_conditions().items():
            axis, target = self.cfg.face_axis_target(face)
            bc_type, value = spec[0], spec[1]

            if bc_type == BCType.DIRICHLET.value:
                dofs = self._locate_face_dofs(axis, target)
                g = fem.Constant(self.mesh, default_scalar_type(float(value)))
                bcs.append(fem.dirichletbc(g, dofs, self.V))
                self.log.info("%s boundary: Dirichlet u=%g", face, float(value))
            elif bc_type == BCType.NEUMANN.value:
                if float(value) != 0.0:
                    raise NotImplementedError(
                        "Non-homogeneous Neumann requires an extra surface "
                        "integral in BurgersPDE.residual; only homogeneous "
                        "(natural) Neumann is wired in."
                    )
                self.log.info("%s boundary: natural (homogeneous Neumann)", face)
            elif bc_type == BCType.PERIODIC.value:
                raise NotImplementedError(
                    "Periodic BCs need a periodic constraint map (dolfinx_mpc); "
                    "reserved extension point."
                )
            elif bc_type == BCType.ROBIN.value:
                raise NotImplementedError(
                    "Robin BCs add alpha*u*v and beta*v surface integrals to the "
                    "residual; reserved extension point."
                )
            else:
                raise ValueError(f"Unknown BC type '{bc_type}'.")
        return bcs
