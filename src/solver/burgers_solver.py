from __future__ import annotations
import logging
import os
from typing import Dict, List, Optional
import numpy as np
from src.config.config import BurgersConfig
from src.fenics_backend import (
    require_dolfinx,
    MPI, ufl,
    ufl_element, fem, dmesh, io as fio, default_scalar_type,
    NonlinearProblem,
)
from src.utils.logging_utils import build_logger
from src.physics.initial_conditions import InitialConditionFactory
from src.physics.boundary_conditions import BoundaryConditionBuilder
from src.physics.pde import BurgersPDE
from src.diagnostics.diagnostics import compute_diagnostics, check_stability
from src.io import writers, pinn_dataset
from src.viz import plots


class BurgersSolver:
    """End-to-end driver: mesh -> spaces -> IC -> BCs -> time loop -> I/O."""

    def __init__(self, config: BurgersConfig) -> None:
        require_dolfinx()
        self.cfg = config
        self.log = build_logger(
            level=getattr(logging, config.log_level.upper(), logging.INFO)
        )
        self.comm = MPI.COMM_WORLD
        self.mesh: Optional["dmesh.Mesh"] = None
        self.V: Optional["fem.FunctionSpace"] = None
        self.u: Optional["fem.Function"] = None      
        self.u_n: Optional["fem.Function"] = None    
        self.times: List[float] = []                   
        self.snapshots: List[np.ndarray] = []        
        self.coords: Optional[np.ndarray] = None       
        self._sort_idx: Optional[np.ndarray] = None    
        self.diagnostics: List[Dict[str, float]] = []

        self.pde = BurgersPDE(config.nu)
        self._prepare_output_dirs()

    def _prepare_output_dirs(self) -> None:
        """Create the output subdirectory tree (rank 0 only)."""
        if self.comm.rank != 0:
            return
        base = self.cfg.output_dir
        for sub in ("numpy", "csv", "xdmf", "hdf5", "figures", "animations"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
            
        with open(os.path.join(base, "config.json"), "w") as fh:
            fh.write(self.cfg.to_json())

    def _build_mesh(self) -> None:
        cfg = self.cfg
        if cfg.dimension == 1:
            self.mesh = dmesh.create_interval(
                self.comm, cfg.nx, [cfg.xmin, cfg.xmax]
            )
            self.log.info("Mesh (1D): %d cells on [%g, %g]",
                          cfg.nx, cfg.xmin, cfg.xmax)
        elif cfg.dimension == 2:
            self.mesh = dmesh.create_rectangle(
                self.comm,
                [np.array([cfg.xmin, cfg.ymin]),
                 np.array([cfg.xmax, cfg.ymax])],
                [cfg.nx, cfg.ny],
                cell_type=dmesh.CellType.triangle,
            )
            self.log.info("Mesh (2D): %dx%d cells on [%g,%g]x[%g,%g]",
                          cfg.nx, cfg.ny, cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax)
        elif cfg.dimension == 3:
            self.mesh = dmesh.create_box(
                self.comm,
                [np.array([cfg.xmin, cfg.ymin, cfg.zmin]),
                 np.array([cfg.xmax, cfg.ymax, cfg.zmax])],
                [cfg.nx, cfg.ny, cfg.nz],
                cell_type=dmesh.CellType.tetrahedron,
            )
            self.log.info(
                "Mesh (3D): %dx%dx%d cells on [%g,%g]x[%g,%g]x[%g,%g]",
                cfg.nx, cfg.ny, cfg.nz,
                cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax,
            )
        else:  
            raise NotImplementedError(
                f"dimension={cfg.dimension} not supported (must be 1, 2, or 3)."
            )

    def _build_space(self) -> None:
        """Create the scalar CG function space and the two state functions."""
        cfg = self.cfg
        element = ufl_element(
            cfg.element_type,
            self.mesh.basix_cell(),
            cfg.polynomial_degree,
        )
        self.V = fem.functionspace(self.mesh, element)
        self.u = fem.Function(self.V, name="u")        
        self.u_n = fem.Function(self.V, name="u_n")    
        self.log.info(
            "Function space: %s degree %d, %d global DOFs",
            cfg.element_type, cfg.polynomial_degree,
            self.V.dofmap.index_map.size_global,
        )

    def _apply_initial_condition(self) -> None:
        """Interpolate u0(x) into both u and u_n."""
        u0 = InitialConditionFactory(self.cfg).build()
        self.u_n.interpolate(u0)
        self.u.interpolate(u0)
        self.u_n.x.scatter_forward()
        self.u.x.scatter_forward()
        self.log.info("Initial condition '%s' applied.", self.cfg.initial_condition)

    def _build_boundary_conditions(self) -> List["fem.DirichletBC"]:
        return BoundaryConditionBuilder(
            self.cfg, self.mesh, self.V, self.log
        ).build()

    def _build_nonlinear_problem(
        self, bcs: List["fem.DirichletBC"], dt_const: "fem.Constant"
    ) -> "NonlinearProblem":
        v = ufl.TestFunction(self.V)
        F = self.pde.residual(self.u, self.u_n, v, dt_const)
        J = ufl.derivative(F, self.u)

        petsc_options = {
            "snes_type": "newtonls",        
            "snes_rtol": self.cfg.newton_rtol,
            "snes_atol": self.cfg.newton_atol,
            "snes_max_it": self.cfg.newton_max_it,
            "ksp_type": self.cfg.linear_solver, 
            "pc_type": self.cfg.preconditioner,
        }
        petsc_options.update(self.cfg.petsc_options)

        prefix = f"burgers_{id(self):x}_"

        problem = NonlinearProblem(
            F, self.u, bcs=bcs, J=J,
            petsc_options_prefix=prefix,
            petsc_options=petsc_options,
        )
        return problem

    def _init_coordinate_ordering(self) -> None:
        gdim = self.cfg.dimension
        x_dofs = self.V.tabulate_dof_coordinates()[:, :gdim]
        keys = tuple(x_dofs[:, i] for i in reversed(range(gdim)))
        self._sort_idx = np.lexsort(keys)
        self.coords = x_dofs[self._sort_idx]

    def _record_snapshot(self, t: float) -> None:
        """Store the current solution (spatially sorted) and its time level."""
        local = self.u.x.array.real.copy()
        self.times.append(float(t))
        self.snapshots.append(local[self._sort_idx].copy())

    def _snapshot_matrix(self) -> np.ndarray:
        """Return solution history as a (n_times, n_points) array."""
        return np.vstack(self.snapshots) if self.snapshots else np.empty((0, 0))

    def solve(self) -> None:
        """Run the full time-dependent simulation."""
        cfg = self.cfg
        self.log.info("=== Burgers simulation: %s ===", cfg.experiment_name)
        self._build_mesh()
        self._build_space()
        self._apply_initial_condition()
        self._init_coordinate_ordering()

        bcs = self._build_boundary_conditions()
        dt_const = fem.Constant(self.mesh, default_scalar_type(cfg.dt))
        problem = self._build_nonlinear_problem(bcs, dt_const)

        xdmf_writer = None
        if cfg.save_xdmf:
            xdmf_dir = os.path.join(cfg.output_dir, "xdmf")
            if self.comm.rank == 0:
                os.makedirs(xdmf_dir, exist_ok=True)
            self.comm.barrier()
            xdmf_path = os.path.join(xdmf_dir, f"{cfg.experiment_name}.xdmf")
            xdmf_writer = fio.XDMFFile(self.comm, xdmf_path, "w")
            xdmf_writer.write_mesh(self.mesh)

        t = 0.0
        self._record_snapshot(t)
        if cfg.compute_diagnostics:
            d0 = compute_diagnostics(self.u, self.comm, t)
            self.diagnostics.append(d0)
            check_stability(d0, cfg.stability_max_abs)
        if xdmf_writer is not None:
            xdmf_writer.write_function(self.u, t)

        n_steps = int(round(cfg.T / cfg.dt))
        self.log.info("Marching %d steps, dt=%g, T=%g, nu=%g",
                      n_steps, cfg.dt, cfg.T, cfg.nu)

        for step in range(1, n_steps + 1):
            t = step * cfg.dt

            
            problem.solve()
            self.u.x.scatter_forward()

            snes = problem.solver
            n_it = snes.getIterationNumber()
            converged = snes.getConvergedReason() > 0

            if not converged:
                raise RuntimeError(
                    f"Newton (SNES) failed to converge at step {step} "
                    f"(t={t:.4g}); reason={snes.getConvergedReason()}."
                )

            # Advance: u^n <- u^{n+1}
            self.u_n.x.array[:] = self.u.x.array
            self.u_n.x.scatter_forward()

            # Diagnostics & logging
            if cfg.compute_diagnostics:
                diag = compute_diagnostics(self.u, self.comm, t)
                check_stability(diag, cfg.stability_max_abs)
                self.diagnostics.append(diag)
                self.log.info(
                    "step %4d/%d  t=%.4f  Newton_it=%2d  umax=%+.4e umin=%+.4e "
                    "L2=%.4e mass=%+.4e",
                    step, n_steps, t, n_it,
                    diag["umax"], diag["umin"], diag["l2"], diag["mass"],
                )
            else:
                self.log.info("step %4d/%d  t=%.4f  Newton_it=%2d",
                              step, n_steps, t, n_it)

            # Record & stream at the requested cadence.
            if step % cfg.save_every == 0 or step == n_steps:
                self._record_snapshot(t)
                if xdmf_writer is not None:
                    xdmf_writer.write_function(self.u, t)

        if xdmf_writer is not None:
            xdmf_writer.close()

        self.log.info("Time integration complete: %d snapshots recorded.",
                      len(self.snapshots))

        # 6) Persist everything requested
        self._write_outputs()
        if cfg.generate_pinn_dataset:
            self.generate_pinn_dataset()

    def _write_outputs(self) -> None:
        """Delegate persistence to the io.writers module."""
        writers.write_outputs(
            self.cfg,
            self.coords,
            np.asarray(self.times),
            self._snapshot_matrix(),
            self.diagnostics,
            self.u,
            self.comm,
            self.log,
        )

    def generate_pinn_dataset(self) -> np.ndarray:
        """Delegate PINN-dataset export to the io.pinn_dataset module."""
        return pinn_dataset.generate_pinn_dataset(
            self.cfg,
            self.coords,
            np.asarray(self.times),
            self._snapshot_matrix(),
            self.comm,
            self.log,
        )

    
    def plot_solution(self, time_index: int = -1, show: bool = False,
                      save: bool = True) -> None:
        plots.plot_solution(self.cfg, self.coords, np.asarray(self.times),
                            self._snapshot_matrix(), self.log,
                            time_index=time_index, show=show, save=save)

    def plot_snapshots(self, n_curves: int = 6, show: bool = False,
                       save: bool = True) -> None:
        plots.plot_snapshots(self.cfg, self.coords, np.asarray(self.times),
                            self._snapshot_matrix(), self.log,
                            n_curves=n_curves, show=show, save=save)

    def plot_heatmap(self, show: bool = False, save: bool = True) -> None:
        plots.plot_heatmap(self.cfg, self.coords, np.asarray(self.times),
                          self._snapshot_matrix(), self.log,
                          show=show, save=save)

    def plot_animation(self, save: bool = True, fps: int = 20) -> None:
        plots.plot_animation(self.cfg, self.coords, np.asarray(self.times),
                            self._snapshot_matrix(), self.log,
                            save=save, fps=fps)
