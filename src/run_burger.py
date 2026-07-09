"""
src/run_burger.py
=================

Experiment configuration + entry point — *no solver logic lives here*.

Edit the parameters in the ``build_config_*`` presets below, choose which one to
run with the ``DIMENSION`` switch, and run from the repository root::

    python -m src.run_burger

(The ``-m`` form matters: it puts the repo root on ``sys.path`` so the ``src``
package and its absolute imports resolve. Running ``python src/run_burger.py``
directly would make ``src/`` itself the script directory and break
``from src...`` imports — that was bug B2 in the original layout.)

The same solver runs 1D, 2D and 3D — only the config changes. See
``docs/burgers_fem_guide.md`` for a walk-through of every knob.
"""

from __future__ import annotations

# numpy is imported for convenience when supplying a custom ic_function below.
import numpy as np  # noqa: F401

from src.config.config import BurgersConfig
from src.solver.burgers_solver import BurgersSolver


# ===========================================================================
# CHOOSE THE DIMENSION HERE: 1, 2, or 3.
# ===========================================================================
DIMENSION = 1


# ---------------------------------------------------------------------------
# 1D — interval domain [xmin, xmax], nx cells.
# ---------------------------------------------------------------------------
def build_config_1d() -> BurgersConfig:
    return BurgersConfig(
        experiment_name="burgers1d_sin_nu0p01",
        dimension=1,
        # domain + mesh (only x is used in 1D)
        xmin=0.0, xmax=1.0,
        nx=300,
        element_type="Lagrange", polynomial_degree=1,
        # time
        T=1.0, dt=0.002, time_integrator="backward_euler",
        # physics
        nu=0.01,
        # initial condition: sin|gaussian|square|shock|random_smooth|custom
        initial_condition="sin",
        # For a custom IC instead (1D receives the scalar x-row):
        #   initial_condition="custom", ic_function=lambda x: np.sin(2*np.pi*x),
        # boundary conditions (1D aliases map to the xmin/xmax faces)
        left_bc=("Dirichlet", 0.0),
        right_bc=("Dirichlet", 0.0),
        # solver
        newton_rtol=1e-8, newton_atol=1e-10, newton_max_it=50,
        linear_solver="preonly", preconditioner="lu",
        # output
        output_dir="output",
        save_numpy=True, save_csv=True, save_xdmf=True, save_every=1,
        # PINN dataset -> columns [x, t, u]
        generate_pinn_dataset=True, pinn_dataset_basename="burgers_dataset",
        compute_diagnostics=True, log_level="INFO",
    )


# ---------------------------------------------------------------------------
# 2D — rectangle domain [xmin,xmax] x [ymin,ymax], nx*ny triangular cells.
# ---------------------------------------------------------------------------
def build_config_2d() -> BurgersConfig:
    return BurgersConfig(
        experiment_name="burgers2d_sin_nu0p01",
        dimension=2,
        # domain + mesh (x and y are used in 2D)
        xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
        nx=64, ny=64,
        element_type="Lagrange", polynomial_degree=1,
        # time — 2D/3D are heavier, so a coarser step/shorter horizon is typical
        T=0.5, dt=0.005, time_integrator="backward_euler",
        # physics
        nu=0.02,
        # initial condition (sin is separable in 2D: sin(2pi x)*sin(2pi y))
        initial_condition="sin",
        # Generalized BCs: name each face. Omitted faces default to Dirichlet 0.
        boundary_conditions={
            "xmin": ("Dirichlet", 0.0), "xmax": ("Dirichlet", 0.0),
            "ymin": ("Dirichlet", 0.0), "ymax": ("Dirichlet", 0.0),
        },
        # solver — an iterative solve scales better than dense LU in 2D/3D
        newton_rtol=1e-8, newton_atol=1e-10, newton_max_it=50,
        linear_solver="gmres", preconditioner="hypre",
        petsc_options={"pc_hypre_type": "boomeramg"},
        # output
        output_dir="output",
        save_numpy=True, save_csv=True, save_xdmf=True, save_every=2,
        # PINN dataset -> columns [x, y, t, u]
        generate_pinn_dataset=True, pinn_dataset_basename="burgers2d_dataset",
        compute_diagnostics=True, log_level="INFO",
    )


# ---------------------------------------------------------------------------
# 3D — box domain [xmin,xmax] x [ymin,ymax] x [zmin,zmax], nx*ny*nz tets.
# ---------------------------------------------------------------------------
def build_config_3d() -> BurgersConfig:
    return BurgersConfig(
        experiment_name="burgers3d_gaussian_nu0p05",
        dimension=3,
        # domain + mesh (x, y and z are all used in 3D)
        xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0,
        nx=24, ny=24, nz=24,       # keep modest: DOFs grow as nx*ny*nz
        element_type="Lagrange", polynomial_degree=1,
        # time
        T=0.3, dt=0.01, time_integrator="backward_euler",
        # physics
        nu=0.05,
        # initial condition (radial Gaussian bump centered in the box)
        initial_condition="gaussian",
        ic_gaussian_center=0.5, ic_gaussian_width=0.12,
        # BCs on all six faces (omitted faces default to Dirichlet 0)
        boundary_conditions={
            "xmin": ("Dirichlet", 0.0), "xmax": ("Dirichlet", 0.0),
            "ymin": ("Dirichlet", 0.0), "ymax": ("Dirichlet", 0.0),
            "zmin": ("Dirichlet", 0.0), "zmax": ("Dirichlet", 0.0),
        },
        # solver — iterative + AMG is essential at 3D problem sizes
        newton_rtol=1e-7, newton_atol=1e-9, newton_max_it=50,
        linear_solver="gmres", preconditioner="hypre",
        petsc_options={"pc_hypre_type": "boomeramg"},
        # output — 3D fields are inspected in ParaView via XDMF, not inline plots
        output_dir="output",
        save_numpy=True, save_csv=False, save_xdmf=True, save_every=1,
        # PINN dataset -> columns [x, y, z, t, u]
        generate_pinn_dataset=True, pinn_dataset_basename="burgers3d_dataset",
        compute_diagnostics=True, log_level="INFO",
    )


def build_config() -> BurgersConfig:
    """Return the configuration for the dimension selected by ``DIMENSION``."""
    builders = {1: build_config_1d, 2: build_config_2d, 3: build_config_3d}
    if DIMENSION not in builders:
        raise ValueError(f"DIMENSION must be 1, 2, or 3 (got {DIMENSION}).")
    return builders[DIMENSION]()


def main() -> None:
    config = build_config()
    solver = BurgersSolver(config)
    solver.solve()

    # Optional post-processing figures (comment out for headless batch runs).
    # 1D renders line/space-time plots; 2D renders filled-contour panels;
    # 3D skips inline plots (inspect the XDMF output in ParaView instead).
    solver.plot_snapshots()
    solver.plot_heatmap()
    solver.plot_solution(time_index=-1)
    # solver.plot_animation()   # requires an animation writer (ffmpeg/pillow)


if __name__ == "__main__":
    main()
