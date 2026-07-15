from __future__ import annotations
import numpy as np  # noqa: F401

from src.config.config import BurgersConfig
from src.solver.burgers_solver import BurgersSolver
DIMENSION = 1

def build_config_1d() -> BurgersConfig:
    return BurgersConfig(
        experiment_name="burgers1d_sin_nu0p01",
        dimension=1,
        xmin=0.0, xmax=1.0,
        nx=500,
        element_type="Lagrange", polynomial_degree=1,
        T=1.0, dt=0.002, time_integrator="backward_euler",
        nu=0.01,
        initial_condition="sin",
        left_bc=("Dirichlet", 0.0),
        right_bc=("Dirichlet", 0.0),
        newton_rtol=1e-8, newton_atol=1e-10, newton_max_it=50,
        linear_solver="preonly", preconditioner="lu",
        output_dir="output",
        save_numpy=True, save_csv=True, save_xdmf=True, save_every=1,
        generate_pinn_dataset=True, pinn_dataset_basename="burgers_dataset",
        compute_diagnostics=True, log_level="INFO",
    )


def build_config_2d() -> BurgersConfig:
    return BurgersConfig(
        experiment_name="burgers2d_sin_nu0p01",
        dimension=2,
        xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
        nx=64, ny=64,
        element_type="Lagrange", polynomial_degree=1,
        T=0.5, dt=0.005, time_integrator="backward_euler",
        nu=0.02,
        initial_condition="sin",
        boundary_conditions={
            "xmin": ("Dirichlet", 0.0), "xmax": ("Dirichlet", 0.0),
            "ymin": ("Dirichlet", 0.0), "ymax": ("Dirichlet", 0.0),
        },
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
    solver.plot_snapshots()
    solver.plot_heatmap()
    solver.plot_solution(time_index=-1)
    solver.plot_animation()   # requires an animation writer (ffmpeg/pillow)


if __name__ == "__main__":
    main()
