"""Classical finite-difference / GMRES solver for the 1D viscous Burgers equation.

Implements a semi-implicit scheme in which the diffusion operator is treated
implicitly and the nonlinear convection term explicitly:

    (I - nu * dt * L) u^{n+1} = u^n - dt * C(u^n)

The resulting sparse linear system is solved with SciPy's GMRES.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "BurgersConfig",
    "GMRESConfig",
    "SolverMetrics",
    "BurgersSolution",
    "InitialConditionFactory",
    "cfl_timestep",
    "relative_l2_error",
    "linf_error",
    "build_diffusion_matrix",
    "build_first_derivative_matrix",
    "GMRESBurgersSolver",
    "plot_solution_evolution",
    "plot_error_history",
    "plot_convergence",
    "plot_runtime_scaling",
    "plot_comparison",
]

logger = logging.getLogger(__name__)
if not logger.handlers:  # pragma: no cover - convenience for interactive use
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

InitialConditionName = Literal["riemann", "sine", "gaussian", "smooth_step"]


# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BurgersConfig:
    """Physical and discretisation parameters for the 1D Burgers problem."""

    n_points: int = 128
    x_min: float = 0.0
    x_max: float = 1.0
    viscosity: float = 0.01
    t_final: float = 1.0
    cfl: float = 0.1
    dt_fixed: float | None = None
    adaptive_dt: bool = True
    max_steps: int = 200_000
    initial_condition: InitialConditionName = "riemann"
    u_left: float = 1.0
    u_right: float = 0.0
    bc_left: float = 1.0
    bc_right: float = 0.0
    step_location: float = 0.5
    smooth_width: float = 0.02
    gaussian_center: float = 0.5
    gaussian_width: float = 0.1
    store_every: int = 1

    def __post_init__(self) -> None:
        if self.n_points < 4:
            raise ValueError("n_points must be at least 4.")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must exceed x_min.")
        if self.viscosity <= 0.0:
            raise ValueError("viscosity must be strictly positive.")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be strictly positive.")
        if not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1].")
        if self.dt_fixed is not None and self.dt_fixed <= 0.0:
            raise ValueError("dt_fixed must be strictly positive when given.")
        if self.store_every < 1:
            raise ValueError("store_every must be >= 1.")

    @property
    def domain_length(self) -> float:
        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        return self.domain_length / (self.n_points - 1)

    @property
    def reynolds(self) -> float:
        return 1.0 / (2.0 * self.viscosity)

    def grid(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_points)


@dataclass(frozen=True, slots=True)
class GMRESConfig:
    """Krylov solver options."""

    rtol: float = 1e-10
    atol: float = 0.0
    restart: int = 50
    maxiter: int = 1_000
    use_ilu_preconditioner: bool = True
    ilu_drop_tol: float = 1e-5
    ilu_fill_factor: float = 10.0

    def __post_init__(self) -> None:
        if self.rtol <= 0.0:
            raise ValueError("rtol must be strictly positive.")
        if self.restart < 1:
            raise ValueError("restart must be >= 1.")
        if self.maxiter < 1:
            raise ValueError("maxiter must be >= 1.")


@dataclass(slots=True)
class SolverMetrics:
    """Runtime and resource measurements for a solver run."""

    wall_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    n_steps: int = 0
    total_krylov_iterations: int = 0
    mean_krylov_iterations: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class BurgersSolution:
    """Container for a time-resolved Burgers solution."""

    x: np.ndarray
    times: np.ndarray
    snapshots: np.ndarray  # shape (n_saved, n_points)
    residual_history: np.ndarray
    dt_history: np.ndarray
    config: BurgersConfig
    metrics: SolverMetrics = field(default_factory=SolverMetrics)

    @property
    def final(self) -> np.ndarray:
        return self.snapshots[-1]

    @property
    def initial(self) -> np.ndarray:
        return self.snapshots[0]

    def interpolate_at(self, t: float) -> np.ndarray:
        """Linear interpolation of the solution field at an arbitrary time."""
        if not self.times[0] <= t <= self.times[-1]:
            raise ValueError(
                f"t={t} lies outside the stored interval "
                f"[{self.times[0]}, {self.times[-1]}]."
            )
        idx = int(np.searchsorted(self.times, t))
        if idx == 0:
            return self.snapshots[0].copy()
        t0, t1 = self.times[idx - 1], self.times[idx]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1.0 - w) * self.snapshots[idx - 1] + w * self.snapshots[idx]

    def save(self, path: str | Path) -> Path:
        """Persist solution arrays and configuration to a .npz archive."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            times=self.times,
            snapshots=self.snapshots,
            residual_history=self.residual_history,
            dt_history=self.dt_history,
            config_json=json.dumps(asdict(self.config)),
            metrics_json=json.dumps(self.metrics.as_dict()),
        )
        logger.info("Saved solution to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BurgersSolution":
        """Restore a solution previously written by :meth:`save`."""
        path = Path(path).with_suffix(".npz")
        if not path.exists():
            raise FileNotFoundError(f"No solution archive at {path}.")
        with np.load(path, allow_pickle=False) as data:
            config = BurgersConfig(**json.loads(str(data["config_json"])))
            metrics = SolverMetrics(**json.loads(str(data["metrics_json"])))
            return cls(
                x=data["x"],
                times=data["times"],
                snapshots=data["snapshots"],
                residual_history=data["residual_history"],
                dt_history=data["dt_history"],
                config=config,
                metrics=metrics,
            )


# --------------------------------------------------------------------------- #
# Shared numerical helpers
# --------------------------------------------------------------------------- #
class InitialConditionFactory:
    """Builds the initial velocity profiles used across all solvers."""

    @staticmethod
    def build(config: BurgersConfig) -> np.ndarray:
        x = config.grid()
        name = config.initial_condition
        if name == "riemann":
            u = np.where(x <= config.step_location, config.u_left, config.u_right)
        elif name == "smooth_step":
            u = config.u_right + 0.5 * (config.u_left - config.u_right) * (
                1.0 - np.tanh((x - config.step_location) / config.smooth_width)
            )
        elif name == "sine":
            u = np.sin(2.0 * np.pi * (x - config.x_min) / config.domain_length)
        elif name == "gaussian":
            u = np.exp(
                -((x - config.gaussian_center) ** 2) / (2.0 * config.gaussian_width**2)
            )
        else:  # pragma: no cover - guarded by Literal typing
            raise ValueError(f"Unknown initial condition '{name}'.")
        u = u.astype(np.float64)
        u[0] = config.bc_left
        u[-1] = config.bc_right
        return u

    @staticmethod
    def callable_for(config: BurgersConfig) -> Callable[[np.ndarray], np.ndarray]:
        """Return a mesh-free callable u0(x) matching the configured profile."""
        name = config.initial_condition

        def u0(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            if name == "riemann":
                return np.where(x <= config.step_location, config.u_left, config.u_right)
            if name == "smooth_step":
                return config.u_right + 0.5 * (config.u_left - config.u_right) * (
                    1.0 - np.tanh((x - config.step_location) / config.smooth_width)
                )
            if name == "sine":
                return np.sin(2.0 * np.pi * (x - config.x_min) / config.domain_length)
            return np.exp(
                -((x - config.gaussian_center) ** 2) / (2.0 * config.gaussian_width**2)
            )

        return u0


def cfl_timestep(u: np.ndarray, dx: float, viscosity: float, cfl: float) -> float:
    """Advective/diffusive CFL-limited timestep."""
    if dx <= 0.0:
        raise ValueError("dx must be strictly positive.")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be strictly positive.")
    u_max = float(np.max(np.abs(u)))
    advective = dx / u_max if u_max > 1e-14 else np.inf
    diffusive = dx * dx / viscosity
    return float(cfl * min(advective, diffusive))


def relative_l2_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 norm ||pred - ref|| / ||ref||."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {ref.shape}.")
    denom = float(np.linalg.norm(ref))
    if denom < 1e-30:
        return float(np.linalg.norm(pred - ref))
    return float(np.linalg.norm(pred - ref) / denom)


def linf_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Absolute L-infinity error."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {ref.shape}.")
    return float(np.max(np.abs(pred - ref)))


def build_diffusion_matrix(n: int, dx: float, fmt: str = "csr") -> sp.spmatrix:
    """Second-derivative operator with a standard three-point stencil."""
    if n < 3:
        raise ValueError("n must be at least 3.")
    diagonals = [
        np.ones(n - 1),
        -2.0 * np.ones(n),
        np.ones(n - 1),
    ]
    laplacian = sp.diags(diagonals, offsets=[-1, 0, 1], format="lil") / (dx * dx)
    # Boundary rows are replaced by identity for Dirichlet enforcement.
    laplacian[0, :] = 0.0
    laplacian[-1, :] = 0.0
    return laplacian.asformat(fmt)


def build_first_derivative_matrix(n: int, dx: float, fmt: str = "csr") -> sp.spmatrix:
    """Central first-derivative operator with one-sided boundary rows."""
    if n < 3:
        raise ValueError("n must be at least 3.")
    grad = sp.diags(
        [-np.ones(n - 1), np.zeros(n), np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="lil",
    ) / (2.0 * dx)
    grad[0, 0], grad[0, 1] = -1.0 / dx, 1.0 / dx
    grad[-1, -2], grad[-1, -1] = -1.0 / dx, 1.0 / dx
    return grad.asformat(fmt)


class _IterationCounter:
    """Callback object recording GMRES iteration counts."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, _: object) -> None:
        self.count += 1


def _gmres(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    config: GMRESConfig,
    x0: np.ndarray | None = None,
    preconditioner: spla.LinearOperator | None = None,
) -> tuple[np.ndarray, int]:
    """Thin wrapper around ``scipy.sparse.linalg.gmres`` with version shims."""
    counter = _IterationCounter()
    kwargs: dict[str, object] = {
        "x0": x0,
        "restart": config.restart,
        "maxiter": config.maxiter,
        "M": preconditioner,
        "callback": counter,
    }
    try:
        solution, info = spla.gmres(
            matrix, rhs, rtol=config.rtol, atol=config.atol, **kwargs
        )
    except TypeError:  # SciPy < 1.12 uses `tol` instead of `rtol`
        solution, info = spla.gmres(
            matrix, rhs, tol=config.rtol, atol=config.atol, **kwargs
        )
    if info > 0:
        logger.warning("GMRES did not converge within %d iterations.", info)
    elif info < 0:
        raise RuntimeError(f"GMRES failed with illegal input (info={info}).")
    return solution, counter.count


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class GMRESBurgersSolver:
    """Semi-implicit finite-difference Burgers solver using sparse GMRES."""

    def __init__(
        self,
        config: BurgersConfig | None = None,
        gmres_config: GMRESConfig | None = None,
    ) -> None:
        self.config = config or BurgersConfig()
        self.gmres_config = gmres_config or GMRESConfig()
        self.x = self.config.grid()
        self.dx = self.config.dx
        self.laplacian = build_diffusion_matrix(self.config.n_points, self.dx)
        self.gradient = build_first_derivative_matrix(self.config.n_points, self.dx)
        self._identity = sp.identity(self.config.n_points, format="csr")
        self._system_cache: dict[float, tuple[sp.spmatrix, spla.LinearOperator | None]] = {}
        self.solution: BurgersSolution | None = None

    # -- operators ---------------------------------------------------------- #
    def convection(self, u: np.ndarray) -> np.ndarray:
        """Nonlinear convection term u * du/dx."""
        return u * (self.gradient @ u)

    def diffusion(self, u: np.ndarray) -> np.ndarray:
        """Viscous term nu * d2u/dx2."""
        return self.config.viscosity * (self.laplacian @ u)

    def rhs(self, u: np.ndarray) -> np.ndarray:
        """Full spatial operator -u u_x + nu u_xx with Dirichlet rows zeroed."""
        r = -self.convection(u) + self.diffusion(u)
        r[0] = 0.0
        r[-1] = 0.0
        return r

    def steady_residual(self, u: np.ndarray) -> float:
        """L2 norm of the spatial residual, used as a convergence indicator."""
        return float(np.linalg.norm(self.rhs(u)) / np.sqrt(u.size))

    def apply_dirichlet(self, u: np.ndarray) -> np.ndarray:
        u[0] = self.config.bc_left
        u[-1] = self.config.bc_right
        return u

    # -- linear system ------------------------------------------------------ #
    def _system_matrix(
        self, dt: float
    ) -> tuple[sp.spmatrix, spla.LinearOperator | None]:
        key = round(dt, 15)
        cached = self._system_cache.get(key)
        if cached is not None:
            return cached

        matrix = (self._identity - self.config.viscosity * dt * self.laplacian).tolil()
        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0
        matrix[-1, :] = 0.0
        matrix[-1, -1] = 1.0
        matrix = matrix.tocsc()

        preconditioner: spla.LinearOperator | None = None
        if self.gmres_config.use_ilu_preconditioner and self.config.n_points >= 16:
            try:
                ilu = spla.spilu(
                    matrix,
                    drop_tol=self.gmres_config.ilu_drop_tol,
                    fill_factor=self.gmres_config.ilu_fill_factor,
                )
                preconditioner = spla.LinearOperator(matrix.shape, ilu.solve)
            except (RuntimeError, ValueError) as exc:
                logger.warning("ILU preconditioner unavailable (%s); continuing.", exc)

        result = (matrix.tocsr(), preconditioner)
        if len(self._system_cache) > 64:
            self._system_cache.clear()
        self._system_cache[key] = result
        return result

    def step(self, u: np.ndarray, dt: float) -> tuple[np.ndarray, int]:
        """Advance the solution by one semi-implicit timestep."""
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        matrix, preconditioner = self._system_matrix(dt)
        rhs = u - dt * self.convection(u)
        rhs[0] = self.config.bc_left
        rhs[-1] = self.config.bc_right
        u_next, iterations = _gmres(
            matrix, rhs, self.gmres_config, x0=u, preconditioner=preconditioner
        )
        return self.apply_dirichlet(u_next), iterations

    # -- driver ------------------------------------------------------------- #
    def solve(self, u0: np.ndarray | None = None) -> BurgersSolution:
        """Integrate from t=0 to ``config.t_final`` and return the solution."""
        cfg = self.config
        u = (
            InitialConditionFactory.build(cfg)
            if u0 is None
            else np.asarray(u0, dtype=np.float64).copy()
        )
        if u.shape != (cfg.n_points,):
            raise ValueError(f"u0 must have shape ({cfg.n_points},), got {u.shape}.")
        u = self.apply_dirichlet(u)

        times = [0.0]
        snapshots = [u.copy()]
        residuals = [self.steady_residual(u)]
        dts: list[float] = []

        tracemalloc.start()
        t_start = time.perf_counter()
        t = 0.0
        step_index = 0
        total_iterations = 0

        while t < cfg.t_final - 1e-14 and step_index < cfg.max_steps:
            if cfg.adaptive_dt or cfg.dt_fixed is None:
                dt = cfl_timestep(u, self.dx, cfg.viscosity, cfg.cfl)
            else:
                dt = cfg.dt_fixed
            dt = min(dt, cfg.t_final - t)
            if not np.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(f"Non-positive timestep encountered at t={t}.")

            u, iterations = self.step(u, dt)
            if not np.all(np.isfinite(u)):
                raise RuntimeError(f"Solution diverged at step {step_index}.")

            total_iterations += iterations
            t += dt
            step_index += 1
            dts.append(dt)

            if step_index % cfg.store_every == 0 or t >= cfg.t_final - 1e-14:
                times.append(t)
                snapshots.append(u.copy())
                residuals.append(self.steady_residual(u))

        wall_time = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if step_index >= cfg.max_steps and t < cfg.t_final - 1e-12:
            logger.warning(
                "Reached max_steps=%d before t_final (t=%.6g).", cfg.max_steps, t
            )

        metrics = SolverMetrics(
            wall_time_s=wall_time,
            peak_memory_mb=peak / 1024**2,
            n_steps=step_index,
            total_krylov_iterations=total_iterations,
            mean_krylov_iterations=total_iterations / max(step_index, 1),
        )
        logger.info(
            "GMRES run complete: N=%d, steps=%d, t=%.4f, wall=%.3fs",
            cfg.n_points,
            step_index,
            t,
            wall_time,
        )

        self.solution = BurgersSolution(
            x=self.x.copy(),
            times=np.asarray(times, dtype=np.float64),
            snapshots=np.asarray(snapshots, dtype=np.float64),
            residual_history=np.asarray(residuals, dtype=np.float64),
            dt_history=np.asarray(dts, dtype=np.float64),
            config=cfg,
            metrics=metrics,
        )
        return self.solution

    # -- diagnostics -------------------------------------------------------- #
    def errors_against(
        self, reference: np.ndarray, solution: BurgersSolution | None = None
    ) -> dict[str, float]:
        """Relative L2 and absolute L-infinity error at the final time."""
        sol = solution or self.solution
        if sol is None:
            raise RuntimeError("No solution available; call solve() first.")
        return {
            "relative_l2": relative_l2_error(sol.final, reference),
            "linf": linf_error(sol.final, reference),
        }

    @staticmethod
    def resolution_study(
        resolutions: Sequence[int],
        base_config: BurgersConfig | None = None,
        gmres_config: GMRESConfig | None = None,
    ) -> dict[str, list[float]]:
        """Run the solver across grid resolutions and collect runtime metrics."""
        if not resolutions:
            raise ValueError("resolutions must be non-empty.")
        base = base_config or BurgersConfig()
        record: dict[str, list[float]] = {
            "n_points": [],
            "wall_time_s": [],
            "peak_memory_mb": [],
            "n_steps": [],
            "final_residual": [],
        }
        for n in resolutions:
            cfg = BurgersConfig(**{**asdict(base), "n_points": int(n)})
            solver = GMRESBurgersSolver(cfg, gmres_config)
            sol = solver.solve()
            record["n_points"].append(float(n))
            record["wall_time_s"].append(sol.metrics.wall_time_s)
            record["peak_memory_mb"].append(sol.metrics.peak_memory_mb)
            record["n_steps"].append(float(sol.metrics.n_steps))
            record["final_residual"].append(float(sol.residual_history[-1]))
        return record


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def _finalise(fig: Figure, save_path: str | Path | None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Figure written to %s", path)
    return fig


def plot_solution_evolution(
    solution: BurgersSolution,
    n_curves: int = 6,
    save_path: str | Path | None = None,
) -> Figure:
    """Overlay velocity snapshots and show the space-time field."""
    if n_curves < 2:
        raise ValueError("n_curves must be at least 2.")
    idx = np.unique(
        np.linspace(0, len(solution.times) - 1, n_curves, dtype=int)
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    cmap = plt.get_cmap("viridis")
    for k, i in enumerate(idx):
        axes[0].plot(
            solution.x,
            solution.snapshots[i],
            color=cmap(k / max(len(idx) - 1, 1)),
            label=f"t = {solution.times[i]:.3f}",
        )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, t)")
    axes[0].set_title("Solution evolution (GMRES)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    mesh = axes[1].pcolormesh(
        solution.x,
        solution.times,
        solution.snapshots,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=axes[1], label="u")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Space-time field")
    return _finalise(fig, save_path)


def plot_error_history(
    solution: BurgersSolution,
    reference: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot the spatial residual history and, optionally, error against a reference."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(solution.times, solution.residual_history, label="PDE residual")
    if reference is not None:
        errors = [relative_l2_error(s, reference) for s in solution.snapshots]
        ax.semilogy(solution.times, errors, "--", label="relative $L_2$ error")
    ax.set_xlabel("t")
    ax.set_ylabel("magnitude")
    ax.set_title("Residual / error history")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_convergence(
    resolutions: Sequence[int],
    errors: Sequence[float],
    reference_order: float | None = 2.0,
    save_path: str | Path | None = None,
) -> Figure:
    """Log-log grid-refinement convergence plot with an optional slope guide."""
    resolutions = np.asarray(resolutions, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if resolutions.shape != errors.shape:
        raise ValueError("resolutions and errors must have matching length.")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(resolutions, errors, "o-", label="measured")
    if reference_order is not None and len(resolutions) > 1:
        guide = errors[0] * (resolutions / resolutions[0]) ** (-reference_order)
        ax.loglog(resolutions, guide, "k--", alpha=0.6,
                  label=f"$O(N^{{-{reference_order:g}}})$")
    ax.set_xlabel("grid resolution N")
    ax.set_ylabel("relative $L_2$ error")
    ax.set_title("Grid convergence")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_runtime_scaling(
    study: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> Figure:
    """Runtime and peak-memory scaling versus grid resolution."""
    for key in ("n_points", "wall_time_s", "peak_memory_mb"):
        if key not in study:
            raise KeyError(f"study is missing required key '{key}'.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].loglog(study["n_points"], study["wall_time_s"], "o-")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_title("Runtime scaling")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].loglog(study["n_points"], study["peak_memory_mb"], "s-", color="tab:red")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("peak memory [MB]")
    axes[1].set_title("Memory scaling")
    axes[1].grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_comparison(
    x: np.ndarray,
    fields: dict[str, np.ndarray],
    reference_key: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Compare several solvers' final profiles and their pointwise deviations."""
    if not fields:
        raise ValueError("fields must contain at least one entry.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, field in fields.items():
        if field.shape != x.shape:
            raise ValueError(f"Field '{label}' has shape {field.shape}, expected {x.shape}.")
        axes[0].plot(x, field, label=label)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].set_title("Final profiles")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if reference_key is not None:
        if reference_key not in fields:
            raise KeyError(f"reference_key '{reference_key}' not present in fields.")
        ref = fields[reference_key]
        for label, field in fields.items():
            if label == reference_key:
                continue
            axes[1].semilogy(x, np.abs(field - ref) + 1e-18, label=label)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel(f"|u - u_{{{reference_key}}}|")
        axes[1].set_title("Pointwise deviation")
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")
    axes[1].grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


if __name__ == "__main__":  # pragma: no cover
    cfg = BurgersConfig(n_points=128, viscosity=0.01, t_final=0.1, store_every=5)
    solver = GMRESBurgersSolver(cfg)
    sol = solver.solve()
    print(f"steps={sol.metrics.n_steps}  wall={sol.metrics.wall_time_s:.3f}s")
    print(f"final residual={sol.residual_history[-1]:.3e}")