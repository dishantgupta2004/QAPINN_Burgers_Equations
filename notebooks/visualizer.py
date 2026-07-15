
from __future__ import annotations
import os
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import burgers_common as bc


CKPT_DIR = "checkpoints"

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_fig(fig, name: str, ckpt_dir: str = CKPT_DIR, dpi: int = 150) -> str:
    _ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")
    return path


def exact_field(
    nx: int = 256,
    nt: int = 100,
    nu: float = bc.NU,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (and return) the analytic u(x,t) on an nx-by-nt grid.

    Returns
    -------
    x_grid : (nx,)
    t_grid : (nt,)
    U      : (nt, nx)   analytic solution, row j == time t_grid[j]
    """
    x_grid = np.linspace(bc.X_MIN, bc.X_MAX, nx)
    t_grid = np.linspace(bc.T_MIN, bc.T_MAX, nt)
    U = np.empty((nt, nx), dtype=np.float64)
    for j, t in enumerate(t_grid):
        U[j] = bc.burgers_exact_grid(x_grid, t, nu)
    return x_grid, t_grid, U


def _model_field(
    model,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate a trained model on the (t, x) grid -> U_pred of shape (nt, nx)."""
    model.eval()
    U = np.empty((len(t_grid), len(x_grid)), dtype=np.float64)
    with torch.no_grad():
        for j, t in enumerate(t_grid):
            xt = torch.tensor(
                np.stack([x_grid, np.full_like(x_grid, t)], axis=1),
                dtype=torch.float32, device=device,
            )
            U[j] = model(xt).cpu().numpy().ravel()
    return U

# 1. Exact-solution heatmap
def plot_exact_solution(
    nx: int = 256,
    nt: int = 100,
    nu: float = bc.NU,
    ckpt_dir: str = CKPT_DIR,
    field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> str:
    """Heatmap of the analytic Burgers solution over (x, t)."""
    x_grid, t_grid, U = field if field is not None else exact_field(nx, nt, nu)

    fig, ax = plt.subplots(figsize=(9, 5))
    pc = ax.pcolormesh(x_grid, t_grid, U, shading="auto", cmap="RdBu_r",
                       vmin=-1, vmax=1)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(fr"Exact Burgers solution $u(x,t)$  ($\nu={nu:.4g}$)")
    fig.colorbar(pc, ax=ax, label="u")
    return _save_fig(fig, "exact_solution_heatmap.png", ckpt_dir)


# 2. Exact-solution snapshots
def plot_exact_snapshots(
    t_values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.99),
    nx: int = 256,
    nu: float = bc.NU,
    ckpt_dir: str = CKPT_DIR,
) -> str:
    """Overlay analytic u(x,·) at several fixed times (shows shock steepening)."""
    x_grid = np.linspace(bc.X_MIN, bc.X_MAX, nx)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(t_values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for c, t in zip(cmap, t_values):
        u = bc.burgers_exact_grid(x_grid, t, nu)
        ax.plot(x_grid, u, color=c, lw=2, label=f"t = {t:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Exact Burgers solution: shock steepening over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, "exact_snapshots.png", ckpt_dir)


# 3. Prediction vs exact snapshots
def plot_prediction_vs_exact(
    model,
    t_values: Sequence[float] = (0.25, 0.5, 0.75, 0.99),
    nx: int = 256,
    device: str = "cpu",
    title: str = "QA-PINN prediction vs exact",
    fname: str = "prediction_vs_exact.png",
    ckpt_dir: str = CKPT_DIR,
) -> str:
    """One panel per time: model (dotted) vs truth (solid) with rel-L2 annotated.

    Reuses bc.evaluate_on_grid so the accuracy metric matches training exactly.
    """
    res = bc.evaluate_on_grid(model, list(t_values), nx=nx, device=device)
    n = len(t_values)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, t_values):
        x_grid, u_pred, u_exact, l2 = res[t]
        ax.plot(x_grid, u_exact, "b-", lw=2, label="exact")
        ax.plot(x_grid, u_pred, "r:", lw=2, label="prediction")
        ax.set_title(f"t = {t:g}")
        ax.set_xlabel("x")
        ax.text(0.03, 0.05, f"rel-L2={l2:.2e}", transform=ax.transAxes, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, fname, ckpt_dir)


# 4. Animation: exact solution evolving in time
def animate_exact(
    nx: int = 256,
    nt: int = 100,
    nu: float = bc.NU,
    fps: int = 20,
    ckpt_dir: str = CKPT_DIR,
    fname: str = "exact_solution.gif",
    field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> str:
    """Animate the analytic u(x,t) as a travelling / steepening front."""
    x_grid, t_grid, U = field if field is not None else exact_field(nx, nt, nu)

    fig, ax = plt.subplots(figsize=(8, 5))
    (line,) = ax.plot([], [], "b-", lw=2)
    ax.set_xlim(bc.X_MIN, bc.X_MAX)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")

    def _init():
        line.set_data([], [])
        return line, title

    def _update(j):
        line.set_data(x_grid, U[j])
        title.set_text(fr"Exact Burgers $u(x,t)$   t = {t_grid[j]:.3f}")
        return line, title

    anim = animation.FuncAnimation(
        fig, _update, init_func=_init, frames=len(t_grid), blit=True
    )
    _ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, fname)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved -> {path}")
    return path


# 5. Animation: prediction vs exact evolving in time
def animate_prediction(
    model,
    nx: int = 256,
    nt: int = 100,
    nu: float = bc.NU,
    fps: int = 20,
    device: str = "cpu",
    ckpt_dir: str = CKPT_DIR,
    fname: str = "prediction_vs_exact.gif",
    field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> str:
    """Animate model prediction (dotted) against the analytic truth (solid).

    The exact field is cached (passed via `field` or computed once); only the
    model is re-evaluated per frame — but that is cheap and batched.
    """
    x_grid, t_grid, U_exact = field if field is not None else exact_field(nx, nt, nu)
    U_pred = _model_field(model, x_grid, t_grid, device=device)

    fig, ax = plt.subplots(figsize=(8, 5))
    (line_e,) = ax.plot([], [], "b-", lw=2, label="exact")
    (line_p,) = ax.plot([], [], "r:", lw=2, label="prediction")
    ax.set_xlim(bc.X_MIN, bc.X_MAX)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    title = ax.set_title("")

    def _init():
        line_e.set_data([], [])
        line_p.set_data([], [])
        return line_e, line_p, title

    def _update(j):
        line_e.set_data(x_grid, U_exact[j])
        line_p.set_data(x_grid, U_pred[j])
        l2 = bc.relative_l2(U_pred[j], U_exact[j])
        title.set_text(f"t = {t_grid[j]:.3f}    rel-L2 = {l2:.2e}")
        return line_e, line_p, title

    anim = animation.FuncAnimation(
        fig, _update, init_func=_init, frames=len(t_grid), blit=True
    )
    _ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, fname)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved -> {path}")
    return path


# Convenience driver
def render_all_exact(nx: int = 256, nt: int = 100, ckpt_dir: str = CKPT_DIR) -> Dict[str, str]:
    """Compute the exact field ONCE and render every truth-only artifact."""
    field = exact_field(nx, nt)
    return {
        "heatmap": plot_exact_solution(ckpt_dir=ckpt_dir, field=field),
        "snapshots": plot_exact_snapshots(ckpt_dir=ckpt_dir),
        "animation": animate_exact(ckpt_dir=ckpt_dir, field=field),
    }