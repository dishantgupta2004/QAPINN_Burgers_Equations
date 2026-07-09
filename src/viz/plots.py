"""
src/viz/plots.py
================

Visualization helpers, dimension-aware (1D lines, 2D filled contours, 3D
delegated to ParaView via the XDMF output).

Matplotlib is imported lazily inside each function so the solver can run in
headless/HPC batch environments without a display backend. Each function takes
plain arrays (``coords`` of shape ``(n_points, gdim)``, ``times``, and the
solution matrix ``U`` of shape ``(n_times, n_points)``) plus the config, so
plotting is decoupled from the solver's internal state.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.config.config import BurgersConfig


def _finish_plot(cfg: BurgersConfig, fig, filename: str, show: bool, save: bool,
                 log: logging.Logger) -> None:
    """Shared helper: save to output/figures and/or show, then close."""
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if save:
        path = os.path.join(cfg.output_dir, "figures", filename)
        fig.savefig(path, dpi=150)
        log.info("Saved figure -> %s", path)
    if show:
        plt.show()
    plt.close(fig)


def _coords2d(coords: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Normalize coords to an ``(n_points, gdim)`` array."""
    coords = np.atleast_2d(coords)
    if coords.shape[0] != U.shape[1] and coords.shape[1] == U.shape[1]:
        coords = coords.T
    return coords


def _warn_3d(cfg: BurgersConfig, log: logging.Logger, what: str) -> bool:
    """3D fields are not rendered inline; point the user at ParaView."""
    if cfg.dimension >= 3:
        log.warning(
            "%s: 3D fields are not plotted inline. Inspect the XDMF output "
            "(output/xdmf/%s.xdmf) in ParaView instead.",
            what, cfg.experiment_name,
        )
        return True
    return False


def plot_solution(cfg: BurgersConfig, coords: np.ndarray, times: np.ndarray,
                  U: np.ndarray, log: logging.Logger,
                  time_index: int = -1, show: bool = False, save: bool = True) -> None:
    """Plot u at a single recorded time level (1D line, 2D filled contour)."""
    import matplotlib.pyplot as plt
    if U.size == 0:
        log.warning("No snapshots to plot.")
        return
    if _warn_3d(cfg, log, "plot_solution"):
        return
    coords = _coords2d(coords, U)
    ti = times[time_index]

    if cfg.dimension == 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(coords[:, 0], U[time_index], lw=2)
        ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.set_title(f"Burgers solution at t = {ti:.4f}")
        ax.grid(True, alpha=0.3)
    else:  # 2D
        fig, ax = plt.subplots(figsize=(6, 5))
        tcf = ax.tricontourf(coords[:, 0], coords[:, 1], U[time_index],
                             levels=40, cmap="viridis")
        fig.colorbar(tcf, ax=ax, label="u")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Burgers solution at t = {ti:.4f}")
    _finish_plot(cfg, fig, "solution.png", show, save, log)


def plot_snapshots(cfg: BurgersConfig, coords: np.ndarray, times: np.ndarray,
                   U: np.ndarray, log: logging.Logger,
                   n_curves: int = 6, show: bool = False, save: bool = True) -> None:
    """Visualize temporal evolution (1D overlaid lines, 2D contour panels)."""
    import matplotlib.pyplot as plt
    if U.size == 0:
        log.warning("No snapshots to plot.")
        return
    if _warn_3d(cfg, log, "plot_snapshots"):
        return
    coords = _coords2d(coords, U)
    idx = np.linspace(0, len(times) - 1, n_curves, dtype=int)

    if cfg.dimension == 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in idx:
            ax.plot(coords[:, 0], U[i], label=f"t={times[i]:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.set_title("Burgers solution snapshots")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:  # 2D — a row of filled-contour panels sharing a color scale
        vmin, vmax = float(U.min()), float(U.max())
        n = len(idx)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
        tcf = None
        for ax, i in zip(axes[0], idx):
            tcf = ax.tricontourf(coords[:, 0], coords[:, 1], U[i], levels=40,
                                 cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"t={times[i]:.3f}", fontsize=9)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
        if tcf is not None:
            fig.colorbar(tcf, ax=axes[0].tolist(), label="u", shrink=0.8)
    _finish_plot(cfg, fig, "snapshots.png", show, save, log)


def plot_heatmap(cfg: BurgersConfig, coords: np.ndarray, times: np.ndarray,
                 U: np.ndarray, log: logging.Logger,
                 show: bool = False, save: bool = True) -> None:
    """1D: space-time heatmap u(x, t). 2D: final-time field. 3D: skipped."""
    import matplotlib.pyplot as plt
    if U.size == 0:
        log.warning("No snapshots to plot.")
        return
    if _warn_3d(cfg, log, "plot_heatmap"):
        return
    coords = _coords2d(coords, U)

    if cfg.dimension == 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.pcolormesh(coords[:, 0], times, U, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label="u")
        ax.set_xlabel("x"); ax.set_ylabel("t")
        ax.set_title("Space-time evolution")
    else:  # 2D — a genuine space-time heatmap is 3D; show the final field.
        fig, ax = plt.subplots(figsize=(6, 5))
        tcf = ax.tricontourf(coords[:, 0], coords[:, 1], U[-1], levels=40,
                             cmap="viridis")
        fig.colorbar(tcf, ax=ax, label="u")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Final field at t = {times[-1]:.4f}")
    _finish_plot(cfg, fig, "heatmap.png", show, save, log)


def plot_animation(cfg: BurgersConfig, coords: np.ndarray, times: np.ndarray,
                   U: np.ndarray, log: logging.Logger,
                   save: bool = True, fps: int = 20) -> None:
    """Animate u over time (1D line, 2D filled contour). 3D is skipped."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    if U.size == 0:
        log.warning("No snapshots to animate.")
        return
    if _warn_3d(cfg, log, "plot_animation"):
        return
    coords = _coords2d(coords, U)

    if cfg.dimension == 1:
        x = coords[:, 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        line, = ax.plot(x, U[0], lw=2)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(float(U.min()) * 1.1 - 0.05, float(U.max()) * 1.1 + 0.05)
        ax.set_xlabel("x"); ax.set_ylabel("u")
        title = ax.set_title("")

        def update(frame: int):
            line.set_ydata(U[frame])
            title.set_text(f"t = {times[frame]:.4f}")
            return line, title
    else:  # 2D
        vmin, vmax = float(U.min()), float(U.max())
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        title = ax.set_title("")

        def update(frame: int):
            ax.clear()
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"t = {times[frame]:.4f}")
            ax.tricontourf(coords[:, 0], coords[:, 1], U[frame], levels=40,
                           cmap="viridis", vmin=vmin, vmax=vmax)
            return ()

    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    if save:
        path = os.path.join(cfg.output_dir, "animations",
                            f"{cfg.experiment_name}.gif")
        try:
            anim.save(path, fps=fps)
            log.info("Saved animation -> %s", path)
        except Exception as exc:  # writer may be unavailable
            log.warning("Could not save animation (%s).", exc)
    plt.close(fig)
