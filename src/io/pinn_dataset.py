"""
src/io/pinn_dataset.py
======================

Export the finite-element solution history as the canonical supervised-learning
table consumed by PINNs / QTN-PINNs / QAPINNs downstream.

The column layout adapts to the spatial dimension while keeping the same
``[spatial..., t, u]`` convention:

    1D:  [x, t, u]
    2D:  [x, y, t, u]
    3D:  [x, y, z, t, u]
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.config.config import BurgersConfig


def generate_pinn_dataset(
    cfg: BurgersConfig,
    coords: np.ndarray,
    t: np.ndarray,
    U: np.ndarray,
    comm,
    log: logging.Logger,
) -> np.ndarray:
    """Flatten the (coords, t) grid and solution into an ``(N, gdim+2)`` table.

    Columns are ``[x, (y, (z,)), t, u]``.

    Parameters
    ----------
    coords : (n_points, gdim) sorted spatial coordinates
    t      : (n_t,) recorded time levels
    U      : (n_t, n_points) solution history

    Returns
    -------
    np.ndarray
        The flattened dataset (also written to disk if configured).
    """
    if cfg.dimension < 1:  # defensive; config validation prevents this
        raise ValueError("dimension must be >= 1.")
    ncols = cfg.dimension + 2
    if comm.rank != 0:
        return np.empty((0, ncols))

    coords = np.atleast_2d(coords)
    if coords.shape[0] != U.shape[1] and coords.shape[1] == U.shape[1]:
        # Tolerate a (gdim, n_points) layout by transposing.
        coords = coords.T
    n_t, n_pts = U.shape

    # For every recorded time level we stack the full set of spatial points.
    # coords is tiled once per time; t is repeated once per point.
    coords_tiled = np.tile(coords, (n_t, 1))            # (n_t*n_pts, gdim)
    t_col = np.repeat(t, n_pts).reshape(-1, 1)          # (n_t*n_pts, 1)
    u_col = U.reshape(-1, 1)                            # (n_t*n_pts, 1)
    data = np.hstack([coords_tiled, t_col, u_col])

    header = ",".join([*cfg.spatial_columns, "t", "u"])

    if cfg.pinn_export_npy:
        npy_path = os.path.join(cfg.output_dir, "numpy",
                                f"{cfg.pinn_dataset_basename}.npy")
        np.save(npy_path, data)
        log.info("PINN dataset (npy) -> %s  shape=%s  cols=[%s]",
                 npy_path, data.shape, header)
    if cfg.pinn_export_csv:
        csv_path = os.path.join(cfg.output_dir, "csv",
                                f"{cfg.pinn_dataset_basename}.csv")
        np.savetxt(csv_path, data, delimiter=",", header=header,
                   comments="", fmt="%.8e")
        log.info("PINN dataset (csv) -> %s", csv_path)
    return data
