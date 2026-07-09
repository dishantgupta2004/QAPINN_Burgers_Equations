"""
src/io/writers.py
=================

Result persistence in several formats (numpy / csv / hdf5 / vtk / diagnostics),
for 1D, 2D and 3D solutions.

These are free functions that take plain arrays + the config, so output can be
produced and unit-tested without instantiating a full solver. All writers are
rank-0-only (serial-accurate gather semantics), matching the solver's snapshot
recording strategy.

Coordinate convention
----------------------
``coords`` is an ``(n_points, gdim)`` array (a single column in 1D). The NumPy
archive stores it under ``coords``; for 1D it additionally stores the flat
``x`` vector for backward compatibility with existing 1D consumers. The CSV
writer uses a *wide* table in 1D (one column per x) and a *tidy/long* table in
2D/3D (one row per (point, time)).
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List

import numpy as np

from src.config.config import BurgersConfig
from src.fenics_backend import io as fio


def write_outputs(
    cfg: BurgersConfig,
    coords: np.ndarray,
    t: np.ndarray,
    U: np.ndarray,
    diagnostics: List[Dict[str, float]],
    u,
    comm,
    log: logging.Logger,
) -> None:
    """Dispatch to the individual writers based on config flags (rank 0 only)."""
    if comm.rank != 0:
        return
    coords = np.atleast_2d(coords)
    if coords.shape[0] != U.shape[1] and coords.shape[1] == U.shape[1]:
        coords = coords.T
    if cfg.save_numpy:
        write_numpy(cfg, coords, t, U, log)
    if cfg.save_csv:
        write_csv(cfg, coords, t, U, log)
    if cfg.save_hdf5:
        write_hdf5(cfg, coords, t, U, log)
    if cfg.save_vtk:
        write_vtk(cfg, u, t, comm, log)
    if cfg.compute_diagnostics:
        write_diagnostics_csv(cfg, diagnostics, log)


def write_numpy(cfg: BurgersConfig, coords: np.ndarray, t: np.ndarray,
                U: np.ndarray, log: logging.Logger) -> None:
    path = os.path.join(cfg.output_dir, "numpy", f"{cfg.experiment_name}.npz")
    payload = {"coords": coords, "t": t, "u": U}
    if cfg.dimension == 1:
        # Backward-compatible flat x vector for existing 1D consumers.
        payload["x"] = coords[:, 0]
    np.savez_compressed(path, **payload)
    log.info("Saved NumPy archive -> %s  (coords %s, u %s)",
             path, coords.shape, U.shape)


def _long_table(cfg: BurgersConfig, coords: np.ndarray, t: np.ndarray,
                U: np.ndarray):
    """Build a tidy ``[x, (y, (z,)), t, u]`` table and its header."""
    n_t, n_pts = U.shape
    coords_tiled = np.tile(coords, (n_t, 1))
    t_col = np.repeat(t, n_pts).reshape(-1, 1)
    u_col = U.reshape(-1, 1)
    data = np.hstack([coords_tiled, t_col, u_col])
    header = [*cfg.spatial_columns, "t", "u"]
    return data, header


def write_csv(cfg: BurgersConfig, coords: np.ndarray, t: np.ndarray,
              U: np.ndarray, log: logging.Logger) -> None:
    path = os.path.join(cfg.output_dir, "csv", f"{cfg.experiment_name}.csv")
    if cfg.dimension == 1:
        # Wide CSV: first column time, remaining columns u at each x.
        x = coords[:, 0]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + [f"x={xi:.6g}" for xi in x])
            for i, ti in enumerate(t):
                w.writerow([f"{ti:.6g}"] + [f"{val:.8e}" for val in U[i]])
    else:
        # Tidy/long CSV: one row per (point, time). Wide form is impractical for
        # scattered 2D/3D points.
        data, header = _long_table(cfg, coords, t, U)
        np.savetxt(path, data, delimiter=",", header=",".join(header),
                   comments="", fmt="%.8e")
    log.info("Saved CSV -> %s", path)


def write_hdf5(cfg: BurgersConfig, coords: np.ndarray, t: np.ndarray,
               U: np.ndarray, log: logging.Logger) -> None:
    try:
        import h5py
    except ImportError as exc:
        log.warning("h5py not available (%s); skipping HDF5.", exc)
        return
    path = os.path.join(cfg.output_dir, "hdf5", f"{cfg.experiment_name}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("coords", data=coords)
        if cfg.dimension == 1:
            f.create_dataset("x", data=coords[:, 0])
        f.create_dataset("t", data=t)
        f.create_dataset("u", data=U)
        f.attrs["config"] = cfg.to_json()
    log.info("Saved HDF5 -> %s", path)


def write_vtk(cfg: BurgersConfig, u, times: np.ndarray, comm,
              log: logging.Logger) -> None:
    path = os.path.join(cfg.output_dir, "xdmf", f"{cfg.experiment_name}.pvd")
    last_t = float(times[-1]) if len(times) else 0.0
    with fio.VTKFile(comm, path, "w") as vtk:
        vtk.write_function(u, last_t)
    log.info("Saved VTK -> %s", path)


def write_diagnostics_csv(cfg: BurgersConfig, diagnostics: List[Dict[str, float]],
                          log: logging.Logger) -> None:
    if not diagnostics:
        return
    path = os.path.join(cfg.output_dir, "csv",
                        f"{cfg.experiment_name}_diagnostics.csv")
    keys = list(diagnostics[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(diagnostics)
    log.info("Saved diagnostics -> %s", path)
