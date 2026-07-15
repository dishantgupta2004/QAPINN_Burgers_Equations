from __future__ import annotations
import logging
import os
import numpy as np
from src.config.config import BurgersConfig


def generate_pinn_dataset(cfg: BurgersConfig, coords: np.ndarray, t: np.ndarray, U: np.ndarray, comm, log: logging.Logger,) -> np.ndarray:
    if cfg.dimension < 1:  
        raise ValueError("dimension must be >= 1.")
    ncols = cfg.dimension + 2
    if comm.rank != 0:
        return np.empty((0, ncols))

    coords = np.atleast_2d(coords)
    if coords.shape[0] != U.shape[1] and coords.shape[1] == U.shape[1]:
        coords = coords.T
    n_t, n_pts = U.shape

    coords_tiled = np.tile(coords, (n_t, 1))           
    t_col = np.repeat(t, n_pts).reshape(-1, 1)        
    u_col = U.reshape(-1, 1)                        
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
