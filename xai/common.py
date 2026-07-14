"""
xai.common
==========

Shared configuration, path resolution and lightweight data utilities for the
QA-PINN / Classical-PINN Explainable-AI (XAI) suite.

Everything downstream (metrics, expressivity, landscapes, plotting) imports its
paths, physics constants and the pre-computed feature bundle from here so that
the analysis is reproducible regardless of the current working directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
# xai/  lives at the workspace root next to  checkpoints/  and  output/ .
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))


def _first_existing(*candidates: str) -> Optional[str]:
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def resolve_checkpoint(name: str) -> Optional[str]:
    """Locate a checkpoint file.

    The task brief refers to weights "in the root directory"; in this workspace
    they actually live under ``checkpoints/``.  We search both so the suite is
    robust to either layout.
    """
    return _first_existing(
        os.path.join(ROOT, "checkpoints", name),
        os.path.join(ROOT, name),
    )


CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(ROOT, "output", "xai")
DOCS_DIR = os.path.join(ROOT, "docs")

# Canonical checkpoint filenames referenced by the brief.
QAPINN_CHECKPOINTS: Dict[int, str] = {
    3: "qapinn_3qubit.pt",
    4: "qapinn_4qubit.pt",
    5: "qapinn_5qubit.pt",
}
CLASSICAL_CHECKPOINT = "burgers_pinn.pt"
FEATURES_FILE = "qapinn_features.npz"
SWEEP_FILE = "qapinn_sweep_results.json"


def ensure_output_dir() -> str:
    """Create ``output/xai/`` (idempotent) and return its path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# --------------------------------------------------------------------------- #
# Physics / domain constants (Burgers, matching both training notebooks)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Physics:
    """1-D viscous Burgers configuration shared by both models."""

    nu: float = 0.01
    xmin: float = 0.0
    xmax: float = 1.0
    tmin: float = 0.0
    T: float = 1.0

    @property
    def sx(self) -> float:
        """d/dx = sx * d/dx_hat  after mapping x -> [-1, 1]."""
        return 2.0 / (self.xmax - self.xmin)

    @property
    def st(self) -> float:
        """d/dt = st * d/dt_hat  after mapping t -> [-1, 1]."""
        return 2.0 / (self.T - self.tmin)

    def norm_x(self, x):
        return 2.0 * (x - self.xmin) / (self.xmax - self.xmin) - 1.0

    def norm_t(self, t):
        return 2.0 * (t - self.tmin) / (self.T - self.tmin) - 1.0


PHYSICS = Physics()


# --------------------------------------------------------------------------- #
# Pre-computed feature bundle
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBundle:
    """Wrapper around ``qapinn_features.npz`` produced by the QA-PINN notebook.

    Layer-wise intermediate representations of the *best* QA-PINN (3-qubit) on
    the full FEM evaluation grid, plus the reference fields.  All feature arrays
    are ``(N, d)`` with ``N = len(t_grid) * len(x_grid)`` rows in row-major
    (t-outer, x-inner) order.
    """

    x_grid: np.ndarray
    t_grid: np.ndarray
    U_grid: np.ndarray          # FEM reference,  (nt, nx)
    U_qapinn: np.ndarray        # QA-PINN field,  (nt, nx)
    n_qubits: int
    n_layers: int
    encoder_pre: np.ndarray     # (N, n)   raw linear projection
    encoder_angles: np.ndarray  # (N, n)   pi*tanh angles into RY encoding
    quantum_probs: np.ndarray   # (N, 2**n) measured basis probabilities
    decoder_hidden: np.ndarray  # (N, 2**(n-1)) classical hidden state
    output: np.ndarray          # (N, 1)   scalar prediction
    raw: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def layer_stack(self) -> "Dict[str, np.ndarray]":
        """Ordered dict: Classical encoder -> Quantum probs -> Decoder hidden.

        Used for the information-bottleneck (SVD / effective-dimension) study.
        """
        return {
            "encoder_pre": self.encoder_pre,
            "encoder_angles": self.encoder_angles,
            "quantum_probs": self.quantum_probs,
            "decoder_hidden": self.decoder_hidden,
            "output": self.output,
        }


def load_features(path: Optional[str] = None) -> FeatureBundle:
    """Load and validate the QA-PINN feature bundle."""
    path = path or resolve_checkpoint(FEATURES_FILE)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not locate '{FEATURES_FILE}'. Expected under checkpoints/."
        )
    d = np.load(path, allow_pickle=True)
    raw = {k: d[k] for k in d.files}
    return FeatureBundle(
        x_grid=raw["x_grid"].astype(np.float64),
        t_grid=raw["t_grid"].astype(np.float64),
        U_grid=raw["U_grid"].astype(np.float64),
        U_qapinn=raw["U_qapinn"].astype(np.float64),
        n_qubits=int(raw["n_qubits"]),
        n_layers=int(raw["n_layers"]),
        encoder_pre=raw["encoder_pre"].astype(np.float32),
        encoder_angles=raw["encoder_angles"].astype(np.float32),
        quantum_probs=raw["quantum_probs"].astype(np.float32),
        decoder_hidden=raw["decoder_hidden"].astype(np.float32),
        output=raw["output"].astype(np.float32),
        raw=raw,
    )


def load_sweep_results(path: Optional[str] = None) -> dict:
    """Load ``qapinn_sweep_results.json`` (qubit sweep training summary)."""
    path = path or resolve_checkpoint(SWEEP_FILE)
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Misc utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0) -> None:
    """Seed numpy + torch (if available) for reproducibility."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # torch optional at import time
        pass


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars / arrays."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def dump_json(obj, path: str) -> None:
    """Write ``obj`` to ``path`` as pretty JSON (numpy-aware)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=NumpyJSONEncoder)


def flat_grid(x_grid: np.ndarray, t_grid: np.ndarray):
    """Return normalized (x, t) column vectors in the same row-major (t-outer,
    x-inner) order used when the feature bundle was generated."""
    TT, XX = np.meshgrid(t_grid, x_grid, indexing="ij")  # (nt, nx)
    x_col = XX.reshape(-1, 1)
    t_col = TT.reshape(-1, 1)
    return x_col, t_col
