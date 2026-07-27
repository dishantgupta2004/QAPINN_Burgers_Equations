"""Matrix Product State (MPS) tensor-network solver for the 1D Burgers equation.

The velocity field on a grid of N = 2^L points is stored as an MPS of L rank-3
tensors. Spatial derivatives are applied as Matrix Product Operators, the
nonlinear advection term is evaluated by a site-wise Hadamard product, and the
state is advanced with explicit RK4 under a CFL-limited timestep. SVD-based
truncation after every update bounds the bond dimension.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from classical_gmres import (
    BurgersConfig,
    InitialConditionFactory,
    cfl_timestep,
    linf_error,
    relative_l2_error,
)

__all__ = [
    "MPSConfig",
    "ConservationRecord",
    "QTNMetrics",
    "QTNSolution",
    "MPS",
    "MPO",
    "build_first_derivative_mpo",
    "build_second_derivative_mpo",
    "QTNBurgersSolver",
    "plot_entanglement_entropy",
    "plot_bond_dimensions",
    "plot_conservation",
    "plot_compression_statistics",
    "plot_solution_evolution",
    "plot_bond_dimension_sweep",
    "plot_runtime_scaling",
    "plot_comparison",
]

logger = logging.getLogger(__name__)
if not logger.handlers:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# Configuration and records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MPSConfig:
    """Tensor-network truncation and adaptivity settings."""

    max_bond_dim: int = 16
    cutoff: float = 1e-12
    adaptive_bond: bool = False
    min_bond_dim: int = 2
    bond_growth_threshold: float = 1e-8
    bond_growth_factor: float = 2.0
    absolute_max_bond_dim: int = 128
    track_entropy: bool = True

    def __post_init__(self) -> None:
        if self.max_bond_dim < 1:
            raise ValueError("max_bond_dim must be >= 1.")
        if self.min_bond_dim < 1 or self.min_bond_dim > self.max_bond_dim:
            raise ValueError("min_bond_dim must lie in [1, max_bond_dim].")
        if self.cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")
        if self.bond_growth_factor <= 1.0:
            raise ValueError("bond_growth_factor must exceed 1.")
        if self.absolute_max_bond_dim < self.max_bond_dim:
            raise ValueError("absolute_max_bond_dim must be >= max_bond_dim.")


@dataclass(slots=True)
class ConservationRecord:
    """Discrete mass, momentum, and energy integrals over time."""

    times: list[float] = field(default_factory=list)
    mass: list[float] = field(default_factory=list)
    momentum: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)

    def record(self, t: float, u: np.ndarray, dx: float) -> None:
        self.times.append(float(t))
        self.mass.append(float(np.sum(u) * dx))
        self.momentum.append(float(np.sum(u**2) * dx))
        self.energy.append(float(0.5 * np.sum(u**2) * dx))

    def as_dict(self) -> dict[str, list[float]]:
        return asdict(self)

    def relative_drift(self) -> dict[str, float]:
        """Relative change of each invariant from the first recorded value."""
        out: dict[str, float] = {}
        for key in ("mass", "momentum", "energy"):
            series = getattr(self, key)
            if not series or abs(series[0]) < 1e-30:
                out[key] = float("nan")
            else:
                out[key] = abs(series[-1] - series[0]) / abs(series[0])
        return out


@dataclass(slots=True)
class QTNMetrics:
    """Runtime, memory, and compression statistics."""

    wall_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    n_steps: int = 0
    n_sites: int = 0
    max_bond_observed: int = 0
    mps_parameters: int = 0
    dense_parameters: int = 0
    compression_ratio: float = 1.0
    total_truncation_error: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class QTNSolution:
    """Time-resolved MPS solution together with diagnostics."""

    x: np.ndarray
    times: np.ndarray
    snapshots: np.ndarray
    entropy_history: np.ndarray       # (n_saved, n_sites - 1)
    bond_history: np.ndarray          # (n_saved, n_sites - 1)
    truncation_history: np.ndarray
    dt_history: np.ndarray
    conservation: ConservationRecord
    config: BurgersConfig
    mps_config: MPSConfig
    metrics: QTNMetrics = field(default_factory=QTNMetrics)

    @property
    def final(self) -> np.ndarray:
        return self.snapshots[-1]

    @property
    def initial(self) -> np.ndarray:
        return self.snapshots[0]

    @property
    def max_entropy_history(self) -> np.ndarray:
        return self.entropy_history.max(axis=1)

    def save(self, path: str | Path) -> Path:
        """Persist arrays, configuration, and metrics to a .npz archive."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            times=self.times,
            snapshots=self.snapshots,
            entropy_history=self.entropy_history,
            bond_history=self.bond_history,
            truncation_history=self.truncation_history,
            dt_history=self.dt_history,
            conservation_json=json.dumps(self.conservation.as_dict()),
            config_json=json.dumps(asdict(self.config)),
            mps_config_json=json.dumps(asdict(self.mps_config)),
            metrics_json=json.dumps(self.metrics.as_dict()),
        )
        logger.info("Saved QTN solution to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QTNSolution":
        """Restore a solution written by :meth:`save`."""
        path = Path(path).with_suffix(".npz")
        if not path.exists():
            raise FileNotFoundError(f"No QTN archive at {path}.")
        with np.load(path, allow_pickle=False) as data:
            return cls(
                x=data["x"],
                times=data["times"],
                snapshots=data["snapshots"],
                entropy_history=data["entropy_history"],
                bond_history=data["bond_history"],
                truncation_history=data["truncation_history"],
                dt_history=data["dt_history"],
                conservation=ConservationRecord(
                    **json.loads(str(data["conservation_json"]))
                ),
                config=BurgersConfig(**json.loads(str(data["config_json"]))),
                mps_config=MPSConfig(**json.loads(str(data["mps_config_json"]))),
                metrics=QTNMetrics(**json.loads(str(data["metrics_json"]))),
            )


# --------------------------------------------------------------------------- #
# Core tensor-network structures
# --------------------------------------------------------------------------- #
def _truncated_svd(
    matrix: np.ndarray, max_bond: int, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SVD with combined rank and singular-value truncation.

    Returns ``(u, s, vh, discarded_weight)`` where the discarded weight is the
    relative sum of squared singular values that were dropped.
    """
    try:
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:  # fall back to a more robust driver
        u, s, vh = np.linalg.svd(matrix, full_matrices=False, compute_uv=True)

    total = float(np.sum(s**2))
    if total <= 0.0:
        return u[:, :1], s[:1], vh[:1, :], 0.0

    keep = int(np.sum(s > cutoff * s[0])) if s[0] > 0 else 1
    keep = max(1, min(keep, max_bond, len(s)))
    discarded = float(np.sum(s[keep:] ** 2) / total)
    return u[:, :keep], s[:keep], vh[:keep, :], discarded


class MPS:
    """Open-boundary matrix product state over ``n_sites`` two-level sites."""

    def __init__(self, tensors: Sequence[np.ndarray]) -> None:
        if not tensors:
            raise ValueError("An MPS requires at least one tensor.")
        for i, tensor in enumerate(tensors):
            if tensor.ndim != 3:
                raise ValueError(f"Tensor {i} must be rank-3, got ndim={tensor.ndim}.")
            if tensor.shape[1] != 2:
                raise ValueError(f"Tensor {i} must have physical dimension 2.")
        for i in range(len(tensors) - 1):
            if tensors[i].shape[2] != tensors[i + 1].shape[0]:
                raise ValueError(f"Bond mismatch between sites {i} and {i + 1}.")
        self.tensors: list[np.ndarray] = [np.asarray(t, dtype=np.float64) for t in tensors]

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_vector(
        cls, vector: np.ndarray, max_bond: int = 64, cutoff: float = 1e-14
    ) -> "MPS":
        """Sequential-SVD decomposition of a length-2^L vector."""
        vector = np.asarray(vector, dtype=np.float64).ravel()
        n = vector.size
        n_sites = int(np.round(np.log2(n)))
        if 2**n_sites != n:
            raise ValueError(f"Vector length {n} is not a power of two.")

        tensors: list[np.ndarray] = []
        residual = vector.reshape(1, n)
        left = 1
        for site in range(n_sites - 1):
            matrix = residual.reshape(left * 2, -1)
            u, s, vh, _ = _truncated_svd(matrix, max_bond, cutoff)
            chi = u.shape[1]
            tensors.append(u.reshape(left, 2, chi))
            residual = (np.diag(s) @ vh).reshape(chi, -1)
            left = chi
        tensors.append(residual.reshape(left, 2, 1))
        return cls(tensors)

    @classmethod
    def zeros_like(cls, other: "MPS") -> "MPS":
        """Zero-valued MPS with unit bond dimensions matching site count."""
        n = other.n_sites
        return cls([np.zeros((1, 2, 1)) for _ in range(n)])

    # -- properties --------------------------------------------------------- #
    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def dim(self) -> int:
        return 2**self.n_sites

    @property
    def bond_dimensions(self) -> np.ndarray:
        return np.array([t.shape[2] for t in self.tensors[:-1]], dtype=int)

    @property
    def max_bond_dimension(self) -> int:
        bonds = self.bond_dimensions
        return int(bonds.max()) if bonds.size else 1

    @property
    def n_parameters(self) -> int:
        return int(sum(t.size for t in self.tensors))

    def copy(self) -> "MPS":
        return MPS([t.copy() for t in self.tensors])

    # -- conversion --------------------------------------------------------- #
    def to_vector(self) -> np.ndarray:
        """Contract the full tensor train back to a dense vector."""
        result = self.tensors[0]
        for tensor in self.tensors[1:]:
            left, phys, _ = result.shape
            result = np.tensordot(result, tensor, axes=([2], [0]))
            result = result.reshape(left, phys * tensor.shape[1], tensor.shape[2])
        return result.reshape(-1)

    # -- canonicalisation and compression ----------------------------------- #
    def right_canonicalise(self) -> None:
        """Bring the state into right-canonical form by a right-to-left QR sweep."""
        for site in range(self.n_sites - 1, 0, -1):
            tensor = self.tensors[site]
            left, phys, right = tensor.shape
            matrix = tensor.reshape(left, phys * right)
            q, r = np.linalg.qr(matrix.T)
            new_left = q.shape[1]
            self.tensors[site] = q.T.reshape(new_left, phys, right)
            prev = self.tensors[site - 1]
            self.tensors[site - 1] = np.tensordot(prev, r.T, axes=([2], [0]))

    def compress(self, max_bond: int, cutoff: float = 1e-12) -> float:
        """Truncate every bond by a left-to-right SVD sweep.

        Returns the accumulated discarded weight.
        """
        if max_bond < 1:
            raise ValueError("max_bond must be >= 1.")
        self.right_canonicalise()
        discarded_total = 0.0
        carry = np.eye(self.tensors[0].shape[0])
        for site in range(self.n_sites - 1):
            tensor = np.tensordot(carry, self.tensors[site], axes=([1], [0]))
            left, phys, right = tensor.shape
            u, s, vh, discarded = _truncated_svd(
                tensor.reshape(left * phys, right), max_bond, cutoff
            )
            discarded_total += discarded
            chi = u.shape[1]
            self.tensors[site] = u.reshape(left, phys, chi)
            carry = np.diag(s) @ vh
        self.tensors[-1] = np.tensordot(carry, self.tensors[-1], axes=([1], [0]))
        return discarded_total

    # -- diagnostics -------------------------------------------------------- #
    def entanglement_spectrum(self) -> list[np.ndarray]:
        """Normalised singular values at every bond."""
        work = self.copy()
        work.right_canonicalise()
        spectra: list[np.ndarray] = []
        carry = np.eye(work.tensors[0].shape[0])
        for site in range(work.n_sites - 1):
            tensor = np.tensordot(carry, work.tensors[site], axes=([1], [0]))
            left, phys, right = tensor.shape
            u, s, vh = np.linalg.svd(tensor.reshape(left * phys, right), full_matrices=False)
            norm = float(np.linalg.norm(s))
            spectra.append(s / norm if norm > 0 else s)
            work.tensors[site] = u.reshape(left, phys, u.shape[1])
            carry = np.diag(s) @ vh
        return spectra

    def entanglement_entropy(self) -> np.ndarray:
        """Von Neumann entropy across each bipartition."""
        entropies = []
        for s in self.entanglement_spectrum():
            p = s**2
            p = p[p > 1e-16]
            entropies.append(float(-np.sum(p * np.log(p))) if p.size else 0.0)
        return np.asarray(entropies, dtype=np.float64)

    # -- algebra ------------------------------------------------------------ #
    def scale(self, factor: float) -> "MPS":
        """Return a scaled copy of the state."""
        out = self.copy()
        out.tensors[0] = out.tensors[0] * float(factor)
        return out

    def add(self, other: "MPS", max_bond: int | None = None, cutoff: float = 1e-12) -> "MPS":
        """Direct-sum addition followed by optional compression."""
        if self.n_sites != other.n_sites:
            raise ValueError("Cannot add MPS with different site counts.")
        n = self.n_sites
        tensors: list[np.ndarray] = []
        for site in range(n):
            a, b = self.tensors[site], other.tensors[site]
            la, _, ra = a.shape
            lb, _, rb = b.shape
            if site == 0:
                block = np.zeros((1, 2, ra + rb))
                block[:, :, :ra] = a
                block[:, :, ra:] = b
            elif site == n - 1:
                block = np.zeros((la + lb, 2, 1))
                block[:la, :, :] = a
                block[la:, :, :] = b
            else:
                block = np.zeros((la + lb, 2, ra + rb))
                block[:la, :, :ra] = a
                block[la:, :, ra:] = b
            tensors.append(block)
        result = MPS(tensors)
        if max_bond is not None:
            result.compress(max_bond, cutoff)
        return result

    def hadamard(
        self, other: "MPS", max_bond: int | None = None, cutoff: float = 1e-12
    ) -> "MPS":
        """Site-wise elementwise product of two states.

        Bond dimensions multiply, so compression is applied when ``max_bond``
        is supplied.
        """
        if self.n_sites != other.n_sites:
            raise ValueError("Cannot multiply MPS with different site counts.")
        tensors: list[np.ndarray] = []
        for a, b in zip(self.tensors, other.tensors, strict=True):
            la, _, ra = a.shape
            lb, _, rb = b.shape
            block = np.empty((la * lb, 2, ra * rb))
            for phys in range(2):
                block[:, phys, :] = np.kron(a[:, phys, :], b[:, phys, :])
            tensors.append(block)
        result = MPS(tensors)
        if max_bond is not None:
            result.compress(max_bond, cutoff)
        return result

    def norm(self) -> float:
        """Euclidean norm of the represented vector."""
        left = np.tensordot(self.tensors[0], self.tensors[0], axes=([0, 1], [0, 1]))
        for tensor in self.tensors[1:]:
            tmp = np.tensordot(left, tensor, axes=([0], [0]))
            left = np.tensordot(tensor, tmp, axes=([0, 1], [0, 1]))
        return float(np.sqrt(max(float(left.ravel()[0]), 0.0)))


class MPO:
    """Matrix product operator acting on a two-level-site MPS."""

    def __init__(self, tensors: Sequence[np.ndarray]) -> None:
        if not tensors:
            raise ValueError("An MPO requires at least one tensor.")
        for i, tensor in enumerate(tensors):
            if tensor.ndim != 4:
                raise ValueError(f"MPO tensor {i} must be rank-4.")
            if tensor.shape[1] != 2 or tensor.shape[2] != 2:
                raise ValueError(f"MPO tensor {i} must have physical dimensions 2x2.")
        self.tensors: list[np.ndarray] = [np.asarray(t, dtype=np.float64) for t in tensors]

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, n_sites: int, cutoff: float = 1e-14
    ) -> "MPO":
        """Sequential-SVD decomposition of a dense 2^L x 2^L operator."""
        matrix = np.asarray(matrix, dtype=np.float64)
        dim = 2**n_sites
        if matrix.shape != (dim, dim):
            raise ValueError(f"Expected shape ({dim}, {dim}), got {matrix.shape}.")

        # Interleave row/column indices site by site: (r0 c0 r1 c1 ...)
        tensor = matrix.reshape([2] * n_sites + [2] * n_sites)
        perm: list[int] = []
        for site in range(n_sites):
            perm.extend([site, n_sites + site])
        tensor = tensor.transpose(perm).reshape(1, -1)

        tensors: list[np.ndarray] = []
        left = 1
        for site in range(n_sites - 1):
            u, s, vh, _ = _truncated_svd(
                tensor.reshape(left * 4, -1), max_bond=4**n_sites, cutoff=cutoff
            )
            chi = u.shape[1]
            tensors.append(u.reshape(left, 2, 2, chi))
            tensor = (np.diag(s) @ vh).reshape(chi, -1)
            left = chi
        tensors.append(tensor.reshape(left, 2, 2, 1))
        return cls(tensors)

    def apply(
        self, state: MPS, max_bond: int | None = None, cutoff: float = 1e-12
    ) -> MPS:
        """Contract the operator onto a state, optionally compressing after."""
        if self.n_sites != state.n_sites:
            raise ValueError("MPO and MPS site counts differ.")
        tensors: list[np.ndarray] = []
        for w, a in zip(self.tensors, state.tensors, strict=True):
            # w: (wl, p_out, p_in, wr), a: (al, p_in, ar)
            contracted = np.tensordot(w, a, axes=([2], [1]))  # (wl,p_out,wr,al,ar)
            contracted = contracted.transpose(0, 3, 1, 2, 4)  # (wl,al,p_out,wr,ar)
            wl, al, phys, wr, ar = contracted.shape
            tensors.append(contracted.reshape(wl * al, phys, wr * ar))
        result = MPS(tensors)
        if max_bond is not None:
            result.compress(max_bond, cutoff)
        return result

    def to_matrix(self) -> np.ndarray:
        """Contract to the dense operator (intended for validation only)."""
        result = self.tensors[0]  # (wl, p_out, p_in, wr)
        for tensor in self.tensors[1:]:
            wl, po, pi, _ = result.shape
            merged = np.tensordot(result, tensor, axes=([3], [0]))
            # merged: (wl, po, pi, p_out', p_in', wr'); group out and in indices
            merged = merged.transpose(0, 1, 3, 2, 4, 5)
            result = merged.reshape(
                wl, po * tensor.shape[1], pi * tensor.shape[2], tensor.shape[3]
            )
        dim = result.shape[1]
        return result.reshape(dim, dim)


def _finite_difference_matrix(
    n: int, dx: float, order: int, bc_left: float, bc_right: float
) -> np.ndarray:
    """Dense derivative operator with Dirichlet rows zeroed."""
    matrix = np.zeros((n, n))
    if order == 1:
        for i in range(1, n - 1):
            matrix[i, i - 1] = -1.0 / (2.0 * dx)
            matrix[i, i + 1] = 1.0 / (2.0 * dx)
    elif order == 2:
        for i in range(1, n - 1):
            matrix[i, i - 1] = 1.0 / dx**2
            matrix[i, i] = -2.0 / dx**2
            matrix[i, i + 1] = 1.0 / dx**2
    else:
        raise ValueError("order must be 1 or 2.")
    return matrix


def build_first_derivative_mpo(n_sites: int, dx: float) -> MPO:
    """Central first-derivative MPO with zeroed Dirichlet boundary rows."""
    n = 2**n_sites
    return MPO.from_matrix(_finite_difference_matrix(n, dx, 1, 0.0, 0.0), n_sites)


def build_second_derivative_mpo(n_sites: int, dx: float) -> MPO:
    """Three-point Laplacian MPO with zeroed Dirichlet boundary rows."""
    n = 2**n_sites
    return MPO.from_matrix(_finite_difference_matrix(n, dx, 2, 0.0, 0.0), n_sites)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class QTNBurgersSolver:
    """RK4 time integration of the Burgers equation in MPS representation."""

    def __init__(
        self,
        config: BurgersConfig | None = None,
        mps_config: MPSConfig | None = None,
    ) -> None:
        self.config = config or BurgersConfig(n_points=64)
        self.mps_config = mps_config or MPSConfig()

        n = self.config.n_points
        self.n_sites = int(np.round(np.log2(n)))
        if 2**self.n_sites != n:
            raise ValueError(
                f"QTN requires n_points to be a power of two; got {n}."
            )
        if self.n_sites < 2:
            raise ValueError("QTN requires at least 4 grid points (2 sites).")

        self.x = self.config.grid()
        self.dx = self.config.dx
        self.d1 = build_first_derivative_mpo(self.n_sites, self.dx)
        self.d2 = build_second_derivative_mpo(self.n_sites, self.dx)
        self._current_bond = self.mps_config.max_bond_dim
        self._truncation_accumulator = 0.0
        self.solution: QTNSolution | None = None

    # -- helpers ------------------------------------------------------------ #
    @property
    def _cutoff(self) -> float:
        return self.mps_config.cutoff

    def _compress(self, state: MPS) -> MPS:
        self._truncation_accumulator += state.compress(self._current_bond, self._cutoff)
        return state

    def _apply_boundary(self, state: MPS) -> MPS:
        """Re-impose Dirichlet values by correcting the dense endpoints.

        The correction is expressed as a rank-1 MPS supported on the two
        boundary indices, then added and recompressed.
        """
        vector = state.to_vector()
        delta = np.zeros_like(vector)
        delta[0] = self.config.bc_left - vector[0]
        delta[-1] = self.config.bc_right - vector[-1]
        if np.allclose(delta, 0.0):
            return state
        correction = MPS.from_vector(delta, self._current_bond, self._cutoff)
        return self._compress(state.add(correction))

    # -- spatial operator --------------------------------------------------- #
    def rhs(self, state: MPS) -> MPS:
        """Evaluate -u u_x + nu u_xx entirely in MPS form."""
        du = self.d1.apply(state, self._current_bond, self._cutoff)
        advection = state.hadamard(du, self._current_bond, self._cutoff)
        diffusion = self.d2.apply(state, self._current_bond, self._cutoff)
        result = advection.scale(-1.0).add(
            diffusion.scale(self.config.viscosity),
            self._current_bond,
            self._cutoff,
        )
        return result

    def rk4_step(self, state: MPS, dt: float) -> MPS:
        """One explicit fourth-order Runge-Kutta update."""
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        chi, cut = self._current_bond, self._cutoff

        k1 = self.rhs(state)
        s2 = self._compress(state.add(k1.scale(dt / 2.0), chi, cut))
        k2 = self.rhs(s2)
        s3 = self._compress(state.add(k2.scale(dt / 2.0), chi, cut))
        k3 = self.rhs(s3)
        s4 = self._compress(state.add(k3.scale(dt), chi, cut))
        k4 = self.rhs(s4)

        increment = (
            k1.add(k2.scale(2.0), chi, cut)
            .add(k3.scale(2.0), chi, cut)
            .add(k4, chi, cut)
            .scale(dt / 6.0)
        )
        updated = self._compress(state.add(increment, chi, cut))
        return self._apply_boundary(updated)

    # -- adaptivity --------------------------------------------------------- #
    def _maybe_grow_bond(self, discarded: float) -> None:
        """Increase the bond cap when the discarded weight exceeds tolerance.

        ``discarded`` is the weight actually removed by truncation during the
        preceding step, since inspecting the post-truncation spectrum would
        report zero tail by construction.
        """
        if not self.mps_config.adaptive_bond:
            return
        if discarded > self.mps_config.bond_growth_threshold:
            new_bond = min(
                int(self._current_bond * self.mps_config.bond_growth_factor),
                self.mps_config.absolute_max_bond_dim,
            )
            if new_bond > self._current_bond:
                logger.debug("Growing bond dimension %d -> %d", self._current_bond, new_bond)
                self._current_bond = new_bond

    # -- driver ------------------------------------------------------------- #
    def solve(self, u0: np.ndarray | None = None) -> QTNSolution:
        """Integrate to ``config.t_final`` and return diagnostics."""
        cfg = self.config
        u = (
            InitialConditionFactory.build(cfg)
            if u0 is None
            else np.asarray(u0, dtype=np.float64).copy()
        )
        if u.shape != (cfg.n_points,):
            raise ValueError(f"u0 must have shape ({cfg.n_points},), got {u.shape}.")

        self._current_bond = self.mps_config.max_bond_dim
        self._truncation_accumulator = 0.0

        state = MPS.from_vector(u, self._current_bond, self._cutoff)
        conservation = ConservationRecord()
        conservation.record(0.0, u, self.dx)

        times = [0.0]
        snapshots = [u.copy()]
        entropies = [state.entanglement_entropy()]
        bonds = [state.bond_dimensions.copy()]
        truncations = [0.0]
        dts: list[float] = []
        max_bond_observed = state.max_bond_dimension

        tracemalloc.start()
        t_start = time.perf_counter()
        t = 0.0
        step_index = 0

        while t < cfg.t_final - 1e-14 and step_index < cfg.max_steps:
            current = state.to_vector()
            if cfg.adaptive_dt or cfg.dt_fixed is None:
                dt = cfl_timestep(current, self.dx, cfg.viscosity, cfg.cfl)
            else:
                dt = cfg.dt_fixed
            dt = min(dt, cfg.t_final - t)
            if not np.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(f"Non-positive timestep at t={t}.")

            before = self._truncation_accumulator
            state = self.rk4_step(state, dt)
            self._maybe_grow_bond(self._truncation_accumulator - before)

            u_new = state.to_vector()
            if not np.all(np.isfinite(u_new)):
                raise RuntimeError(f"MPS solution diverged at step {step_index}.")

            t += dt
            step_index += 1
            dts.append(dt)
            max_bond_observed = max(max_bond_observed, state.max_bond_dimension)

            if step_index % cfg.store_every == 0 or t >= cfg.t_final - 1e-14:
                times.append(t)
                snapshots.append(u_new)
                entropies.append(
                    state.entanglement_entropy()
                    if self.mps_config.track_entropy
                    else np.zeros(self.n_sites - 1)
                )
                bonds.append(state.bond_dimensions.copy())
                truncations.append(self._truncation_accumulator - before)
                conservation.record(t, u_new, self.dx)

        wall = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if step_index >= cfg.max_steps and t < cfg.t_final - 1e-12:
            logger.warning("Reached max_steps=%d before t_final.", cfg.max_steps)

        mps_params = state.n_parameters
        dense_params = cfg.n_points
        metrics = QTNMetrics(
            wall_time_s=wall,
            peak_memory_mb=peak / 1024**2,
            n_steps=step_index,
            n_sites=self.n_sites,
            max_bond_observed=int(max_bond_observed),
            mps_parameters=mps_params,
            dense_parameters=dense_params,
            compression_ratio=dense_params / max(mps_params, 1),
            total_truncation_error=self._truncation_accumulator,
        )
        logger.info(
            "QTN run complete: N=%d, sites=%d, chi_max=%d, steps=%d, wall=%.3fs",
            cfg.n_points,
            self.n_sites,
            max_bond_observed,
            step_index,
            wall,
        )

        self.solution = QTNSolution(
            x=self.x.copy(),
            times=np.asarray(times, dtype=np.float64),
            snapshots=np.asarray(snapshots, dtype=np.float64),
            entropy_history=np.asarray(entropies, dtype=np.float64),
            bond_history=np.asarray(bonds, dtype=int),
            truncation_history=np.asarray(truncations, dtype=np.float64),
            dt_history=np.asarray(dts, dtype=np.float64),
            conservation=conservation,
            config=cfg,
            mps_config=self.mps_config,
            metrics=metrics,
        )
        return self.solution

    # -- diagnostics -------------------------------------------------------- #
    def errors_against(
        self, reference: np.ndarray, solution: QTNSolution | None = None
    ) -> dict[str, float]:
        """Relative L2 and L-infinity error at the final time."""
        sol = solution or self.solution
        if sol is None:
            raise RuntimeError("No solution available; call solve() first.")
        return {
            "relative_l2": relative_l2_error(sol.final, reference),
            "linf": linf_error(sol.final, reference),
        }

    @staticmethod
    def bond_dimension_sweep(
        bond_dims: Sequence[int],
        reference: np.ndarray,
        base_config: BurgersConfig | None = None,
        base_mps_config: MPSConfig | None = None,
    ) -> dict[str, list[float]]:
        """Measure accuracy and cost across a range of bond dimensions."""
        if not bond_dims:
            raise ValueError("bond_dims must be non-empty.")
        cfg = base_config or BurgersConfig(n_points=64)
        base_mps = base_mps_config or MPSConfig()
        record: dict[str, list[float]] = {
            "bond_dim": [], "relative_l2": [], "wall_time_s": [],
            "max_entropy": [], "compression_ratio": [],
        }
        for chi in bond_dims:
            mps_cfg = MPSConfig(**{**asdict(base_mps), "max_bond_dim": int(chi)})
            solver = QTNBurgersSolver(cfg, mps_cfg)
            sol = solver.solve()
            record["bond_dim"].append(float(chi))
            record["relative_l2"].append(relative_l2_error(sol.final, reference))
            record["wall_time_s"].append(sol.metrics.wall_time_s)
            record["max_entropy"].append(float(sol.entropy_history[-1].max()))
            record["compression_ratio"].append(sol.metrics.compression_ratio)
        return record

    @staticmethod
    def resolution_study(
        resolutions: Sequence[int],
        base_config: BurgersConfig | None = None,
        base_mps_config: MPSConfig | None = None,
    ) -> dict[str, list[float]]:
        """Runtime and entropy scaling across grid resolutions."""
        if not resolutions:
            raise ValueError("resolutions must be non-empty.")
        base = base_config or BurgersConfig(n_points=64)
        mps_cfg = base_mps_config or MPSConfig()
        record: dict[str, list[float]] = {
            "n_points": [], "wall_time_s": [], "peak_memory_mb": [],
            "max_bond_observed": [], "max_entropy": [], "compression_ratio": [],
        }
        for n in resolutions:
            cfg = BurgersConfig(**{**asdict(base), "n_points": int(n)})
            sol = QTNBurgersSolver(cfg, mps_cfg).solve()
            record["n_points"].append(float(n))
            record["wall_time_s"].append(sol.metrics.wall_time_s)
            record["peak_memory_mb"].append(sol.metrics.peak_memory_mb)
            record["max_bond_observed"].append(float(sol.metrics.max_bond_observed))
            record["max_entropy"].append(float(sol.entropy_history[-1].max()))
            record["compression_ratio"].append(sol.metrics.compression_ratio)
        return record


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def _finalise(fig: Figure, save_path: str | Path | None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Figure written to %s", p)
    return fig


def plot_entanglement_entropy(
    solution: QTNSolution, save_path: str | Path | None = None
) -> Figure:
    """Maximum entropy over time and the entropy profile across bonds."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(solution.times, solution.max_entropy_history, color="purple")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"max $S_{vN}$")
    axes[0].set_title("Entanglement growth")
    axes[0].grid(alpha=0.3)

    mesh = axes[1].pcolormesh(
        np.arange(1, solution.entropy_history.shape[1] + 1),
        solution.times,
        solution.entropy_history,
        shading="auto",
        cmap="inferno",
    )
    fig.colorbar(mesh, ax=axes[1], label=r"$S_{vN}$")
    axes[1].set_xlabel("bond index")
    axes[1].set_ylabel("t")
    axes[1].set_title("Entropy across bipartitions")
    return _finalise(fig, save_path)


def plot_bond_dimensions(
    solution: QTNSolution, save_path: str | Path | None = None
) -> Figure:
    """Evolution of the maximum and per-bond dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(solution.times, solution.bond_history.max(axis=1), "o-", ms=3)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"max $\chi$")
    axes[0].set_title("Bond dimension evolution")
    axes[0].grid(alpha=0.3)

    for k in range(solution.bond_history.shape[1]):
        axes[1].plot(solution.times, solution.bond_history[:, k], label=f"bond {k + 1}")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\chi$")
    axes[1].set_title("Per-bond dimensions")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    return _finalise(fig, save_path)


def plot_conservation(
    solution: QTNSolution, save_path: str | Path | None = None
) -> Figure:
    """Mass, momentum, and energy integrals over time."""
    record = solution.conservation
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(record.times, record.mass, label="mass")
    ax.plot(record.times, record.momentum, label="momentum")
    ax.plot(record.times, record.energy, label="energy")
    ax.set_xlabel("t")
    ax.set_ylabel("integral value")
    ax.set_title("Conserved quantity monitoring")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finalise(fig, save_path)


def plot_compression_statistics(
    solution: QTNSolution, save_path: str | Path | None = None
) -> Figure:
    """Per-step discarded weight and cumulative truncation error."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogy(
        solution.times, np.maximum(solution.truncation_history, 1e-20), "o-", ms=3
    )
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("discarded weight")
    axes[0].set_title("Truncation per stored step")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy(
        solution.times,
        np.maximum(np.cumsum(solution.truncation_history), 1e-20),
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("cumulative discarded weight")
    axes[1].set_title(
        f"Compression ratio = {solution.metrics.compression_ratio:.3f}"
    )
    axes[1].grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_solution_evolution(
    solution: QTNSolution,
    n_curves: int = 6,
    save_path: str | Path | None = None,
) -> Figure:
    """Snapshot overlay and space-time field of the MPS solution."""
    if n_curves < 2:
        raise ValueError("n_curves must be at least 2.")
    idx = np.unique(np.linspace(0, len(solution.times) - 1, n_curves, dtype=int))
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
    axes[0].set_ylabel("u")
    axes[0].set_title("Solution evolution (QTN/MPS)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    mesh = axes[1].pcolormesh(
        solution.x, solution.times, solution.snapshots, shading="auto", cmap="viridis"
    )
    fig.colorbar(mesh, ax=axes[1], label="u")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Space-time field")
    return _finalise(fig, save_path)


def plot_bond_dimension_sweep(
    sweep: dict[str, list[float]], save_path: str | Path | None = None
) -> Figure:
    """Accuracy and runtime versus the maximum bond dimension."""
    for key in ("bond_dim", "relative_l2", "wall_time_s"):
        if key not in sweep:
            raise KeyError(f"sweep is missing required key '{key}'.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].loglog(sweep["bond_dim"], np.maximum(sweep["relative_l2"], 1e-18), "D-",
                   color="tab:green")
    axes[0].set_xlabel(r"max bond dimension $\chi$")
    axes[0].set_ylabel("relative $L_2$ error")
    axes[0].set_title("Accuracy vs bond dimension")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].loglog(sweep["bond_dim"], sweep["wall_time_s"], "o-", color="tab:blue")
    axes[1].set_xlabel(r"$\chi$")
    axes[1].set_ylabel("wall time [s]")
    axes[1].set_title("Cost vs bond dimension")
    axes[1].grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_runtime_scaling(
    study: dict[str, list[float]], save_path: str | Path | None = None
) -> Figure:
    """Runtime and observed bond dimension against grid resolution."""
    for key in ("n_points", "wall_time_s"):
        if key not in study:
            raise KeyError(f"study is missing required key '{key}'.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].loglog(study["n_points"], study["wall_time_s"], "o-")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_title("QTN runtime scaling")
    axes[0].grid(alpha=0.3, which="both")

    if "max_bond_observed" in study:
        axes[1].semilogx(study["n_points"], study["max_bond_observed"], "s-",
                         color="tab:red")
        axes[1].set_ylabel(r"observed max $\chi$")
    else:
        axes[1].semilogx(study["n_points"], study.get("max_entropy", []), "s-")
        axes[1].set_ylabel(r"max $S_{vN}$")
    axes[1].set_xlabel("N")
    axes[1].set_title("Resource growth")
    axes[1].grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_comparison(
    x: np.ndarray,
    fields: dict[str, np.ndarray],
    reference_key: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Compare final profiles across solvers with pointwise deviations."""
    if not fields:
        raise ValueError("fields must contain at least one entry.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, values in fields.items():
        if values.shape != x.shape:
            raise ValueError(
                f"Field '{label}' has shape {values.shape}, expected {x.shape}."
            )
        axes[0].plot(x, values, label=label)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].set_title("Final profiles")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if reference_key is not None:
        if reference_key not in fields:
            raise KeyError(f"reference_key '{reference_key}' not present in fields.")
        ref = fields[reference_key]
        for label, values in fields.items():
            if label == reference_key:
                continue
            axes[1].semilogy(x, np.abs(values - ref) + 1e-18, label=label)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel(f"|u - u_{{{reference_key}}}|")
        axes[1].set_title("Pointwise deviation")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which="both")
    else:
        axes[1].axis("off")
    return _finalise(fig, save_path)


if __name__ == "__main__":  # pragma: no cover
    cfg = BurgersConfig(
        n_points=64, viscosity=0.01, t_final=0.1,
        initial_condition="smooth_step", store_every=5,
    )
    solver = QTNBurgersSolver(cfg, MPSConfig(max_bond_dim=16))
    sol = solver.solve()
    print(f"steps={sol.metrics.n_steps}  wall={sol.metrics.wall_time_s:.3f}s")
    print(f"max chi={sol.metrics.max_bond_observed}  "
          f"compression={sol.metrics.compression_ratio:.3f}")
    print("conservation drift:", sol.conservation.relative_drift())