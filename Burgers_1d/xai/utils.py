"""
utils.py — shared helpers for the XAI module (no PDE-specific logic).
"""
from __future__ import annotations
from typing import Optional, Sequence, Callable
import os, json, time
import numpy as np
import torch

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def savefig(fig, name: str, outdir: str = "outputs/xai", dpi: int = 160) -> str:
    ensure_dir(outdir)
    p = os.path.join(outdir, f"{name}.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    return p


class NpEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays and complex numbers."""
    def default(self, o):
        if isinstance(o, (np.integer,)):   return int(o)
        if isinstance(o, (np.floating,)):  return float(o)
        if isinstance(o, (np.bool_,)):     return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist() if o.size < 4096 else {"__ndarray_shape__": list(o.shape)}
        if isinstance(o, complex):         return {"re": o.real, "im": o.imag}
        return super().default(o)


def dump_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)
    return path

def sample_domain(bounds: Sequence[tuple], n: int, seed: int = 0,
                  device: str = "cpu", grid: bool = False) -> torch.Tensor:
    """
    Sample n points inside an axis-aligned box.

    bounds : sequence of (lo, hi) per input dimension, e.g. [(-1,1),(0,1)]
    grid   : if True, return a (near-)uniform meshgrid with ~n points instead
             of random samples (useful for spectra / heatmaps).
    """
    d = len(bounds)
    if grid:
        per = max(2, int(round(n ** (1.0 / d))))
        axes = [np.linspace(lo, hi, per) for (lo, hi) in bounds]
        mesh = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=1)
    else:
        g = np.random.default_rng(seed)
        pts = np.stack([g.uniform(lo, hi, size=n) for (lo, hi) in bounds], axis=1)
    return torch.as_tensor(pts, dtype=torch.float32, device=device)


def line_samples(bounds: Sequence[tuple], axis: int, n: int,
                 fixed: Optional[dict] = None, device: str = "cpu") -> torch.Tensor:
    """
    Points along a single axis (others held fixed at the mid-point unless given).
    Used for 1-D Fourier spectra of the field.
    """
    d = len(bounds)
    fixed = fixed or {}
    cols = []
    line = np.linspace(bounds[axis][0], bounds[axis][1], n)
    for k in range(d):
        if k == axis:
            cols.append(line)
        else:
            v = fixed.get(k, 0.5 * (bounds[k][0] + bounds[k][1]))
            cols.append(np.full(n, v))
    pts = np.stack(cols, axis=1)
    return torch.as_tensor(pts, dtype=torch.float32, device=device)


def density_from_state(psi: np.ndarray) -> np.ndarray:
    """rho = |psi><psi| for a single pure state vector."""
    psi = np.asarray(psi).ravel()
    return np.outer(psi, psi.conj())


def partial_trace_keep(rho: np.ndarray, keep: int, n_qubits: int) -> np.ndarray:
    """
    Reduced density matrix on a single qubit `keep`, tracing out the rest.
    rho: (2^n, 2^n). Returns (2,2).
    """
    dims = [2] * n_qubits
    rho_t = rho.reshape(dims + dims)
    trace_axes = [i for i in range(n_qubits) if i != keep]
    # trace out each unwanted qubit (bra index = ket index + n_qubits)
    for off, ax in enumerate(sorted(trace_axes)):
        a = ax - off
        b = ax + n_qubits - 2 * off
        rho_t = np.trace(rho_t, axis1=a, axis2=b)
    return rho_t.reshape(2, 2)


def meyer_wallach_Q(states: np.ndarray, n_qubits: int) -> float:
    """
    Meyer-Wallach global entanglement measure, averaged over a batch of pure states.

    Q = (2/n) * sum_k (1 - Tr[rho_k^2]),  rho_k = single-qubit reduced state.
    Q in [0, 1]; 0 = product state, 1 = maximally entangled (per this measure).

    states : (B, 2^n) complex array (a batch), or (2^n,) for a single state.
    """
    states = np.atleast_2d(states)
    B = states.shape[0]
    acc = 0.0
    for b in range(B):
        rho = density_from_state(states[b])
        s = 0.0
        for k in range(n_qubits):
            rk = partial_trace_keep(rho, k, n_qubits)
            s += (1.0 - np.real(np.trace(rk @ rk)))
        acc += (2.0 / n_qubits) * s
    return float(acc / B)


def concurrence_pair(rho2: np.ndarray) -> float:
    """
    Wootters concurrence for a 2-qubit reduced density matrix (4x4).
    Returns 0 for separable, up to 1 for maximally entangled.
    """
    Y = np.array([[0, -1j], [1j, 0]])
    YY = np.kron(Y, Y)
    rho_tilde = YY @ rho2.conj() @ YY
    R = rho2 @ rho_tilde
    ev = np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(R)), 0, None)))[::-1]
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def vn_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """Von Neumann entropy S(rho) = -Tr[rho log rho]."""
    ev = np.clip(np.real(np.linalg.eigvalsh(rho)), 1e-12, None)
    ev = ev / ev.sum()
    return float(-np.sum(ev * (np.log(ev) / np.log(base))))


def shannon_entropy(p: np.ndarray, base: float = 2.0) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, None)
    p = p / p.sum()
    return float(-np.sum(p * (np.log(p) / np.log(base))))


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, None); p /= p.sum()
    q = np.clip(np.asarray(q, float), 1e-12, None); q /= q.sum()
    return float(np.sum(p * (np.log(p / q) / np.log(base))))


def input_gradient(adapter, X: torch.Tensor) -> np.ndarray:
    """
    d(output)/d(input) via autograd, shape (N, d_in). Used for input-sensitivity
    / saliency of the field w.r.t. each collocation coordinate.
    """
    X = X.clone().detach().requires_grad_(True)
    u = adapter(X)
    if u.ndim > 1 and u.shape[-1] > 1:
        u = u.sum(-1, keepdim=True)   # scalarise vector fields for saliency
    g = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=False)[0]
    return g.detach().cpu().numpy()


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
