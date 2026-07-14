r"""
xai.quantum_metrics
===================

Structural and state-space characterisation of the QA-PINN's quantum core.

Implemented diagnostics
-----------------------
1. **Circuit structural specs** -- exact gate counts, circuit depth and
   two-qubit (CNOT) count via ``qml.specs`` / resource tracking.

2. **Entanglement of the encoded state** over the training domain:
     * **Meyer-Wallach** global-entanglement measure

         .. math::  Q(|\psi\rangle) = 2\Bigl(1 - \tfrac1n \sum_{k=1}^{n}
                    \operatorname{Tr}\rho_k^2\Bigr),

       where :math:`\rho_k` is the single-qubit reduced density matrix
       (:math:`Q=0` product state, :math:`Q\to1` maximally entangled).
     * **Bipartite von-Neumann entropy** averaged over all contiguous cuts

         .. math::  S(\rho_A) = -\operatorname{Tr}\,\rho_A \log_2 \rho_A .

3. **Entanglement dynamics** -- how mean entanglement responds as the classical
   encoder is scaled from the (near-product) initialisation toward the trained
   operating point.  Because per-epoch encoder snapshots were not persisted, we
   trace a *homotopy* :math:`W(\alpha)=\alpha\,W^\star` from product-like
   (:math:`\alpha\!\to\!0`) to trained (:math:`\alpha=1`), which is the honest,
   reproducible proxy for the optimisation-driven entanglement growth.

4. **Effective dimension / SVD** of the layer-wise representations
   (Classical encoder -> Quantum probs -> Decoder hidden) to localise the
   information bottleneck of the hybrid pipeline.  Effective dimension is the
   participation ratio of the singular-value (energy) spectrum

         .. math::  d_{\mathrm{eff}} = \frac{(\sum_i \sigma_i^2)^2}
                                            {\sum_i \sigma_i^4}.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pennylane as qml
import torch

from . import models
from .common import PHYSICS, FeatureBundle, flat_grid


# --------------------------------------------------------------------------- #
# 1. Circuit structural specs
# --------------------------------------------------------------------------- #
def circuit_specs(n_qubits: int, n_layers: int = models.LAYERS_PAPER) -> Dict:
    """Exact structural statistics of the ansatz for ``n_qubits``.

    Returns depth, total gate count, per-gate-type counts and the number of
    trainable RX parameters.  Uses PennyLane's resource tracking on a concrete
    tape (random angles -- structure is data-independent).
    """
    qnode = models.make_probs_qnode(n_qubits, n_layers)
    inputs = np.random.uniform(-np.pi, np.pi, size=n_qubits)
    weights = np.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits))
    specs = qml.specs(qnode)(inputs, weights)

    resources = getattr(specs, "resources", None)
    gate_types: Dict[str, int] = {}
    depth = None
    num_gates = None
    if resources is not None:
        gate_types = dict(getattr(resources, "gate_types", {}) or {})
        depth = getattr(resources, "depth", None)
        num_gates = getattr(resources, "num_gates", None)

    # Analytic fall-backs / cross-checks (structure is fully deterministic).
    n_cnot_layers = sum(1 for l in range(n_layers) if l % 2 == 0) if n_qubits > 1 else 0
    analytic = {
        "n_RY_encoding": n_qubits,
        "n_RX_variational": n_layers * n_qubits,
        "n_CNOT": n_cnot_layers * n_qubits,
        "n_entangling_layers": n_cnot_layers,
    }
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "hilbert_dim": 2 ** n_qubits,
        "trainable_rx_params": n_layers * n_qubits,
        "circuit_depth": depth,
        "num_gates": num_gates,
        "gate_types": gate_types,
        "analytic": analytic,
    }


# --------------------------------------------------------------------------- #
# 2. Entanglement measures from a state vector
# --------------------------------------------------------------------------- #
def _reduced_dm(state: np.ndarray, keep: List[int], n_qubits: int) -> np.ndarray:
    """Partial trace of a pure state down to the wires in ``keep``."""
    psi = state.reshape([2] * n_qubits)
    keep = sorted(keep)
    trace_out = [i for i in range(n_qubits) if i not in keep]
    # rho = sum over traced indices  psi psi^*
    # Build via tensordot over the traced axes.
    psi_c = np.conj(psi)
    rho = np.tensordot(psi, psi_c, axes=(trace_out, trace_out))
    dim = 2 ** len(keep)
    return rho.reshape(dim, dim)


def _von_neumann_entropy(rho: np.ndarray) -> float:
    """S(rho) = -Tr rho log2 rho, computed from eigenvalues (base 2)."""
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals.real, 1e-12, None)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


def meyer_wallach(state: np.ndarray, n_qubits: int) -> float:
    r"""Meyer-Wallach global entanglement ``Q`` of a pure state."""
    purities = []
    for k in range(n_qubits):
        rho_k = _reduced_dm(state, [k], n_qubits)
        purities.append(np.trace(rho_k @ rho_k).real)
    return float(2.0 * (1.0 - np.mean(purities)))


def mean_bipartite_entropy(state: np.ndarray, n_qubits: int) -> float:
    r"""Von-Neumann entropy averaged over all contiguous bipartitions.

    Cuts wires ``{0..k}`` vs the rest for ``k = 0 .. n-2``.
    """
    if n_qubits < 2:
        return 0.0
    entropies = []
    for cut in range(1, n_qubits):
        rho_a = _reduced_dm(state, list(range(cut)), n_qubits)
        entropies.append(_von_neumann_entropy(rho_a))
    return float(np.mean(entropies))


def entanglement_over_domain(
    model: models.QAPINN,
    fb: Optional[FeatureBundle] = None,
    n_samples: int = 512,
    seed: int = 0,
) -> Dict:
    """Average Meyer-Wallach ``Q`` and bipartite entropy of the *trained* model's
    encoded state over a random sample of grid points.

    Angles are taken from the trained encoder so this reflects the actual states
    the network occupies while representing the Burgers solution.
    """
    rng = np.random.default_rng(seed)
    n = model.n_qubits
    state_qnode = models.make_state_qnode(n, model.n_layers)
    weights = model.qlayer.weights.detach().cpu().numpy()

    # sample (x, t), map to encoder angles using the trained encoder
    if fb is not None:
        x_col, t_col = flat_grid(fb.x_grid, fb.t_grid)
    else:  # dense fallback grid
        xs = np.linspace(PHYSICS.xmin, PHYSICS.xmax, 200)
        ts = np.linspace(PHYSICS.tmin, PHYSICS.T, 200)
        x_col, t_col = flat_grid(xs, ts)
    idx = rng.choice(x_col.shape[0], size=min(n_samples, x_col.shape[0]), replace=False)
    xn = torch.tensor(PHYSICS.norm_x(x_col[idx]), dtype=torch.float32)
    tn = torch.tensor(PHYSICS.norm_t(t_col[idx]), dtype=torch.float32)
    with torch.no_grad():
        angles = model.angles(xn, tn).cpu().numpy()

    Qs, Ss = [], []
    for a in angles:
        state = np.asarray(state_qnode(a, weights))
        Qs.append(meyer_wallach(state, n))
        Ss.append(mean_bipartite_entropy(state, n))
    return {
        "n_qubits": n,
        "n_samples": int(len(angles)),
        "meyer_wallach_mean": float(np.mean(Qs)),
        "meyer_wallach_std": float(np.std(Qs)),
        "meyer_wallach_max": float(np.max(Qs)),
        "bipartite_entropy_mean": float(np.mean(Ss)),
        "bipartite_entropy_std": float(np.std(Ss)),
        "bipartite_entropy_max": float(np.max(Ss)),
        "max_possible_entropy_bits": float(n // 2),  # ~ for balanced cut
        "samples": {"meyer_wallach": Qs, "bipartite_entropy": Ss},
    }


def entanglement_dynamics(
    model: models.QAPINN,
    fb: Optional[FeatureBundle] = None,
    alphas: Optional[np.ndarray] = None,
    n_samples: int = 256,
    seed: int = 0,
) -> Dict:
    r"""Trace mean entanglement as the encoder is homotopied ``W(alpha)=alpha*W*``.

    ``alpha = 0`` -> zero angles -> product state -> ``Q = 0``; ``alpha = 1`` ->
    trained operating point.  This is the reproducible proxy for how entanglement
    is generated as the classical encoder parameters move during optimisation.
    """
    rng = np.random.default_rng(seed)
    n = model.n_qubits
    if alphas is None:
        alphas = np.linspace(0.0, 1.2, 13)
    state_qnode = models.make_state_qnode(n, model.n_layers)
    weights = model.qlayer.weights.detach().cpu().numpy()

    if fb is not None:
        x_col, t_col = flat_grid(fb.x_grid, fb.t_grid)
    else:
        xs = np.linspace(PHYSICS.xmin, PHYSICS.xmax, 200)
        ts = np.linspace(PHYSICS.tmin, PHYSICS.T, 200)
        x_col, t_col = flat_grid(xs, ts)
    idx = rng.choice(x_col.shape[0], size=min(n_samples, x_col.shape[0]), replace=False)
    xn = torch.tensor(PHYSICS.norm_x(x_col[idx]), dtype=torch.float32)
    tn = torch.tensor(PHYSICS.norm_t(t_col[idx]), dtype=torch.float32)
    with torch.no_grad():
        base_angles = model.angles(xn, tn).cpu().numpy()

    q_curve, s_curve = [], []
    for alpha in alphas:
        angles = alpha * base_angles
        Qs = [meyer_wallach(np.asarray(state_qnode(a, weights)), n) for a in angles]
        Ss = [mean_bipartite_entropy(np.asarray(state_qnode(a, weights)), n) for a in angles]
        q_curve.append(float(np.mean(Qs)))
        s_curve.append(float(np.mean(Ss)))
    return {
        "n_qubits": n,
        "alphas": alphas.tolist(),
        "meyer_wallach_curve": q_curve,
        "bipartite_entropy_curve": s_curve,
    }


# --------------------------------------------------------------------------- #
# 3. Effective dimension / SVD of layer-wise representations
# --------------------------------------------------------------------------- #
def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def effective_dimension(x: np.ndarray) -> Dict:
    r"""Singular-value spectrum + effective dimension of a representation matrix.

    ``x`` has shape ``(N, d)``.  We centre, take the SVD, and report the singular
    values, the energy (variance) they explain and the participation-ratio
    effective dimension ``d_eff = (sum s^2)^2 / sum s^4``.
    """
    xc = _center(x.astype(np.float64))
    # economy SVD on the (possibly tall) matrix
    s = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    energy = s ** 2
    total = float(energy.sum()) + 1e-30
    d_eff = float((energy.sum() ** 2) / (np.sum(energy ** 2) + 1e-30))
    cum = np.cumsum(energy) / total
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    return {
        "ambient_dim": int(x.shape[1]),
        "singular_values": s.tolist(),
        "explained_variance_ratio": (energy / total).tolist(),
        "effective_dimension": d_eff,
        "n_components_90pct": n90,
        "n_components_99pct": n99,
        "rank": int(np.sum(s > (s.max() * 1e-8))),
    }


def information_bottleneck(fb: FeatureBundle, max_rows: int = 20000, seed: int = 0) -> Dict:
    """Effective-dimension analysis across the hybrid pipeline layers.

    Layers analysed (in forward order)::

        encoder_pre  ->  encoder_angles  ->  quantum_probs  ->  decoder_hidden  ->  output

    The layer with the sharpest drop in effective dimension marks the
    information bottleneck.
    """
    rng = np.random.default_rng(seed)
    stack = fb.layer_stack
    N = next(iter(stack.values())).shape[0]
    idx = rng.choice(N, size=min(max_rows, N), replace=False)
    results = {name: effective_dimension(arr[idx]) for name, arr in stack.items()}

    order = list(stack.keys())
    d_eff_seq = [results[k]["effective_dimension"] for k in order]
    # bottleneck = layer with minimum normalised effective dimension
    norm = [d / results[k]["ambient_dim"] for d, k in zip(d_eff_seq, order)]
    bottleneck_idx = int(np.argmin(norm))
    return {
        "layers": order,
        "per_layer": results,
        "effective_dimension_sequence": d_eff_seq,
        "normalised_effective_dimension": norm,
        "bottleneck_layer": order[bottleneck_idx],
        "n_rows_used": int(len(idx)),
    }


# --------------------------------------------------------------------------- #
# Orchestration helper
# --------------------------------------------------------------------------- #
def analyse_all(fb: FeatureBundle, qubit_range=(3, 4, 5), seed: int = 0) -> Dict:
    """Run every quantum-metric analysis for the checkpoint suite."""
    out: Dict = {"specs": {}, "entanglement": {}, "entanglement_dynamics": {}}
    for n in qubit_range:
        out["specs"][str(n)] = circuit_specs(n)
        try:
            model, _ = models.load_qapinn(n)
        except Exception as exc:  # pragma: no cover - defensive
            out["entanglement"][str(n)] = {"error": str(exc)}
            continue
        # entanglement over domain uses the model's own encoder; the shared
        # feature grid is only meaningful for the model that produced it.
        use_fb = fb if n == fb.n_qubits else None
        out["entanglement"][str(n)] = entanglement_over_domain(model, use_fb, seed=seed)
        out["entanglement_dynamics"][str(n)] = entanglement_dynamics(model, use_fb, seed=seed)

    out["information_bottleneck"] = information_bottleneck(fb, seed=seed)
    return out
