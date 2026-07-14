"""
xai.models
==========

Faithful re-implementations of the two networks compared in this project, built
so that the *saved* checkpoints load exactly (``strict=True``):

* :class:`QAPINN`   -- hybrid quantum-assisted PINN (paper-aligned ansatz):
      Linear(2 -> n) encoder -> ``pi*tanh`` -> RY angle encoding ->
      ``n_layers`` RX-only variational blocks with alternating full-ring CNOTs ->
      full ``2**n`` computational-basis probability read-out ->
      Linear(2**n -> 2**(n-1)) -> Tanh -> Linear(2**(n-1) -> 1).

* :class:`ClassicalPINN` -- a plain tanh MLP; its widths are inferred directly
      from the checkpoint ``state_dict`` (the stored ``arch`` string is stale).

Both consume **normalised** inputs ``x, t in [-1, 1]`` (see :class:`~xai.common.Physics`).

The module also exposes bare PennyLane QNodes (:func:`make_probs_qnode`,
:func:`make_state_qnode`, :func:`make_expval_qnode`) reused by the quantum-metric
and barren-plateau analyses so that circuit structure is defined in exactly one
place.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from .common import resolve_checkpoint

LAYERS_PAPER = 8 


# --------------------------------------------------------------------------- #
# Circuit factory (single source of truth for the ansatz)
# --------------------------------------------------------------------------- #
def _apply_ansatz(inputs, weights, n_qubits: int, n_layers: int) -> None:
    """Apply the paper-aligned circuit body (encoding + variational layers).

    Parameters
    ----------
    inputs : array-like, shape (..., n_qubits)
        RY encoding angles (already ``pi*tanh`` scaled by the classical encoder).
    weights : array-like, shape (n_layers, n_qubits)
        Trainable RX rotation angles.
    """
    # --- data encoding: one RY per wire ---
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)
    # --- RX-only variational layers with alternating ring-CNOT entanglers ---
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i], wires=i)
        if l % 2 == 0 and n_qubits > 1:
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])


def make_probs_qnode(n_qubits: int, n_layers: int = LAYERS_PAPER):
    """QNode returning the full ``2**n`` basis-probability vector (torch iface)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        _apply_ansatz(inputs, weights, n_qubits, n_layers)
        return qml.probs(wires=range(n_qubits))

    return circuit


def make_state_qnode(n_qubits: int, n_layers: int = LAYERS_PAPER):
    """QNode returning the full state vector (for entanglement measures)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs, weights):
        _apply_ansatz(inputs, weights, n_qubits, n_layers)
        return qml.state()

    return circuit


def make_expval_qnode(n_qubits: int, n_layers: int = LAYERS_PAPER):
    """QNode returning ``<Z_0>`` -- the standard barren-plateau cost probe."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        _apply_ansatz(inputs, weights, n_qubits, n_layers)
        return qml.expval(qml.PauliZ(0))

    return circuit


# --------------------------------------------------------------------------- #
# Hybrid QA-PINN
# --------------------------------------------------------------------------- #
class QAPINN(nn.Module):
    """Hybrid quantum-assisted PINN (see module docstring)."""

    def __init__(self, n_qubits: int, n_layers: int = LAYERS_PAPER):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        H = 2 ** (n_qubits - 1)  # dynamic decoder hidden width
        self.encoder = nn.Linear(2, n_qubits)
        qnode = make_probs_qnode(n_qubits, n_layers)
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits)})
        self.dec_fc1 = nn.Linear(2 ** n_qubits, H)
        self.dec_act = nn.Tanh()
        self.dec_fc2 = nn.Linear(H, 1)

    # -- functional pieces (reused for feature extraction / landscapes) ------
    def angles(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return np.pi * torch.tanh(self.encoder(torch.cat([x, t], dim=1)))

    def forward(self, x, t, return_features: bool = False):
        enc_pre = self.encoder(torch.cat([x, t], dim=1))
        angles = np.pi * torch.tanh(enc_pre)
        probs = self.qlayer(angles)
        hidden = self.dec_act(self.dec_fc1(probs))
        u = self.dec_fc2(hidden)
        if return_features:
            return u, {
                "encoder_pre": enc_pre.detach(),
                "encoder_angles": angles.detach(),
                "quantum_probs": probs.detach(),
                "decoder_hidden": hidden.detach(),
                "output": u.detach(),
            }
        return u


# --------------------------------------------------------------------------- #
# Classical PINN
# --------------------------------------------------------------------------- #
class ClassicalPINN(nn.Module):
    """Plain fully-connected tanh PINN.

    ``layers`` is the list of layer widths, e.g. ``[2, 8, 8, 8, 8, 1]``.  The
    module name ``net`` and the even indexing (Linear at 0,2,4,... with Tanh in
    between) match the training checkpoint so ``state_dict`` loads strictly.
    """

    def __init__(self, layers):
        super().__init__()
        seq = []
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seq.append(nn.Tanh())
        self.net = nn.Sequential(*seq)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def _infer_mlp_layers(state_dict: Dict[str, torch.Tensor]):
    """Recover ``[in, h1, ..., out]`` widths from a ``net.*`` state dict."""
    widths = []
    idx = 0
    while f"net.{idx}.weight" in state_dict:
        w = state_dict[f"net.{idx}.weight"]
        if not widths:
            widths.append(w.shape[1])  # input dim
        widths.append(w.shape[0])
        idx += 2  # skip activation modules
    return widths


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_qapinn(n_qubits: int, path: str = None, map_location="cpu") -> Tuple[QAPINN, dict]:
    """Instantiate and load a QA-PINN checkpoint. Returns (model, checkpoint)."""
    if path is None:
        path = resolve_checkpoint(f"qapinn_{n_qubits}qubit.pt")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    n_layers = int(ckpt.get("architecture", {}).get("n_layers", LAYERS_PAPER))
    model = QAPINN(n_qubits, n_layers)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt


def load_classical(path: str = None, map_location="cpu") -> Tuple[ClassicalPINN, dict]:
    """Instantiate and load the classical PINN, inferring widths from weights."""
    if path is None:
        path = resolve_checkpoint("burgers_pinn.pt")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    layers = _infer_mlp_layers(ckpt["state_dict"])
    model = ClassicalPINN(layers)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt
