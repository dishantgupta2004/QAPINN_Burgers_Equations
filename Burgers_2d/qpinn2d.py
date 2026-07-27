"""Hybrid quantum-classical PINN for the 2D coupled Burgers equation.

Architecture:
    (x, y, t) -> classical encoder -> PQC (StronglyEntanglingLayers,
    optional data re-uploading) -> classical head -> (u, v)

Design constraints:
  * pde_residual requires SECOND derivatives, so the QNode must support
    double-backward. Only diff_method="backprop" on a statevector
    simulator provides this. adjoint/parameter-shift do not.
  * No .cpu() inside forward(): device switching severs the autograd
    graph. PennyLane's torch interface follows the input tensor device.
  * Encoder-first: raw (x, y, t) sits poorly inside the circuit's
    reachable Fourier spectrum (Schuld-Sweke-Meyer). A learned affine
    map into [-pi, pi] lets the optimiser place features where the
    encoding can represent them.
"""
import torch
import torch.nn as nn
import pennylane as qml


def make_strong_ent_qnode(n_qubits, n_layers, reupload=True):
    """Build a batched QNode returning per-wire Pauli-Z expectations.

    Parameters
    ----------
    n_qubits : int
        Circuit width. Encoder output dimension must equal this.
    n_layers : int
        Number of StronglyEntanglingLayers blocks. With reupload=True the
        features are re-encoded before each block, which multiplies the
        accessible Fourier frequency support by n_layers.
    reupload : bool
        Enable data re-uploading.

    Returns
    -------
    (qnode, weight_shape)
        weight_shape is (n_layers, n_qubits, 3) as required by
        StronglyEntanglingLayers.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        if reupload:
            for layer in range(weights.shape[0]):
                qml.AngleEmbedding(inputs, wires=range(n_qubits),
                                   rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer:layer + 1],
                                             wires=range(n_qubits))
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit, (n_layers, n_qubits, 3)


class QPINN2D(nn.Module):
    """Encoder (GPU) -> PQC (CPU) -> head (GPU).

    default.qubit allocates its statevector on CPU regardless of input
    device, so the quantum layer is pinned to CPU. The .to() transfers
    are autograd-tracked in both directions, so second-order derivatives
    still flow. q_weights is deliberately a CPU parameter: keeping it on
    GPU forces a cross-device op inside apply_operation.
    """

    def __init__(self, n_qubits=4, n_layers=6, enc_width=32,
                 head_width=32, reupload=True, out_dim=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload

        self.encoder = nn.Sequential(
            nn.Linear(3, enc_width), nn.Tanh(),
            nn.Linear(enc_width, enc_width), nn.Tanh(),
            nn.Linear(enc_width, n_qubits), nn.Tanh(),
        )

        self.qnode, shape = make_strong_ent_qnode(n_qubits, n_layers, reupload)
        # Registered as a buffer-free CPU parameter; excluded from .to(device)
        # by overriding _apply below.
        self.q_weights = nn.Parameter(0.1 * torch.randn(*shape))

        self.head = nn.Sequential(
            nn.Linear(n_qubits, head_width), nn.Tanh(),
            nn.Linear(head_width, head_width), nn.Tanh(),
            nn.Linear(head_width, out_dim),
        )

        for m in list(self.encoder) + list(self.head):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self._last_q = None

    def _apply(self, fn, recurse=True):
        """Move everything except q_weights; the PQC stays on CPU."""
        qw = self.q_weights
        del self._parameters["q_weights"]
        super()._apply(fn, recurse)
        self._parameters["q_weights"] = qw
        return self

    def forward(self, z):
        dev = z.device
        feats = torch.pi * self.encoder(z)
        q_out = self.qnode(feats.to("cpu"), self.q_weights)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out, dim=-1)
        if q_out.dim() == 1:
            q_out = q_out.unsqueeze(0)
        return self.head(q_out.to(dev).to(z.dtype))

    @torch.no_grad()
    def quantum_features(self, z):
        feats = torch.pi * self.encoder(z)
        q = self.qnode(feats.to("cpu"), self.q_weights)
        q = torch.stack(q, dim=-1) if isinstance(q, (list, tuple)) else q
        return q.detach()
