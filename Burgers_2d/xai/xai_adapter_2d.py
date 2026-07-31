import torch
import numpy as np
import pennylane as qml
from xai.adapter import QuantumProbe


def _make_return_qnode(n_qubits, n_layers, reupload, ret):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        if reupload:
            for layer in range(weights.shape[0]):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer:layer + 1],
                                             wires=range(n_qubits))
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits)) if ret == "probs" else qml.state()
    return circuit


def make_probe(model, field="u", device="cpu", name=None):
    nq, nl, ru = model.n_qubits, model.n_layers, model.reupload
    probs_q = _make_return_qnode(nq, nl, ru, "probs")
    state_q = _make_return_qnode(nq, nl, ru, "state")
    idx = 0 if field == "u" else 1

    def encoded(z):
        return (torch.pi * model.encoder(z)).to("cpu")

    def probs_fn(Z, w):
        return probs_q(encoded(Z), w).to(Z.dtype)

    def state_fn(Z, w):
        return state_q(encoded(Z), w)

    probe = QuantumProbe(
        module=model, n_qubits=nq, q_weights=model.q_weights,
        probs_fn=probs_fn, state_fn=state_fn, head_fn=model.head,
        n_layers=nl, reupload=ru, device=device,
        name=name or f"qpinn2d_{nq}q_{field}", d_in=3,
    )
    probe._forward = lambda z: model(z)[:, idx:idx + 1]
    return probe