from __future__ import annotations
from typing import Callable, Optional, Sequence, Any
import numpy as np
import torch 

class ModelAdapter:
    """
    Minimal interface every analysis relies on.

    Parameters
    ----------
    forward : callable
        Maps a torch tensor of shape (N, d_in) -> (N, d_out). For a PINN this is
        the network's scalar (or vector) field prediction u(x, t, ...).
    parameters : iterable of torch.nn.Parameter, optional
        The *trainable* parameters (used by Layer-3 gradient / Hessian probes).
    device : str
    name : str
        Human-readable identifier used in figures / manifests.
    d_in : int
        Input dimensionality of the collocation coordinates (2 for 1D Burgers
        (x,t); 3 for 2D Burgers (x,y,t); etc.).
    """

    def __init__(self, forward: Callable[[torch.Tensor], torch.Tensor],
                 parameters: Optional[Sequence[torch.nn.Parameter]] = None,
                 device: str = "cpu", name: str = "model", d_in: int = 2):
        self._forward = forward
        self._params = list(parameters) if parameters is not None else []
        self.device = device
        self.name = name
        self.d_in = d_in

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self._forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """numpy -> numpy convenience wrapper (no grad)."""
        t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self._forward(t).detach().cpu().numpy()

    @property
    def parameters(self) -> list:
        return self._params

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self._params))

    @property
    def is_quantum(self) -> bool:
        return isinstance(self, QuantumProbe) or getattr(self, "_quantum", False)


class TorchModelAdapter(ModelAdapter):
    """Wrap any nn.Module whose ``forward(X)`` returns the field prediction."""

    def __init__(self, module: torch.nn.Module, device: str = "cpu",
                 name: str = "model", d_in: int = 2):
        super().__init__(forward=module.__call__,
                         parameters=list(module.parameters()),
                         device=device, name=name, d_in=d_in)
        self.module = module

class QuantumProbe(TorchModelAdapter):
    """
    Adapter for a QA-PINN. In addition to the forward map it exposes the quantum
    layer directly so Layer-2 analyses can interrogate the circuit itself.

    Required constructor arguments
    ------------------------------
    module : torch.nn.Module
        The full QA-PINN (quantum layer + classical head).
    n_qubits : int
    q_weights : torch.nn.Parameter
        The trainable rotation angles fed to the circuit.
    probs_fn : callable(inputs_tensor, weights) -> torch.Tensor
        Returns measurement *probabilities* over the computational basis,
        shape (N, 2**n_qubits). This is the quantum layer's raw output *before*
        the classical head — i.e. exactly your ``QAPINN._probs`` / ``qnode``.
    state_fn : callable(inputs_tensor, weights) -> complex tensor, optional
        Returns the statevector, shape (N, 2**n_qubits) or (2**n_qubits,).
        Used by entanglement / state-evolution analyses. If omitted, those
        analyses fall back to reconstructing rho from probabilities where
        possible, or are skipped.
    head_fn : callable(probs_tensor) -> torch.Tensor, optional
        The classical head applied to the probabilities (for measurement->
        prediction attribution).
    n_layers, reupload : circuit metadata (used for reporting / Fourier bounds).

    Notes
    -----
    This maps 1:1 onto the uploaded ``models.QAPINN``:
        probs_fn  = model._probs                     (or model.qnode(., q_weights))
        state_fn  = model.statevector
        q_weights = model.q_weights
        head_fn   = model.head
    """

    def __init__(self, module: torch.nn.Module, n_qubits: int,
                 q_weights: torch.nn.Parameter,
                 probs_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 state_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 head_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 n_layers: int = 8, reupload: bool = False,
                 device: str = "cpu", name: str = "qapinn", d_in: int = 2):
        super().__init__(module, device=device, name=name, d_in=d_in)
        self._quantum = True
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.reupload = bool(reupload)
        self.q_weights = q_weights
        self._probs_fn = probs_fn
        self._state_fn = state_fn
        self._head_fn = head_fn

    # -- quantum-layer outputs ------------------------------------------------ #
    def probs(self, X: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.q_weights if weights is None else weights
        return self._probs_fn(X, w)

    def statevector(self, X: torch.Tensor, weights: Optional[torch.Tensor] = None):
        if self._state_fn is None:
            raise NotImplementedError("This QuantumProbe was built without a state_fn.")
        w = self.q_weights if weights is None else weights
        return self._state_fn(X, w)

    def has_state(self) -> bool:
        return self._state_fn is not None

    def head(self, probs: torch.Tensor) -> torch.Tensor:
        if self._head_fn is None:
            raise NotImplementedError("This QuantumProbe was built without a head_fn.")
        return self._head_fn(probs)

    # -- convenience factory for the uploaded QAPINN class -------------------- #
    @classmethod
    def from_qapinn(cls, model, device: str = "cpu", name: Optional[str] = None,
                    d_in: int = 2) -> "QuantumProbe":
        """
        Build a probe straight from the uploaded ``models.QAPINN`` instance.

        Relies only on the public attributes that class already exposes:
        ``n_qubits, n_layers, reupload, q_weights, _probs, statevector, head``.
        """
        return cls(
            module=model,
            n_qubits=model.n_qubits,
            q_weights=model.q_weights,
            probs_fn=lambda X, w: model.qnode(X, w).to(X.dtype),
            state_fn=lambda X, w: model.statevector(X) if w is model.q_weights
                                   else _state_with_weights(model, X, w),
            head_fn=model.head,
            n_layers=getattr(model, "n_layers", 8),
            reupload=getattr(model, "reupload", False),
            device=device,
            name=name or f"qapinn_{model.n_qubits}q",
            d_in=d_in,
        )


def _state_with_weights(model, X, w):
    """Statevector at arbitrary weights (needed for state-evolution over epochs)."""
    import pennylane as qml
    sq = model.__class__.__mro__  # noqa: F841 (kept explicit for readability)
    # Reuse the model's own qnode factory if present; else fall back.
    from types import MethodType  # noqa
    # models.make_qnode(n_qubits, n_layers, reupload, ret="state")
    import importlib
    m = importlib.import_module(model.__module__)
    sq = m.make_qnode(model.n_qubits, model.n_layers, model.reupload, ret="state")
    return sq(X, w)
