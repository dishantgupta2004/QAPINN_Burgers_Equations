import numpy as np, torch, torch.nn as nn, pennylane as qml
from config import DEVICE

# ---------------- Classical PINN ----------------
class ClassicalPINN(nn.Module):
    def __init__(self, depth=4, width=8, act=nn.Tanh):
        super().__init__()
        self.depth, self.width = depth, width
        L = [nn.Linear(2,width), act()]
        for _ in range(depth-1): L += [nn.Linear(width,width), act()]
        L += [nn.Linear(width,1)]
        self.net = nn.Sequential(*L)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, X): return self.net(X)

    @torch.no_grad()
    def layer_activations(self, X):
        """Returns dict {'input','h1',...,'hD','output'} of post-activation tensors."""
        out, h = {"input": X.detach().cpu().numpy()}, X
        li = 0
        for m in self.net:
            h = m(h)
            if isinstance(m, nn.Tanh):
                li += 1; out[f"h{li}"] = h.detach().cpu().numpy()
        out["output"] = h.detach().cpu().numpy()
        return out

# ---------------- QA-PINN ----------------
def make_qnode(n_qubits, n_layers=8, reupload=False, ret="probs"):
    """FIX: AngleEmbedding now tiles inputs across ALL n_qubits (no dangling ancilla)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    def _encode(inp):
        # inp: (..., 2). Tile (x,t,x,t,...) to n_qubits wires.
        idx = [i % inp.shape[-1] for i in range(n_qubits)]
        tiled = inp[..., idx]
        qml.AngleEmbedding(tiled, wires=range(n_qubits), rotation="X")

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        _encode(inputs)
        for l in range(n_layers):
            if reupload and l > 0: _encode(inputs)
            for w in range(n_qubits): qml.RY(weights[l, w, 0], wires=w)
            for w in range(n_qubits): qml.RZ(weights[l, w, 1], wires=w)
            for i in range(n_qubits):
                for j in range(i+1, n_qubits): qml.CNOT(wires=[i,j])
        return qml.probs(wires=range(n_qubits)) if ret=="probs" else qml.state()
    return circuit

class QAPINN(nn.Module):
    def __init__(self, n_qubits, hidden=8, n_layers=8, reupload=True):
        super().__init__()
        self.n_qubits, self.n_layers, self.reupload = n_qubits, n_layers, reupload
        self.hidden = hidden
        self.qnode = make_qnode(n_qubits, n_layers, reupload)
        self.q_weights = nn.Parameter(torch.empty(n_layers, n_qubits, 2))
        nn.init.uniform_(self.q_weights, -np.pi, np.pi)
        self.head = nn.Sequential(nn.Linear(2**n_qubits, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1))

    def _probs(self, X):
        return self.qnode(X, self.q_weights).to(X.dtype)

    def forward(self, X):
        return self.head(self._probs(X))

    @torch.no_grad()
    def layer_activations(self, X):
        p = self._probs(X)
        h1 = self.head[1](self.head[0](p))
        h2 = self.head[3](self.head[2](h1))
        out = self.head[4](h2)
        return {"input": X.cpu().numpy(), "quantum_probs": p.cpu().numpy(),
                "head_h1": h1.cpu().numpy(), "head_h2": h2.cpu().numpy(),
                "output": out.cpu().numpy()}

    @torch.no_grad()
    def statevector(self, X):
        sq = make_qnode(self.n_qubits, self.n_layers, self.reupload, ret="state")
        return sq(X, self.q_weights)
