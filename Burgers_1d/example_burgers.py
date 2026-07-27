import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
from xai import report
from config import DEVICE, X_MIN, X_MAX, T_MIN, T_MAX, NU, u0_fn
from models import QAPINN, ClassicalPINN
from physics import build_batches, composite_loss, pde_residual
from train_qapinn import train_qapinn
from train import train_classical
from xai.adapter import QuantumProbe, TorchModelAdapter
from xai import layer2, scaling

BOUNDS = [(X_MIN, X_MAX), (T_MIN, T_MAX)]          
OUT = "outputs/xai"

### 1. train / load 2 models 
def get_models():
    qa, _, qa_snaps = train_qapinn(n_qubits=3, n_layers=2, reupload=True,
                                   epochs=300, snapshot_epochs=(0, 1000, 2000, 2999))
    cl, cl_hist, _ = train_classical(depth=4, width=8, adam_epochs=300)
    return qa, qa_snaps, cl, cl_hist

### 2. physics residual 
def residual_fn(adapter, X):
    """Wrap physics.pde_residual to the (adapter, X)->numpy signature."""
    Xr = X.clone().detach().requires_grad_(True)
    x, t = Xr[:, 0:1], Xr[:, 1:2]
    u = adapter(torch.cat([x, t], 1))
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return (ut + u * ux - NU * uxx).detach().cpu().numpy()


def main():
    qa, qa_snaps, cl, cl_hist = get_models()

    probe = QuantumProbe.from_qapinn(qa, device=DEVICE, d_in=2)   # full Layer-2 access
    classical = TorchModelAdapter(cl, device=DEVICE, name="classical_pinn", d_in=2)

    B = build_batches(n_pde=256, n_ic=256, n_bc=128)
    loss_q = lambda: composite_loss(qa, B)[0]
    loss_c = lambda: composite_loss(cl, B)[0]
    
    rep = report.XAIReport(probe, BOUNDS, classical=classical,
                           residual_fn=residual_fn, outdir=OUT, tag="burgers_4q")
    rep.run_layer2()
    rep.run_layer3(loss_q, classical_loss_fn=loss_c)

    def build_probe_and_metrics(nq):
        m, h, _ = train_qapinn(n_qubits=nq, epochs=1500)
        p = QuantumProbe.from_qapinn(m, device=DEVICE, d_in=2)
        return p, dict(runtime=float(h["wall"][-1]))
    rep.run_qubit_scaling(build_probe_and_metrics, qubit_list=(2, 4, 6))

    def make_fresh(nq, depth):
        m = QAPINN(n_qubits=nq, n_layers=depth, reupload=True).to(DEVICE)
        return QuantumProbe.from_qapinn(m, device=DEVICE, d_in=2)
    rep.run_barren_plateau(make_fresh, qubit_list=(2, 4, 6), depth_list=(2, 4, 8))

    weight_snaps = {e: sd["q_weights"].cpu().numpy() for e, sd in qa_snaps.items()}
    def make_from_weights(w):
        m = QAPINN(n_qubits=qa.n_qubits, n_layers=qa.n_layers,
                   reupload=qa.reupload).to(DEVICE)
        with torch.no_grad(): m.q_weights.copy_(torch.tensor(w))
        return QuantumProbe.from_qapinn(m, device=DEVICE, d_in=2)
    rep.run_state_evolution(make_from_weights, weight_snaps)

    from ground_truth import GroundTruth
    gt = GroundTruth("spectral")
    ref_fn = lambda P: gt(P[:, 0], P[:, 1])
    rep.run_domain_generalization({"qapinn": probe, "classical": classical}, ref_fn,
                                  extend_axis=1, factors=(1.0, 1.5, 2.0, 3.0))

    path = rep.save_manifest()
    print("XAI report written:", path)


if __name__ == "__main__":
    main()
