# `xai` — Explainable-AI module for QA-PINNs

Nothing here hard-codes Burgers. To move to 2D Burgers, Navier–Stokes, or a
real-world CFD problem you change three things: the input `bounds`, the
`residual_fn`, and (for Layer 4) the ground-truth reference.

## Install / place

Drop the `xai/` folder into your repo root (next to `models.py`, `physics.py`).
Requires: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`. The quantum
analyses need a model exposing measurement probabilities and (ideally) a
statevector — your `models.QAPINN` already does.

## The one abstraction: adapters

Every analysis talks to the model through an adapter, so it never depends on your
class internals.

```python
from xai.adapter import QuantumProbe, TorchModelAdapter

# QA-PINN: one line, straight from your models.QAPINN instance
probe     = QuantumProbe.from_qapinn(qapinn_model, device=DEVICE, d_in=2)
# classical baseline: wrap any nn.Module
classical = TorchModelAdapter(classical_model, name="classical_pinn", d_in=2)
```

`from_qapinn` relies only on the public attributes your class already exposes
(`n_qubits, n_layers, reupload, q_weights, qnode, statevector, head`).

## Quick start

```python
from xai import report
bounds = [(-1, 1), (0, 1)]          # (x, t)

rep = report.XAIReport(probe, bounds, classical=classical,
                       residual_fn=residual_fn, outdir="outputs/xai")
rep.run_layer2()                                   # all 11 analyses
rep.run_layer3(loss_q, classical_loss_fn=loss_c)   # optimisation geometry
rep.save_manifest()                                # -> manifest_<tag>.json + PNGs
```

`loss_q` / `loss_c` are zero-arg closures returning the **total PINN loss** at
each model's current parameters, e.g. `lambda: composite_loss(model, B)[0]`.

## Layer 2 — Quantum-layer explainability (11 analyses)

| # | Function | What it answers |
|---|----------|-----------------|
| 2.1 | `input_sensitivity` | how the input encoding drives learning; feature importance, encoding robustness |
| 2.2 | `measurement_operators` | informativeness of ⟨X⟩/⟨Y⟩/⟨Z⟩; correlation of each qubit with the PDE residual |
| 2.3 | `circuit_depth_analysis` | depth (1–10) vs loss / rel-L2 / grad-norm / expressivity / runtime |
| 2.4 | `entanglement_analysis` | Meyer–Wallach Q, per-qubit entropy, pair concurrence/correlation, Q vs error |
| 2.5 | `fourier_spectrum` | reachable vs realised frequency content (shocks); classical overlay |
| 2.6 | `expressivity_analysis` | function-class size: state overlap, kernel rank, effective dimension |
| 2.7 | `barren_plateau_analysis` | gradient variance vs #qubits / depth |
| 2.8 | `loss_landscape` | filter-normalised 2-D loss surface around the optimum |
| 2.9 | `measurement_distribution` | basis-measurement histogram: entropy, KL to uniform / reference |
| 2.10 | `feature_attribution` | saliency maps ‖∂u/∂input‖ over the domain (quantum vs classical) |
| 2.11 | `quantum_state_evolution` | Bloch trajectory, purity, fidelity across training snapshots |

Analyses 2.3, 2.7, 2.11 need training/rebuilds and are **opt-in** (you pass a
callback), so a default report never trains.

### Built-in honesty guardrails (per your project principles)

* **Fourier support is encoding-determined** (Schuld–Sweke–Meyer): 2.5 reports
  the *theoretical reachable band* (`= n_layers` if re-uploading, else 1)
  separately from the *empirical spectral centroid*; weights only redistribute
  amplitude within the band.
* **Entanglement is measured, not assumed**: 2.4 returns the observed Q with a
  conditional verdict and warns that a near-zero Q on few qubits may be a
  shape/batching artefact — states are processed per-sample to avoid that.
* **Random-init spectra are labelled**; trained/untrained are never conflated.

## Layer 3 — Optimisation analysis

Independent of quantum properties, so it runs identically on both models:
`gradient_diagnostics` (grad norm/variance, param norm), `hessian_diagnostics`
(Hutchinson trace, λmax/λmin via power iteration, condition number, stable-lr
ceiling `2/λmax`), `training_stability` (from your loss `hist`). `optimization_report`
bundles them into one figure per model.

## Layer 4 — Domain generalisation

`domain.domain_generalization` evaluates each model on progressively extrapolated
domains (e.g. `t∈[0,1] → [0,3]`) against a ground-truth callable and plots where
each breaks down. Wrap your solver: `ref_fn = lambda P: gt(P[:,0], P[:,1])`.

## Qubit scaling

`scaling.qubit_scaling` sweeps `{2,4,6,8,10}` qubits and tabulates accuracy,
effective dimension, Meyer–Wallach Q, measurement entropy, params, runtime via a
`build_probe_and_metrics(n_qubits)` callback.

## Reuse for another PDE / CFD problem

```python
bounds = [(x0,x1),(y0,y1),(t0,t1)]          # 2D Burgers / NS: d_in = 3
def residual_fn(adapter, X):                # your PDE residual, ->(N,) numpy
    ...
probe = QuantumProbe.from_qapinn(model, d_in=3)
report.run_full_report(probe, bounds, classical=classical,
                       loss_fn=loss_q, residual_fn=residual_fn)
```

Everything else is unchanged. Saliency (2.10) and Fourier (2.5) pick axes by
index, so they work in any dimension.

## Outputs

Each analysis returns a JSON-serialisable dict and writes a PNG. `save_manifest()`
emits `manifest_<tag>.json` capturing every result with provenance
(model, qubits, layers, reupload, bounds) — consistent with the rest of your
checkpointing.

See `example_burgers.py` for a complete, runnable integration against the
uploaded codebase.
