from __future__ import annotations
from typing import Callable, Optional, Sequence, Dict, Any
import os
import numpy as np

from .adapter import QuantumProbe, ModelAdapter
from . import layer2, layer3, domain as domain_mod, scaling as scaling_mod, utils


class XAIReport:
    def __init__(self, probe: QuantumProbe, bounds: Sequence[tuple],
                 classical: Optional[ModelAdapter] = None,
                 residual_fn: Optional[Callable] = None,
                 outdir: str = "outputs/xai", tag: Optional[str] = None):
        self.probe = probe
        self.bounds = bounds
        self.classical = classical
        self.residual_fn = residual_fn
        self.outdir = utils.ensure_dir(outdir)
        self.tag = tag or utils.timestamp()
        self.results: Dict[str, Any] = {}

    # -- Layer 2 ------------------------------------------------------------- #
    def run_layer2(self, seed: int = 0) -> Dict[str, Any]:
        p, b, o = self.probe, self.bounds, self.outdir
        L2 = {}
        L2["2.1_input_sensitivity"] = layer2.input_sensitivity(p, b, seed=seed, outdir=o)
        L2["2.2_measurement_operators"] = layer2.measurement_operators(
            p, b, residual_fn=self.residual_fn, seed=seed, outdir=o)
        L2["2.4_entanglement"] = layer2.entanglement_analysis(
            p, b, residual_fn=self.residual_fn, seed=seed, outdir=o)
        L2["2.5_fourier_spectrum"] = layer2.fourier_spectrum(
            p, b, classical_ref=self.classical, outdir=o)
        L2["2.6_expressivity"] = layer2.expressivity_analysis(p, b, seed=seed, outdir=o)
        L2["2.9_measurement_distribution"] = layer2.measurement_distribution(p, b, seed=seed, outdir=o)
        L2["2.10_feature_attribution"] = layer2.feature_attribution(p, b, outdir=o)
        if self.classical is not None:
            L2["2.10_feature_attribution_classical"] = layer2.feature_attribution(
                self.classical, b, outdir=o)
        self.results["layer2"] = L2
        return L2

    # -- Layer 3 ------------------------------------------------------------- #
    def run_layer3(self, loss_fn: Callable, hist: Optional[dict] = None,
                   include_landscape: bool = True,
                   classical_loss_fn: Optional[Callable] = None,
                   classical_hist: Optional[dict] = None) -> Dict[str, Any]:
        L3 = {"optimization_report": layer3.optimization_report(
            self.probe, loss_fn, hist=hist, outdir=self.outdir)}
        # Layer 3 needs a loss closure over the *classical* model's own params.
        if self.classical is not None and classical_loss_fn is not None:
            L3["optimization_report_classical"] = layer3.optimization_report(
                self.classical, classical_loss_fn, hist=classical_hist,
                outdir=self.outdir)
        if include_landscape:
            L3["loss_landscape"] = layer2.loss_landscape(
                self.probe, lambda: float(loss_fn()), outdir=self.outdir)
        self.results["layer3"] = L3
        return L3

    # -- opt-in heavy analyses ---------------------------------------------- #
    def run_depth_sweep(self, build_train_eval, **kw):
        r = layer2.circuit_depth_analysis(build_train_eval, name=self.probe.name,
                                          outdir=self.outdir, **kw)
        self.results.setdefault("layer2", {})["2.3_circuit_depth"] = r
        return r

    def run_barren_plateau(self, make_probe, **kw):
        r = layer2.barren_plateau_analysis(make_probe, self.bounds,
                                           name=self.probe.name, outdir=self.outdir, **kw)
        self.results.setdefault("layer2", {})["2.7_barren_plateau"] = r
        return r

    def run_state_evolution(self, make_probe_from_weights, weight_snapshots, **kw):
        r = layer2.quantum_state_evolution(make_probe_from_weights, weight_snapshots,
                                           self.bounds, name=self.probe.name,
                                           outdir=self.outdir, **kw)
        self.results.setdefault("layer2", {})["2.11_state_evolution"] = r
        return r

    def run_domain_generalization(self, adapters, ref_fn, **kw):
        r = domain_mod.domain_generalization(adapters, ref_fn, self.bounds,
                                             outdir=self.outdir, **kw)
        self.results["layer4"] = {"domain_generalization": r}
        return r

    def run_qubit_scaling(self, build_probe_and_metrics, **kw):
        r = scaling_mod.qubit_scaling(build_probe_and_metrics, self.bounds,
                                      name=self.probe.name, outdir=self.outdir, **kw)
        self.results["qubit_scaling"] = r
        return r

    # -- persistence --------------------------------------------------------- #
    def save_manifest(self) -> str:
        manifest = dict(
            tag=self.tag, model=self.probe.name,
            n_qubits=self.probe.n_qubits, n_layers=self.probe.n_layers,
            reupload=self.probe.reupload, bounds=[list(b) for b in self.bounds],
            has_statevector=self.probe.has_state(),
            has_classical_baseline=self.classical is not None,
            results=self.results,
        )
        path = os.path.join(self.outdir, f"manifest_{self.tag}.json")
        return utils.dump_json(manifest, path)


# ----------------------------------------------------------------------------- #
#  One-call convenience                                                          #
# ----------------------------------------------------------------------------- #
def run_full_report(probe: QuantumProbe, bounds: Sequence[tuple],
                    classical: Optional[ModelAdapter] = None,
                    loss_fn: Optional[Callable] = None,
                    residual_fn: Optional[Callable] = None,
                    hist: Optional[dict] = None,
                    outdir: str = "outputs/xai", tag: Optional[str] = None,
                    seed: int = 0) -> XAIReport:
    """
    Run the *non-training* portion of the battery end to end (Layer 2 always;
    Layer 3 + landscape if `loss_fn` given), then write the manifest. Heavy
    training-based analyses (depth sweep, barren plateau, scaling, state
    evolution, domain generalization) remain opt-in on the returned object.
    """
    rep = XAIReport(probe, bounds, classical=classical,
                    residual_fn=residual_fn, outdir=outdir, tag=tag)
    rep.run_layer2(seed=seed)
    if loss_fn is not None:
        rep.run_layer3(loss_fn, hist=hist)
    rep.save_manifest()
    return rep
