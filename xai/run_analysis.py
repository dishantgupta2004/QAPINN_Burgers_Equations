#!/usr/bin/env python
r"""
xai.run_analysis
===============

End-to-end orchestrator for the QA-PINN / Classical-PINN XAI suite.

It loads the saved checkpoints (``qapinn_{3,4,5}qubit.pt``, ``burgers_pinn.pt``)
and the pre-computed feature bundle (``qapinn_features.npz``), runs every
analysis module, writes a consolidated JSON of numerical results and renders all
figures to ``output/xai/``.

Usage
-----
    python xai/run_analysis.py                 # full pipeline
    python xai/run_analysis.py --fast          # smaller samples (quick smoke)
    python xai/run_analysis.py --skip-hessian  # skip the expensive Hessian stage
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# allow "python xai/run_analysis.py" from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from xai import common, expressivity, landscapes, quantum_metrics, visualizer  # noqa: E402


QUBITS = (3, 4, 5)


def _banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print("  " + msg)
    print("=" * 72, flush=True)


def _guard(name, fn, *args, **kwargs):
    """Run an analysis step, timing it and never letting one failure abort the run."""
    t0 = time.time()
    try:
        res = fn(*args, **kwargs)
        print(f"    [ok] {name}   ({time.time() - t0:.1f}s)", flush=True)
        return res
    except Exception as exc:  # pragma: no cover - defensive orchestration
        print(f"    [FAIL] {name}: {exc}", flush=True)
        traceback.print_exc()
        return {"error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description="QA-PINN XAI analysis suite")
    ap.add_argument("--fast", action="store_true", help="smaller sample sizes")
    ap.add_argument("--skip-hessian", action="store_true", help="skip Hessian stage")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    common.set_seed(args.seed)
    out_dir = common.ensure_output_dir()
    print(f"Output directory: {out_dir}")

    # -- load shared artefacts ------------------------------------------------
    _banner("Loading checkpoints & feature bundle")
    fb = common.load_features()
    sweep = common.load_sweep_results()
    print(f"  feature bundle: best QA-PINN = {fb.n_qubits} qubits, "
          f"{fb.n_layers} layers | grid {fb.U_grid.shape}")

    results: dict = {
        "meta": {
            "qubits": list(QUBITS),
            "best_qubits": fb.n_qubits,
            "physics": {
                "nu": common.PHYSICS.nu,
                "domain": [common.PHYSICS.xmin, common.PHYSICS.xmax],
                "T": common.PHYSICS.T,
            },
            "fast": args.fast,
        },
        "sweep": sweep,
    }
    figures: dict = {}

    ent_samples = 128 if args.fast else 384
    bp_samples = 60 if args.fast else 200

    # -- A. Fourier expressivity ---------------------------------------------
    _banner("A. Encoding schemes & Fourier spectrum analysis")
    spectra_x = _guard("fourier(x)", expressivity.compare_spectra,
                       QUBITS, axis="x", fixed=0.5, n=512)
    spectra_t = _guard("fourier(t)", expressivity.compare_spectra,
                       QUBITS, axis="t", fixed=0.5, n=512)
    results["fourier_x"] = spectra_x
    results["fourier_t"] = spectra_t
    figures["fourier_x"] = _guard("plot fourier(x)", visualizer.plot_fourier_spectra,
                                  spectra_x, out_dir, "x")
    figures["fourier_t"] = _guard("plot fourier(t)", visualizer.plot_fourier_spectra,
                                  spectra_t, out_dir, "t")
    figures["max_freq"] = _guard("plot max-freq", visualizer.plot_max_frequency_bar,
                                 spectra_x, out_dir)

    # -- B. Circuit depth & entanglement -------------------------------------
    _banner("B. Circuit depth & quantum entanglement metrics")
    qmetrics = {"specs": {}, "entanglement": {}, "entanglement_dynamics": {}}
    for n in QUBITS:
        qmetrics["specs"][str(n)] = _guard(f"specs {n}q",
                                           quantum_metrics.circuit_specs, n)
    from xai import models  # local import
    for n in QUBITS:
        model, _ = models.load_qapinn(n)
        use_fb = fb if n == fb.n_qubits else None
        qmetrics["entanglement"][str(n)] = _guard(
            f"entanglement {n}q", quantum_metrics.entanglement_over_domain,
            model, use_fb, n_samples=ent_samples, seed=args.seed)
        qmetrics["entanglement_dynamics"][str(n)] = _guard(
            f"ent-dynamics {n}q", quantum_metrics.entanglement_dynamics,
            model, use_fb, n_samples=min(ent_samples, 192), seed=args.seed)
    results["quantum_metrics"] = qmetrics
    figures["entanglement"] = _guard("plot entanglement",
                                     visualizer.plot_entanglement_trends,
                                     qmetrics["entanglement"], out_dir)
    figures["ent_dynamics"] = _guard("plot ent-dynamics",
                                     visualizer.plot_entanglement_dynamics,
                                     qmetrics["entanglement_dynamics"], out_dir)

    # -- C. Measurement operator / information bottleneck --------------------
    _banner("C. Measurement operator & information-bottleneck analysis")
    ib = _guard("information bottleneck",
                quantum_metrics.information_bottleneck, fb, seed=args.seed)
    results["information_bottleneck"] = ib
    figures["bottleneck"] = _guard("plot bottleneck",
                                   visualizer.plot_information_bottleneck, ib, out_dir)
    figures["embeddings"] = _guard("plot embeddings",
                                   visualizer.plot_feature_embeddings, fb, out_dir,
                                   3000 if args.fast else 5000, args.seed)

    # -- D. Loss landscape & barren plateaus ---------------------------------
    _banner("D. Loss landscape & barren-plateau diagnostics")
    bp = _guard("barren plateau", landscapes.barren_plateau_scan,
                QUBITS, n_samples=bp_samples, seed=args.seed)
    results["barren_plateau"] = bp
    figures["barren"] = _guard("plot barren", visualizer.plot_barren_plateau, bp, out_dir)

    slice_n = 25 if args.fast else 41
    surf_n = 17 if args.fast else 25
    slices_1d, surfaces_2d, hess = {}, {}, {}

    def _do_model(label, model):
        slices_1d[label] = _guard(f"1D slice {label}", landscapes.loss_slice_1d,
                                  model, fb, span=1.0, n=slice_n, seed=args.seed)
        surfaces_2d[label] = _guard(f"2D surface {label}", landscapes.loss_surface_2d,
                                    model, fb, span=1.0, n=surf_n, seed=args.seed)
        if not args.skip_hessian:
            hess[label] = _guard(f"hessian {label}", landscapes.hessian_spectrum,
                                 model, fb, k=6, seed=args.seed)

    cmodel, _ = models.load_classical()
    _do_model("classical", cmodel)
    for n in QUBITS:
        model, _ = models.load_qapinn(n)
        _do_model(f"{n}q", model)

    results["loss_slice_1d"] = slices_1d
    results["loss_surface_2d"] = surfaces_2d
    results["hessian"] = hess
    figures["slices_1d"] = _guard("plot 1D slices", visualizer.plot_loss_slices_1d,
                                  slices_1d, out_dir)
    figures["surfaces_2d"] = _guard("plot 2D surfaces", visualizer.plot_loss_surfaces_2d,
                                    surfaces_2d, out_dir)
    if hess:
        figures["hessian"] = _guard("plot hessian", visualizer.plot_hessian_spectrum,
                                    hess, out_dir)

    # -- E. Qubit scaling vs. expressivity + gradient dynamics ---------------
    _banner("E. Qubit scaling vs. problem expressivity")
    grad_dyn = _guard("gradient dynamics",
                      expressivity.gradient_dynamics_suite, QUBITS, seed=args.seed)
    results["gradient_dynamics"] = grad_dyn
    figures["qubit_scaling"] = _guard("plot qubit scaling",
                                      visualizer.plot_qubit_scaling, sweep, None, out_dir)
    figures["solution_fields"] = _guard("plot solution fields",
                                        visualizer.plot_solution_fields, fb, out_dir)

    # compute the expressivity-advantage threshold summary
    results["expressivity_threshold"] = _expressivity_threshold(sweep, bp, grad_dyn)

    # -- persist --------------------------------------------------------------
    _banner("Writing results")
    json_path = os.path.join(out_dir, "xai_results.json")
    common.dump_json(results, json_path)
    print(f"  numerical results -> {json_path}")
    saved = [p for p in figures.values() if isinstance(p, str) and p]
    print(f"  figures saved ({len(saved)}):")
    for p in sorted(saved):
        print(f"    - {os.path.relpath(p, common.ROOT)}")

    _print_headline(results)
    return 0


def _expressivity_threshold(sweep: dict, bp: dict, grad_dyn: dict) -> dict:
    r"""Quantify when adding qubits *helps* vs. *hurts*.

    We combine the marginal accuracy gain (drop in relative-L2) against the
    marginal optimisation cost (gradient-variance suppression + wall-clock).  The
    qubit count is 'beneficial' while accuracy improves faster than gradients
    vanish; it becomes 'detrimental' once the barren-plateau suppression
    (~2^-n) dominates the accuracy return.
    """
    res = sweep.get("results", [])
    if not res:
        return {}
    res = sorted(res, key=lambda r: r["n_qubits"])
    rows = []
    for i in range(1, len(res)):
        prev, cur = res[i - 1], res[i]
        d_rel = cur["rel_l2_vs_fem"] - prev["rel_l2_vs_fem"]  # <0 means improved
        d_params = cur["n_params"] - prev["n_params"]
        d_time = cur["train_time_s"] - prev["train_time_s"]
        rows.append({
            "from_qubits": prev["n_qubits"],
            "to_qubits": cur["n_qubits"],
            "delta_rel_l2": d_rel,
            "accuracy_improved": bool(d_rel < 0),
            "delta_params": d_params,
            "delta_train_time_s": d_time,
        })
    best = min(res, key=lambda r: r["rel_l2_vs_fem"])
    return {
        "marginal_steps": rows,
        "best_qubits": best["n_qubits"],
        "best_rel_l2": best["rel_l2_vs_fem"],
        "gradient_decay_factor_per_qubit": bp.get("fit", {}).get("decay_factor_per_qubit"),
        "criterion": (
            "Increase n while |d(rel_L2)|/d(n) exceeds the marginal optimisation "
            "penalty; once gradient variance falls ~2^-n and accuracy stops "
            "improving, added qubits only inflate the 2^n Hilbert space and "
            "harm optimisation."
        ),
    }


def _print_headline(results: dict) -> None:
    _banner("Headline comparison: QA-PINN vs. Classical PINN")
    st = results.get("fourier_x", {}).get("_summary_table", [])
    for row in st:
        line = f"  {row['model']:<16} output max-freq={row['max_frequency']:.2f}"
        if "quantum_max_frequency" in row:
            line += f"  quantum max-freq={row['quantum_max_frequency']:.2f}"
        print(line)
    ent = results.get("quantum_metrics", {}).get("entanglement", {})
    for k in sorted(ent):
        e = ent[k]
        if "error" not in e:
            print(f"  {k}q  Meyer-Wallach Q={e['meyer_wallach_mean']:.3f}  "
                  f"bipartite S={e['bipartite_entropy_mean']:.3f} bits")
    bp = results.get("barren_plateau", {}).get("fit", {})
    if bp:
        print(f"  barren-plateau decay factor/qubit = "
              f"{bp.get('decay_factor_per_qubit'):.3f} (ideal 0.5)")


if __name__ == "__main__":
    raise SystemExit(main())
