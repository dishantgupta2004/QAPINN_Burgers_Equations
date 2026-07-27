"""
scaling.py — Qubit-Scaling analysis  (spec: "for diff qubits: 2, 4, 6, 8, 10")
==============================================================================

Sweeps qubit count and collates the key explainability + performance scalars so
you can see how the quantum layer's behaviour scales: accuracy, expressivity
(effective dimension), entanglement (Meyer-Wallach Q), measurement-space usage
(entropy), trainability (gradient variance / barren-plateau onset), and cost
(runtime, parameter count).

Design: it takes a `build_probe_and_metrics(n_qubits)` callback that returns a
`(QuantumProbe, extra_metrics_dict)` tuple, so it stays agnostic to how you train
each model. The per-qubit explainability scalars are computed here from the probe;
`extra_metrics_dict` carries whatever you already measured (rel_l2, runtime, ...).
"""
from __future__ import annotations
from typing import Callable, Sequence, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .adapter import QuantumProbe
from . import layer2, utils


def qubit_scaling(build_probe_and_metrics: Callable[[int], Tuple[QuantumProbe, Dict[str, float]]],
                  bounds: Sequence[tuple],
                  qubit_list: Sequence[int] = (2, 4, 6, 8, 10),
                  n_eval: int = 400, plot: bool = True,
                  outdir: str = "outputs/xai", name: str = "qapinn") -> Dict[str, Any]:
    """
    Run the light explainability scalars for each qubit count and tabulate.

    Returns a dict with a `table` (list of per-qubit rows) plus a summary figure.
    Heavy metrics (full loss landscape) are intentionally excluded; use them via
    layer2 directly on a chosen qubit count.
    """
    rows = []
    for nq in qubit_list:
        probe, extra = build_probe_and_metrics(nq)
        row = dict(n_qubits=nq)
        row.update({k: float(v) for k, v in (extra or {}).items()})

        # expressivity (effective dimension) + measurement entropy are cheap
        try:
            ex = layer2.expressivity_analysis(probe, bounds, n_states=min(n_eval, 300),
                                              plot=False, outdir=outdir)
            row["effective_dimension"] = ex["effective_dimension"]
            row["kernel_rank"] = ex["kernel_rank"]
        except Exception as e:                       # keep the sweep alive
            row["effective_dimension"] = np.nan; row["_expr_err"] = str(e)

        try:
            md = layer2.measurement_distribution(probe, bounds, n=min(n_eval, 400),
                                                 plot=False, outdir=outdir)
            row["measurement_entropy"] = md["mean_entropy"]
        except Exception as e:
            row["measurement_entropy"] = np.nan; row["_md_err"] = str(e)

        if probe.has_state():
            try:
                en = layer2.entanglement_analysis(probe, bounds, n=min(n_eval, 256),
                                                  plot=False, outdir=outdir)
                row["meyer_wallach_Q"] = en["meyer_wallach_Q"]
            except Exception as e:
                row["meyer_wallach_Q"] = np.nan; row["_ent_err"] = str(e)

        row["param_count"] = probe.n_params()
        row["feature_dim"] = int(2 ** nq)
        rows.append(row)

    res = dict(analysis="qubit_scaling", qubit_list=list(qubit_list), table=rows)

    if plot:
        def col(k): return [r.get(k, np.nan) for r in rows]
        metrics = [m for m in ("rel_l2", "effective_dimension", "meyer_wallach_Q",
                               "measurement_entropy", "param_count", "runtime")
                   if any(m in r for r in rows)]
        n = len(metrics)
        fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
        for a, m in zip(ax[0], metrics):
            a.plot(qubit_list, col(m), "o-")
            a.set(xlabel="n_qubits", ylabel=m, title=f"{m} vs qubits")
            if m in ("rel_l2", "param_count"): a.set_yscale("log")
            a.grid(alpha=.3)
        fig.suptitle(f"Qubit-scaling analysis — {name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"scaling_{name}", outdir)
        plt.close(fig)
    return res
