"""
domain.py — Layer 4: Domain generalization  (spec "Layer 4")
============================================================

"Instead of testing only x in [0,1], test other domains." Layer 4 probes how the
model behaves *outside* the training box — extrapolation in space and/or time —
and compares that degradation between the classical PINN and the QA-PINN.

Given a ground-truth callable (any of the codebase's solvers via the GroundTruth
interface, or an analytic reference), it measures relative-L2 on a sequence of
progressively-extrapolated domains and reports where each model breaks down.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt

from .adapter import ModelAdapter
from . import utils


def _rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-30))


def domain_generalization(adapters: Dict[str, ModelAdapter],
                          ref_fn: Callable[[np.ndarray], np.ndarray],
                          base_bounds: Sequence[tuple],
                          extend_axis: int = 1,
                          factors: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 3.0),
                          n: int = 4000, plot: bool = True,
                          outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    Evaluate each model on domains scaled by `factors` along `extend_axis`
    (e.g. extend t from [0,1] to [0,3]) and report relative-L2 vs `ref_fn`.

    Parameters
    ----------
    adapters : {name: ModelAdapter}   models to compare (classical + quantum).
    ref_fn   : callable(points (N,d_in)) -> (N,) ground-truth field. Wrap your
               GroundTruth: ``lambda P: gt(P[:,0], P[:,1])`` for (x,t).
    base_bounds : the training box, e.g. [(-1,1),(0,1)].
    extend_axis : which axis to stretch.
    """
    results = {name: [] for name in adapters}
    for f in factors:
        b = [list(x) for x in base_bounds]
        lo, hi = b[extend_axis]
        b[extend_axis] = [lo, lo + (hi - lo) * f]
        P = utils.sample_domain([tuple(x) for x in b], n, seed=0).cpu().numpy()
        ref = np.asarray(ref_fn(P)).ravel()
        for name, ad in adapters.items():
            pred = ad.predict(P).ravel()
            results[name].append(_rel_l2(pred, ref))

    res = dict(analysis="domain_generalization", extend_axis=extend_axis,
               factors=list(factors),
               rel_l2={k: v for k, v in results.items()})

    if plot:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for name, errs in results.items():
            ax.plot(factors, errs, "o-", lw=1.8, label=name)
        ax.axvline(1.0, color="k", ls=":", alpha=.6, label="training extent")
        ax.set(xlabel=f"domain extension factor (axis {extend_axis})",
               ylabel="relative L2 vs reference", yscale="log",
               title="Layer 4 — extrapolation / domain generalisation")
        ax.grid(alpha=.3); ax.legend()
        plt.tight_layout(); res["figure"] = utils.savefig(fig, "l4_domain_generalization", outdir)
        plt.close(fig)
    return res
