"""
layer3.py — Optimization Analysis  (spec "Layer 3")
===================================================

Layer 3 is *independent of quantum properties*: it treats the model as a
parametric map and studies the optimisation geometry. It therefore runs
identically on a classical PINN and a QA-PINN, which is exactly what you want —
put the two side by side to see how the quantum layer changes the training
dynamics.

Tracked quantities (from the spec):
  * gradient norm & gradient variance
  * parameter norm
  * Hessian trace  (Hutchinson estimator)
  * condition number (top / bottom eigenvalue via power iteration + Lanczos-lite)
  * loss curvature (dominant Hessian eigenvalue)
  * training stability  (from a supplied loss history)
  * effective learning-rate ceiling  (2 / lambda_max)

Everything routes through a single `loss_fn()` closure that recomputes the total
PINN loss at the model's current parameters, so no training internals leak in.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

from .adapter import ModelAdapter
from . import utils


# ----------------------------------------------------------------------------- #
#  Core gradient / parameter diagnostics                                         #
# ----------------------------------------------------------------------------- #
def gradient_diagnostics(adapter: ModelAdapter, loss_fn: Callable[[], torch.Tensor],
                         n_batches: int = 20) -> Dict[str, Any]:
    """
    Gradient norm, gradient variance (across `n_batches` recomputations — useful
    when the loss uses freshly sampled collocation points each call), and
    parameter norm. `loss_fn()` must return a *scalar tensor* with grad enabled.
    """
    params = [p for p in adapter.parameters if p.requires_grad]
    if not params:
        return dict(analysis="gradient_diagnostics", skipped="no trainable params")
    grad_norms, flat_grads = [], []
    for _ in range(n_batches):
        for p in params:
            if p.grad is not None: p.grad = None
        loss = loss_fn()
        loss.backward()
        present = [p for p in params if p.grad is not None]
        if not present:
            # loss does not depend on this model's params (e.g. wrong closure)
            return dict(analysis="gradient_diagnostics",
                        skipped="loss produced no gradient for this model's parameters",
                        param_norm=float(torch.cat([p.detach().reshape(-1)
                                     for p in params]).norm()),
                        n_params=int(sum(p.numel() for p in params)))
        g = torch.cat([p.grad.detach().reshape(-1) for p in present])
        grad_norms.append(float(g.norm()))
        flat_grads.append(g.cpu().numpy())
    G = np.stack(flat_grads)
    pnorm = float(torch.cat([p.detach().reshape(-1) for p in params]).norm())
    return dict(analysis="gradient_diagnostics",
                grad_norm_mean=float(np.mean(grad_norms)),
                grad_norm_std=float(np.std(grad_norms)),
                grad_variance=float(G.var(0).mean()),
                param_norm=pnorm, n_params=int(sum(p.numel() for p in params)))


# ----------------------------------------------------------------------------- #
#  Hessian-vector products, trace, spectrum, condition number                    #
# ----------------------------------------------------------------------------- #
def _hvp(loss: torch.Tensor, params, vec):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    dot = (flat * vec).sum()
    hv = torch.autograd.grad(dot, params, retain_graph=True)
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def hessian_diagnostics(adapter: ModelAdapter, loss_fn: Callable[[], torch.Tensor],
                        n_hutchinson: int = 20, n_power_iter: int = 40,
                        seed: int = 0) -> Dict[str, Any]:
    """
    Hessian trace (Hutchinson), dominant eigenvalue (power iteration), a smallest-
    magnitude eigenvalue estimate (shifted power iteration), condition number,
    and the implied stable-learning-rate ceiling 2/lambda_max.

    Curvature summary for the loss surface at the current optimum. Comparable
    across classical and quantum models.
    """
    params = [p for p in adapter.parameters if p.requires_grad]
    n = int(sum(p.numel() for p in params))
    g = torch.Generator().manual_seed(seed)

    # Hutchinson trace
    tr = 0.0
    for _ in range(n_hutchinson):
        v = torch.randint(0, 2, (n,), generator=g).float().to(next(iter(params)).device) * 2 - 1
        loss = loss_fn()
        hv = _hvp(loss, params, v)
        tr += float((v * hv).sum())
    tr /= n_hutchinson

    # dominant eigenvalue via power iteration
    v = torch.randn(n, generator=g).to(next(iter(params)).device); v /= v.norm()
    lam_max = 0.0
    for _ in range(n_power_iter):
        loss = loss_fn()
        hv = _hvp(loss, params, v)
        lam_max = float((v * hv).sum())
        nv = hv.norm()
        if nv < 1e-12: break
        v = hv / nv

    # smallest eigenvalue via shifted power iteration on (lam_max*I - H)
    v = torch.randn(n, generator=g).to(next(iter(params)).device); v /= v.norm()
    lam_shift = 0.0
    for _ in range(n_power_iter):
        loss = loss_fn()
        hv = _hvp(loss, params, v)
        shifted = lam_max * v - hv
        lam_shift = float((v * shifted).sum())
        nv = shifted.norm()
        if nv < 1e-12: break
        v = shifted / nv
    lam_min = lam_max - lam_shift

    cond = float(abs(lam_max) / (abs(lam_min) + 1e-12))
    return dict(analysis="hessian_diagnostics",
                hessian_trace=tr, lambda_max=lam_max, lambda_min=lam_min,
                condition_number=cond,
                lr_ceiling=float(2.0 / (abs(lam_max) + 1e-12)),
                n_params=n)


# ----------------------------------------------------------------------------- #
#  Training-stability summary from a loss history                                #
# ----------------------------------------------------------------------------- #
def training_stability(hist: Dict[str, np.ndarray], window: int = 200) -> Dict[str, Any]:
    """
    Summarise stability from a loss history dict (same shape the codebase's
    trainers already emit: keys 'total','pde','ic','bc','iter','wall').

    Reports: final loss, monotonicity fraction (how often loss decreased),
    late-phase relative volatility (std/mean over the last `window` iters), and
    the iteration where 90% of total decrease was achieved (convergence speed).
    """
    tot = np.asarray(hist["total"], float)
    diffs = np.diff(tot)
    monotonic_frac = float((diffs < 0).mean())
    late = tot[-window:] if len(tot) > window else tot
    volatility = float(np.std(late) / (np.mean(late) + 1e-30))
    span = tot[0] - tot[-1]
    idx90 = int(np.argmax(tot <= tot[0] - 0.9 * span)) if span > 0 else len(tot) - 1
    return dict(analysis="training_stability",
                final_loss=float(tot[-1]), monotonic_fraction=monotonic_frac,
                late_volatility=volatility, iters_to_90pct=idx90,
                total_iters=len(tot))


# ----------------------------------------------------------------------------- #
#  One-call optimisation report + figure                                         #
# ----------------------------------------------------------------------------- #
def optimization_report(adapter: ModelAdapter, loss_fn: Callable[[], torch.Tensor],
                        hist: Optional[Dict[str, np.ndarray]] = None,
                        plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """Bundle Layer-3 diagnostics for one model into a single result + figure."""
    grad = gradient_diagnostics(adapter, loss_fn)
    if "skipped" in grad:
        res = dict(analysis="optimization_report", model=adapter.name,
                   gradient=grad, hessian=None,
                   stability=training_stability(hist) if hist is not None else None,
                   note="Layer-3 curvature skipped: loss_fn did not depend on this "
                        "model's parameters (supply a per-model loss closure).")
        return res
    hess = hessian_diagnostics(adapter, loss_fn)
    stab = training_stability(hist) if hist is not None else None
    res = dict(analysis="optimization_report", model=adapter.name,
               gradient=grad, hessian=hess, stability=stab)

    if plot:
        panels = 2 + (1 if stab else 0)
        fig, ax = plt.subplots(1, panels, figsize=(4.2 * panels, 3.4), squeeze=False)
        ax = ax[0]
        ax[0].bar(["‖g‖", "Var[g]", "‖θ‖"],
                  [grad["grad_norm_mean"], grad["grad_variance"], grad["param_norm"]],
                  color=["C0", "C1", "C2"])
        ax[0].set_yscale("log"); ax[0].set_title("Gradient / parameter norms")
        ax[1].bar(["λmax", "|λmin|", "tr(H)", "cond"],
                  [abs(hess["lambda_max"]), abs(hess["lambda_min"]),
                   abs(hess["hessian_trace"]), hess["condition_number"]],
                  color="C3")
        ax[1].set_yscale("log")
        ax[1].set_title(f"Hessian curvature\nlr≤{hess['lr_ceiling']:.2e}")
        if stab:
            tot = np.asarray(hist["total"], float)
            ax[2].semilogy(tot, lw=1.2)
            ax[2].axvline(stab["iters_to_90pct"], color="C3", ls="--",
                          label=f"90% @ {stab['iters_to_90pct']}")
            ax[2].set(xlabel="iter", ylabel="total loss",
                      title=f"Stability\nvol={stab['late_volatility']:.2e}")
            ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        fig.suptitle(f"Layer 3 — Optimisation analysis — {adapter.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l3_optim_{adapter.name}", outdir)
        plt.close(fig)
    return res
