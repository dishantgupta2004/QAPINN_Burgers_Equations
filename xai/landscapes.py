r"""
xai.landscapes
=============

Optimisation-geometry diagnostics for the QA-PINN vs. Classical-PINN comparison.

Contents
--------
1. **Physics-informed loss** (:func:`build_loss_closure`) reconstructed from the
   FEM reference grid: data anchors + PDE residual + initial/boundary conditions,
   with the training weights ``(1, 1, 10, 10)``.  Works for either model.

2. **Barren-plateau diagnostics** (:func:`barren_plateau_scan`) -- variance of
   :math:`\partial\langle Z_0\rangle/\partial\theta` over random initialisations
   for :math:`n\in\{3,4,5\}` qubits, fitted against the McClean *et al.*
   prediction :math:`\operatorname{Var}\sim \mathcal{O}(2^{-n})`.

3. **Hessian spectrum** (:func:`hessian_spectrum`) at the converged parameters
   via Hessian-vector products + Lanczos (``scipy.sparse.linalg.eigsh``): extreme
   eigenvalues, curvature sign, condition number and a Hutchinson trace estimate.

4. **Loss-landscape slices** (:func:`loss_slice_1d`, :func:`loss_surface_2d`)
   with **filter normalisation** (Li *et al.* 2018) so QA-PINN and classical PINN
   landscapes are compared on a common, scale-invariant footing.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import models
from .common import PHYSICS, FeatureBundle, flat_grid
W_DATA, W_PDE, W_IC, W_BC = 1.0, 1.0, 10.0, 10.0

def _fem_tensors(fb: FeatureBundle, n_data=800, n_coll=800, n_bcic=200, seed=0):
    """Sample supervised / collocation / IC / BC tensors from the FEM grid.

    All coordinates are returned **normalised** to ``[-1, 1]``.  Data & IC/BC
    targets are read directly from the FEM field so we do not need the analytic
    initial condition.
    """
    rng = np.random.default_rng(seed)
    x_grid, t_grid, U = fb.x_grid, fb.t_grid, fb.U_grid  # U: (nt, nx)
    nt, nx = U.shape

    # --- supervised anchors: random interior grid nodes ---
    ti = rng.integers(0, nt, n_data)
    xi = rng.integers(0, nx, n_data)
    xd = PHYSICS.norm_x(x_grid[xi]).reshape(-1, 1)
    td = PHYSICS.norm_t(t_grid[ti]).reshape(-1, 1)
    ud = U[ti, xi].reshape(-1, 1)

    # --- PDE collocation: uniform random on normalised domain ---
    xc = rng.uniform(-1, 1, (n_coll, 1))
    tc = rng.uniform(-1, 1, (n_coll, 1))

    # --- initial condition: t = 0 row ---
    xi0 = rng.integers(0, nx, n_bcic)
    xic = PHYSICS.norm_x(x_grid[xi0]).reshape(-1, 1)
    tic = np.full((n_bcic, 1), PHYSICS.norm_t(t_grid[0]))
    uic = U[0, xi0].reshape(-1, 1)

    # --- boundary condition: x = xmin / xmax columns ---
    tb = rng.integers(0, nt, n_bcic)
    left = rng.random(n_bcic) < 0.5
    xb_idx = np.where(left, 0, nx - 1)
    xbc = PHYSICS.norm_x(x_grid[xb_idx]).reshape(-1, 1)
    tbc = PHYSICS.norm_t(t_grid[tb]).reshape(-1, 1)
    ubc = U[tb, xb_idx].reshape(-1, 1)

    to_t = lambda a, rg=False: torch.tensor(a, dtype=torch.float32, requires_grad=rg)
    return {
        "xd": to_t(xd), "td": to_t(td), "ud": to_t(ud),
        "xc": to_t(xc, True), "tc": to_t(tc, True),
        "xic": to_t(xic), "tic": to_t(tic), "uic": to_t(uic),
        "xbc": to_t(xbc), "tbc": to_t(tbc), "ubc": to_t(ubc),
    }


def build_loss_closure(
    model: torch.nn.Module,
    fb: FeatureBundle,
    include_pde: bool = True,
    seed: int = 0,
    **sizes,
) -> Tuple[Callable[[], torch.Tensor], Dict]:
    """Return a zero-arg closure computing the composite PINN loss for ``model``.

    The batch is sampled once and frozen so landscape slices / Hessian products
    are evaluated on a *fixed* objective.  ``include_pde=False`` yields the
    data+IC+BC surrogate (used as a robust Hessian fallback).
    """
    b = _fem_tensors(fb, seed=seed, **sizes)
    mse = torch.nn.MSELoss()

    def closure() -> torch.Tensor:
        loss = W_DATA * mse(model(b["xd"], b["td"]), b["ud"])
        loss = loss + W_IC * mse(model(b["xic"], b["tic"]), b["uic"])
        loss = loss + W_BC * mse(model(b["xbc"], b["tbc"]), b["ubc"])
        if include_pde:
            xc, tc = b["xc"], b["tc"]
            u = model(xc, tc)
            u_th = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True)[0]
            u_xh = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
            u_xxh = torch.autograd.grad(u_xh, xc, torch.ones_like(u_xh), create_graph=True)[0]
            u_t = u_th * PHYSICS.st
            u_x = u_xh * PHYSICS.sx
            u_xx = u_xxh * PHYSICS.sx * PHYSICS.sx
            res = u_t + u * u_x - PHYSICS.nu * u_xx
            loss = loss + W_PDE * (res ** 2).mean()
        return loss

    return closure, b


# --------------------------------------------------------------------------- #
# 1. Barren plateaus
# --------------------------------------------------------------------------- #
def barren_plateau_scan(
    qubit_range=(3, 4, 5),
    n_layers: int = models.LAYERS_PAPER,
    n_samples: int = 200,
    seed: int = 0,
) -> Dict:
    r"""Gradient variance of a variational RX weight vs. qubit count.

    For each ``n`` we draw ``n_samples`` random weight tensors (uniform on
    :math:`[-\pi,\pi]`) and a random encoding input, evaluate the standard cost
    :math:`C=\langle Z_0\rangle`, and record :math:`\partial C/\partial\theta`
    for a fixed variational angle.  The empirical variance is compared with the
    barren-plateau law :math:`\operatorname{Var}\propto 2^{-n}`.
    """
    rng = np.random.default_rng(seed)
    results = {}
    variances = []
    ns = list(qubit_range)
    for n in ns:
        qnode = models.make_expval_qnode(n, n_layers)
        grads = []
        for _ in range(n_samples):
            inp = torch.tensor(rng.uniform(-np.pi, np.pi, n), dtype=torch.float64)
            w = torch.tensor(
                rng.uniform(-np.pi, np.pi, (n_layers, n)),
                dtype=torch.float64,
                requires_grad=True,
            )
            c = qnode(inp, w)
            (g,) = torch.autograd.grad(c, w)
            grads.append(float(g[0, 0]))  # a fixed variational angle
        grads = np.asarray(grads)
        v = float(np.var(grads))
        variances.append(v)
        results[str(n)] = {
            "grad_mean": float(np.mean(grads)),
            "grad_variance": v,
            "grad_abs_mean": float(np.mean(np.abs(grads))),
            "n_samples": n_samples,
        }

    # log-linear fit  log(Var) = a - b*n ; barren-plateau => b ~ ln 2 ~ 0.693
    ns_arr = np.asarray(ns, dtype=float)
    logv = np.log(np.asarray(variances) + 1e-30)
    b_slope, a_int = np.polyfit(ns_arr, logv, 1)
    decay_per_qubit = float(np.exp(b_slope))  # multiplicative factor per +1 qubit
    return {
        "qubits": ns,
        "variances": variances,
        "per_qubit": results,
        "fit": {
            "log_slope": float(b_slope),
            "intercept": float(a_int),
            "decay_factor_per_qubit": decay_per_qubit,
            "ideal_barren_slope": float(-np.log(2)),
            "ideal_decay_factor": 0.5,
            "interpretation": (
                "decay_factor ~ 0.5 per added qubit indicates exponential "
                "gradient suppression (barren plateau)."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# 2. Hessian spectrum via HVP + Lanczos
# --------------------------------------------------------------------------- #
def _flat_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _hvp(loss_closure, params, vec_list):
    """Hessian-vector product H·v via double backprop."""
    loss = loss_closure()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vec_list))
    hv = torch.autograd.grad(dot, params, retain_graph=False)
    return [h.detach() for h in hv]


def hessian_spectrum(
    model: torch.nn.Module,
    fb: FeatureBundle,
    k: int = 6,
    include_pde: bool = False,
    hutchinson_samples: int = 10,
    hess_sizes: Optional[Dict] = None,
    seed: int = 0,
) -> Dict:
    """Extreme Hessian eigenvalues + trace at the converged parameters.

    Uses matrix-free Hessian-vector products fed to ``scipy.sparse.linalg.eigsh``
    (Lanczos) for the ``k`` largest and smallest algebraic eigenvalues, plus a
    Hutchinson stochastic trace estimate.

    By default the curvature is measured on the **supervised** objective
    (data + IC + BC).  This is deliberate: the full PDE term requires two input
    derivatives, so its parameter-Hessian is a *third*-order derivative whose
    HVP is ~50x slower; the supervised Hessian is a faithful, tractable probe of
    how sharp / well-conditioned each model's minimum is.  Set ``include_pde=True``
    to measure the full composite curvature (slow).
    """
    from scipy.sparse.linalg import LinearOperator, eigsh
    from scipy.sparse.linalg import ArpackNoConvergence

    torch.manual_seed(seed)
    params = _flat_params(model)
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    dim = int(sum(numels))

    sizes = hess_sizes or {"n_data": 256, "n_coll": 256, "n_bcic": 100}
    closure, _ = build_loss_closure(model, fb, include_pde=include_pde, seed=seed, **sizes)
    used_pde = include_pde

    def to_vec_list(x: np.ndarray):
        out, i = [], 0
        for sh, ne in zip(shapes, numels):
            out.append(torch.tensor(x[i:i + ne].reshape(sh), dtype=torch.float32))
            i += ne
        return out

    def matvec(x: np.ndarray) -> np.ndarray:
        vl = to_vec_list(np.asarray(x, dtype=np.float64))
        hv = _hvp(closure, params, vl)
        return np.concatenate([h.cpu().numpy().reshape(-1) for h in hv]).astype(np.float64)

    H = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    kk = int(min(k, dim - 2)) if dim > 4 else 1

    def _extreme(which):
        try:
            return eigsh(H, k=kk, which=which, return_eigenvectors=False,
                         maxiter=max(200, dim * 5), tol=1e-3)
        except ArpackNoConvergence as exc:  # partial results still useful
            return exc.eigenvalues

    if dim <= 260:
        # dense assembly is cheap & exact for small models
        cols = [matvec(e) for e in np.eye(dim)]
        Hd = np.stack(cols, axis=1)
        Hd = 0.5 * (Hd + Hd.T)
        ev = np.linalg.eigvalsh(Hd)
        top, bot = ev[-kk:], ev[:kk]
    else:
        top = _extreme("LA")
        bot = _extreme("SA")

    top = np.sort(np.real(top))[::-1]
    bot = np.sort(np.real(bot))

    # Hutchinson trace estimate: E[v^T H v], v ~ Rademacher
    traces = []
    for _ in range(hutchinson_samples):
        v = [torch.randint(0, 2, p.shape, dtype=torch.float32) * 2 - 1 for p in params]
        hv = _hvp(closure, params, v)
        traces.append(float(sum((h * vv).sum() for h, vv in zip(hv, v))))
    trace_est = float(np.mean(traces))

    lam_max = float(top[0])
    lam_min = float(bot[0])
    pos = float(top[top > 0].size)
    return {
        "n_params": dim,
        "used_pde": used_pde,
        "top_eigenvalues": top.tolist(),
        "bottom_eigenvalues": bot.tolist(),
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "negative_curvature": bool(lam_min < -1e-6),
        "condition_number": float(abs(lam_max) / (abs(lam_min) + 1e-12)),
        "hutchinson_trace": trace_est,
        "sharpness": lam_max,  # local sharpness proxy
    }


# --------------------------------------------------------------------------- #
# 3. Filter-normalised loss-landscape slices
# --------------------------------------------------------------------------- #
def _filter_normalized_direction(params, seed=0):
    """A random direction with per-filter (per-parameter-tensor) norm matched to
    the corresponding weights (Li et al. 2018), so distances are comparable
    across networks of different scale."""
    g = torch.Generator().manual_seed(seed)
    direction = []
    for p in params:
        d = torch.randn(p.shape, generator=g)
        # normalise each tensor's direction to the tensor's own norm
        d = d * (p.norm() / (d.norm() + 1e-12))
        direction.append(d.detach())
    return direction


def _set_params(model, base, direction, alpha, direction2=None, beta=0.0):
    with torch.no_grad():
        for p, b, d in zip(model.parameters(), base, direction):
            p.copy_(b + alpha * d)
        if direction2 is not None:
            for p, d2 in zip(model.parameters(), direction2):
                p.add_(beta * d2)


def loss_slice_1d(
    model: torch.nn.Module,
    fb: FeatureBundle,
    span: float = 1.0,
    n: int = 41,
    seed: int = 0,
    include_pde: bool = True,
) -> Dict:
    """1-D filter-normalised loss profile through the converged parameters."""
    params = _flat_params(model)
    base = [p.detach().clone() for p in params]
    direction = _filter_normalized_direction(base, seed)
    closure, _ = build_loss_closure(model, fb, include_pde=include_pde, seed=seed)

    alphas = np.linspace(-span, span, n)
    losses = []
    for a in alphas:
        _set_params(model, base, direction, float(a))
        with torch.no_grad() if not include_pde else torch.enable_grad():
            losses.append(float(closure().item()))
    _set_params(model, base, direction, 0.0)  # restore
    losses = np.asarray(losses)
    mid = n // 2
    # local curvature via central 2nd difference at alpha=0
    h = alphas[1] - alphas[0]
    curv = float((losses[mid + 1] - 2 * losses[mid] + losses[mid - 1]) / (h * h))
    return {
        "alphas": alphas.tolist(),
        "losses": losses.tolist(),
        "loss_at_min_of_slice": float(losses.min()),
        "loss_at_center": float(losses[mid]),
        "local_curvature": curv,
        "roughness": float(np.std(np.diff(losses, 2))),  # 2nd-diff variability
    }


def loss_surface_2d(
    model: torch.nn.Module,
    fb: FeatureBundle,
    span: float = 1.0,
    n: int = 25,
    seed: int = 0,
    include_pde: bool = True,
) -> Dict:
    """2-D filter-normalised loss surface (two independent random directions)."""
    params = _flat_params(model)
    base = [p.detach().clone() for p in params]
    d1 = _filter_normalized_direction(base, seed)
    d2 = _filter_normalized_direction(base, seed + 1)
    closure, _ = build_loss_closure(model, fb, include_pde=include_pde, seed=seed)

    grid = np.linspace(-span, span, n)
    Z = np.zeros((n, n))
    for i, a in enumerate(grid):
        for j, bb in enumerate(grid):
            _set_params(model, base, d1, float(a), d2, float(bb))
            with torch.no_grad() if not include_pde else torch.enable_grad():
                Z[i, j] = float(closure().item())
    _set_params(model, base, d1, 0.0, d2, 0.0)
    return {
        "grid": grid.tolist(),
        "loss_surface": Z.tolist(),
        "center_loss": float(Z[n // 2, n // 2]),
        "min_loss": float(Z.min()),
        "max_loss": float(Z.max()),
    }


# --------------------------------------------------------------------------- #
# Orchestration helper
# --------------------------------------------------------------------------- #
def analyse_all(fb: FeatureBundle, qubit_range=(3, 4, 5), seed: int = 0) -> Dict:
    """Run every landscape / barren-plateau analysis for the checkpoint suite."""
    out: Dict = {}
    out["barren_plateau"] = barren_plateau_scan(qubit_range, seed=seed)

    out["hessian"] = {}
    out["loss_slice_1d"] = {}
    out["loss_surface_2d"] = {}

    # classical baseline
    try:
        cmodel, _ = models.load_classical()
        out["hessian"]["classical"] = hessian_spectrum(cmodel, fb, seed=seed)
        out["loss_slice_1d"]["classical"] = loss_slice_1d(cmodel, fb, seed=seed)
        out["loss_surface_2d"]["classical"] = loss_surface_2d(cmodel, fb, seed=seed)
    except Exception as exc:  # pragma: no cover
        out["hessian"]["classical"] = {"error": str(exc)}

    for q in qubit_range:
        key = f"{q}q"
        try:
            model, _ = models.load_qapinn(q)
            out["hessian"][key] = hessian_spectrum(model, fb, seed=seed)
            out["loss_slice_1d"][key] = loss_slice_1d(model, fb, seed=seed)
            out["loss_surface_2d"][key] = loss_surface_2d(model, fb, seed=seed)
        except Exception as exc:  # pragma: no cover
            out["hessian"][key] = {"error": str(exc)}
    return out
