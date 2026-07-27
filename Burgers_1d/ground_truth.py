"""Unified ground-truth interface + cross-solver validation + interpolation."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from solver_spectral import solve_spectral
from solver_fdm import solve_fdm
from solver_fem import solve_fem

class GroundTruth:
    def __init__(self, method="spectral", **kw):
        self.method = method
        if   method == "spectral": x, t, U = solve_spectral(**kw)
        elif method == "fdm":      x, t, U = solve_fdm(**kw)
        elif method == "fem":      x, t, U = solve_fem(**kw)
        else: raise ValueError(method)
        # spectral grid is periodic -> append endpoint for interpolation
        if method == "spectral":
            x = np.append(x, 1.0); U = np.concatenate([U, U[:, :1]], axis=1)
        self.x, self.t, self.U = x, t, U
        self.interp = RegularGridInterpolator((t, x), U,
                        bounds_error=False, fill_value=None)

    def __call__(self, x, t):
        x = np.asarray(x).ravel(); t = np.asarray(t).ravel()
        return self.interp(np.stack([t, x], axis=1))

    def slice(self, t0, xq):
        return self(xq, np.full_like(np.asarray(xq, float), t0))

def rel_l2(pred, ref):
    return float(np.linalg.norm(pred-ref)/(np.linalg.norm(ref)+1e-30))

def cross_validate(nx=401, ts=(0.25,0.5,0.75,1.0)):
    gts = {m: GroundTruth(m) for m in ("spectral","fdm","fem")}
    xq  = np.linspace(-1,1,nx)
    print(f"{'t':>5} | {'FDM vs SPEC':>12} | {'FEM vs SPEC':>12}")
    out = {}
    for t0 in ts:
        ref = gts["spectral"].slice(t0, xq)
        e1  = rel_l2(gts["fdm"].slice(t0, xq), ref)
        e2  = rel_l2(gts["fem"].slice(t0, xq), ref)
        out[t0] = (e1, e2)
        print(f"{t0:>5.2f} | {e1:>12.3e} | {e2:>12.3e}")
    return gts, out
