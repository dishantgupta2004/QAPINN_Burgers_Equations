import numpy as np
import torch
from scipy import integrate

NU = 0.01 / np.pi        
X_MIN, X_MAX = -1.0, 1.0   
T_MIN, T_MAX = 0.0, 1.0   

def burgers_exact_point(x, t, nu=NU):
    if t <= 0.0:
        return -np.sin(np.pi * x)

    def _num(eta):
        return (np.sin(np.pi * (x - eta))
                * np.exp(-np.cos(np.pi * (x - eta)) / (2 * np.pi * nu))
                * np.exp(-eta ** 2 / (4 * nu * t)))

    def _den(eta):
        return (np.exp(-np.cos(np.pi * (x - eta)) / (2 * np.pi * nu))
                * np.exp(-eta ** 2 / (4 * nu * t)))

    num, _ = integrate.quad(_num, -np.inf, np.inf, limit=200)
    den, _ = integrate.quad(_den, -np.inf, np.inf, limit=200)
    return -num / den


def burgers_exact_grid(x_arr, t_scalar, nu=NU):
    return np.array([burgers_exact_point(float(x), float(t_scalar), nu) for x in x_arr])


def sample_interior(n, device="cpu", seed=None):
    """Random interior collocation points for the PDE residual."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.rand(n, 1, generator=g) * (X_MAX - X_MIN) + X_MIN
    t = torch.rand(n, 1, generator=g) * (T_MAX - T_MIN) + T_MIN
    return x.to(device), t.to(device)


def sample_ic(n, device="cpu", seed=None):
    """Initial-condition points: u(x,0) = -sin(pi x)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.rand(n, 1, generator=g) * (X_MAX - X_MIN) + X_MIN
    t = torch.zeros(n, 1)
    u = -torch.sin(np.pi * x)
    return x.to(device), t.to(device), u.to(device)


def sample_bc(n, device="cpu", seed=None):
    """Boundary points at x = +1 and x = -1 (Dirichlet u = 0)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    t = torch.rand(n, 1, generator=g) * (T_MAX - T_MIN) + T_MIN
    x_pos = torch.ones(n, 1)
    x_neg = -torch.ones(n, 1)
    return x_pos.to(device), x_neg.to(device), t.to(device)

def pde_residual(model, x, t, nu=NU):
    """r = u_t + u u_x - nu u_xx, computed via torch autograd.

    `model` maps a (N,2) tensor [x,t] -> (N,1) prediction u.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(torch.cat([x, t], dim=1))
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut + u * ux - nu * uxx

def relative_l2(pred, exact):
    pred = np.asarray(pred).ravel()
    exact = np.asarray(exact).ravel()
    return np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-12)


def evaluate_on_grid(model, t_values, nx=256, device="cpu"):
    model.eval()
    x_grid = np.linspace(X_MIN, X_MAX, nx)
    out = {}
    with torch.no_grad():
        for t in t_values:
            xt = torch.tensor(
                np.stack([x_grid, np.full_like(x_grid, t)], axis=1),
                dtype=torch.float32, device=device,
            )
            u_pred = model(xt).cpu().numpy().ravel()
            u_exact = burgers_exact_grid(x_grid, t)
            out[t] = (x_grid, u_pred, u_exact, relative_l2(u_pred, u_exact))
    return out
