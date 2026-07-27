import numpy as np, torch
from config import NU, X_MIN, X_MAX, T_MIN, T_MAX, u0_fn, DEVICE

def sample_interior(n, device=DEVICE, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n,1, generator=g)*(X_MAX-X_MIN)+X_MIN
    t = torch.rand(n,1, generator=g)*(T_MAX-T_MIN)+T_MIN
    return x.to(device).requires_grad_(True), t.to(device).requires_grad_(True)

def sample_shock_biased(n, device=DEVICE, seed=0, frac=0.4, width=0.15):
    """Half uniform, half concentrated near x=0 (shock location) for t>0.3."""
    g = torch.Generator().manual_seed(seed)
    n_s = int(n*frac); n_u = n-n_s
    xu = torch.rand(n_u,1, generator=g)*(X_MAX-X_MIN)+X_MIN
    tu = torch.rand(n_u,1, generator=g)
    xs = torch.randn(n_s,1, generator=g)*width
    xs = xs.clamp(X_MIN, X_MAX)
    ts = 0.3 + torch.rand(n_s,1, generator=g)*0.7
    x = torch.cat([xu,xs]); t = torch.cat([tu,ts])
    return x.to(device).requires_grad_(True), t.to(device).requires_grad_(True)

def sample_ic(n, device=DEVICE, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n,1, generator=g)*(X_MAX-X_MIN)+X_MIN
    t = torch.zeros_like(x)
    u = torch.from_numpy(u0_fn(x.numpy())).float()
    return x.to(device), t.to(device), u.to(device)

def sample_bc(n, device=DEVICE, seed=2):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n,1, generator=g)*(T_MAX-T_MIN)+T_MIN
    return (torch.full_like(t, X_MAX).to(device),
            torch.full_like(t, X_MIN).to(device), t.to(device))

def pde_residual(model, x, t, nu=NU):
    u   = model(torch.cat([x,t], 1))
    ut  = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    ux  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut + u*ux - nu*uxx

def build_batches(n_pde=8000, n_ic=512, n_bc=256, seed=0, shock_biased=True):
    sampler = sample_shock_biased if shock_biased else sample_interior
    x_f, t_f = sampler(n_pde, seed=seed)
    x0, t0, u0 = sample_ic(n_ic, seed=seed+1)
    xp, xn, tb = sample_bc(n_bc, seed=seed+2)
    return dict(x_f=x_f, t_f=t_f, x0=x0, t0=t0, u0=u0,
                xb_pos=xp, xb_neg=xn, tb=tb)

W = dict(pde=1.0, ic=20.0, bc=20.0)

def composite_loss(model, B, w=W):
    r  = pde_residual(model, B["x_f"], B["t_f"])
    lp = (r**2).mean()
    li = ((model(torch.cat([B["x0"],B["t0"]],1)) - B["u0"])**2).mean()
    lb = ((model(torch.cat([B["xb_pos"],B["tb"]],1))**2).mean()
        + (model(torch.cat([B["xb_neg"],B["tb"]],1))**2).mean())
    tot = w["pde"]*lp + w["ic"]*li + w["bc"]*lb
    return tot, {"total": tot.item(), "pde": lp.item(),
                 "ic": li.item(), "bc": lb.item()}
