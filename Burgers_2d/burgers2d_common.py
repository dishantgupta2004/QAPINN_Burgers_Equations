import torch, numpy as np

NU_DEFAULT = 0.01

# ---------- exact solution ----------
def exact_uv(x, y, t, nu=NU_DEFAULT):
    """Analytic solution of 2D coupled Burgers. Accepts torch tensors or np arrays."""
    if isinstance(x, torch.Tensor):
        arg = (-4.0*x + 4.0*y - t) / (32.0*nu)
        s = 1.0 / (1.0 + torch.exp(arg))
    else:
        arg = (-4.0*x + 4.0*y - t) / (32.0*nu)
        s = 1.0 / (1.0 + np.exp(arg))
    u = 0.75 - 0.25*s
    v = 0.75 + 0.25*s
    return u, v

# ---------- collocation sampling ----------
def sample_interior(n, device, x_lo=0., x_hi=1., y_lo=0., y_hi=1., t_hi=1.0):
    r = torch.rand(n, 3, device=device)
    x = x_lo + (x_hi-x_lo)*r[:, 0:1]
    y = y_lo + (y_hi-y_lo)*r[:, 1:2]
    t = t_hi*r[:, 2:3]
    return x, y, t

def sample_initial(n, device, nu=NU_DEFAULT):
    r = torch.rand(n, 2, device=device)
    x, y = r[:, 0:1], r[:, 1:2]
    t = torch.zeros_like(x)
    u, v = exact_uv(x, y, t, nu)
    return x, y, t, u, v

def sample_boundary(n, device, nu=NU_DEFAULT, t_hi=1.0):
    """n points per edge, 4 edges."""
    xs, ys = [], []
    r = torch.rand(n, 1, device=device)
    o = torch.ones(n, 1, device=device); z = torch.zeros(n, 1, device=device)
    for (xa, ya) in [(z, r), (o, r), (r, z), (r, o)]:
        xs.append(xa); ys.append(ya)
    x = torch.cat(xs); y = torch.cat(ys)
    t = t_hi*torch.rand(4*n, 1, device=device)
    u, v = exact_uv(x, y, t, nu)
    return x, y, t, u, v

# ---------- PDE residual ----------
def pde_residual(model, x, y, t, nu=NU_DEFAULT):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    out = model(torch.cat([x, y, t], dim=1))
    u, v = out[:, 0:1], out[:, 1:2]

    g = lambda f, w: torch.autograd.grad(
        f, w, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    u_x, u_y, u_t = g(u, x), g(u, y), g(u, t)
    v_x, v_y, v_t = g(v, x), g(v, y), g(v, t)
    u_xx, u_yy = g(u_x, x), g(u_y, y)
    v_xx, v_yy = g(v_x, x), g(v_y, y)

    r_u = u_t + u*u_x + v*u_y - nu*(u_xx + u_yy)
    r_v = v_t + u*v_x + v*v_y - nu*(v_xx + v_yy)
    return r_u, r_v

# ---------- evaluation ----------
@torch.no_grad()
def eval_grid(model, t_eval, n=101, nu=NU_DEFAULT, device="cuda"):
    lin = torch.linspace(0, 1, n, device=device)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    T = torch.full_like(X, float(t_eval))
    inp = torch.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], dim=1)
    pred = model(inp)
    up, vp = pred[:, 0].reshape(n, n), pred[:, 1].reshape(n, n)
    ue, ve = exact_uv(X, Y, T, nu)
    return (X.cpu().numpy(), Y.cpu().numpy(),
            up.cpu().numpy(), vp.cpu().numpy(),
            ue.cpu().numpy(), ve.cpu().numpy())

def rel_l2(pred, exact):
    return float(np.linalg.norm(pred-exact) / np.linalg.norm(exact))
