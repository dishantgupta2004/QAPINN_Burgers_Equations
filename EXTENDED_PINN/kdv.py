# kdv.py
"""
KdV equation PINN / XPINN comparative analysis module.

PDE:  u_t + u*u_x = 0.0025*u_xxx
Domain: x in [-1, 1], t in [0, 1]
IC: u(x,0) = cos(pi*x)
BC: periodic
"""

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
NU = 0.0025
INTERFACE_X = -0.74


# ----------------------------------------------------------------------
# High-fidelity reference solution: Fourier pseudo-spectral + ETDRK4
def kdv_exact(nx=512, nt=2001, nu=NU, t_max=T_MAX, save_every=1):
    """
    Integrate u_t = -u*u_x + nu*u_xxx on x in [-1,1) with periodic BCs
    using a Fourier pseudo-spectral method with ETDRK4 exponential
    time differencing. Returns (x, t, U) with U of shape (nt_saved, nx).
    """
    L = X_MAX - X_MIN
    x = X_MIN + L * np.arange(nx) / nx
    u0 = np.cos(np.pi * x)

    k = 2.0 * np.pi / L * np.fft.fftfreq(nx, d=1.0 / nx)
    # linear operator: nu*u_xxx  ->  nu*(i k)^3 = -i*nu*k^3
    Lop = -1j * nu * k ** 3

    dt = t_max / (nt - 1)
    E = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2.0)

    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))

    g = -0.5j * k

    def nonlin(vhat):
        u = np.real(np.fft.ifft(vhat))
        return g * np.fft.fft(u ** 2)

    v = np.fft.fft(u0)
    saved = [u0.copy()]
    times = [0.0]

    for n in range(1, nt):
        Nv = nonlin(v)
        a = E2 * v + Q * Nv
        Na = nonlin(a)
        b = E2 * v + Q * Na
        Nb = nonlin(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        if n % save_every == 0:
            saved.append(np.real(np.fft.ifft(v)))
            times.append(n * dt)

    U = np.array(saved)
    t = np.array(times)

    # append periodic endpoint x = 1 for plotting convenience
    x_full = np.concatenate([x, [X_MAX]])
    U_full = np.concatenate([U, U[:, :1]], axis=1)
    return x_full, t, U_full


def build_reference_grid(nx=512, nt=2001, save_every=10):
    x, t, U = kdv_exact(nx=nx, nt=nt, save_every=save_every)
    return {"x": x, "t": t, "U": U}


def interpolate_reference(ref, X, T):
    """Bilinear interpolation of the reference solution at scattered (X, T)."""
    x, t, U = ref["x"], ref["t"], ref["U"]
    X = np.asarray(X).ravel()
    T = np.asarray(T).ravel()

    ix = np.clip(np.searchsorted(x, X) - 1, 0, len(x) - 2)
    it = np.clip(np.searchsorted(t, T) - 1, 0, len(t) - 2)

    x0, x1 = x[ix], x[ix + 1]
    t0, t1 = t[it], t[it + 1]
    wx = (X - x0) / (x1 - x0)
    wt = (T - t0) / (t1 - t0)

    u00 = U[it, ix]
    u01 = U[it, ix + 1]
    u10 = U[it + 1, ix]
    u11 = U[it + 1, ix + 1]

    return ((1 - wt) * ((1 - wx) * u00 + wx * u01)
            + wt * ((1 - wx) * u10 + wx * u11)).reshape(-1, 1)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def _uniform(n, lo, hi, rng):
    return rng.uniform(lo, hi, size=(n, 1))


def sample_residual(n, x_lo=X_MIN, x_hi=X_MAX, rng=None):
    rng = rng or np.random.default_rng(0)
    x = _uniform(n, x_lo, x_hi, rng)
    t = _uniform(n, T_MIN, T_MAX, rng)
    return np.hstack([x, t]).astype(np.float64)


def sample_boundary(n, x_lo=X_MIN, x_hi=X_MAX, rng=None, ic_frac=0.5):
    """
    Boundary set = initial condition points on [x_lo, x_hi] plus periodic
    wall points at x = -1 and x = +1 (only for subdomains touching walls).
    Returns (XT, u) where u is the prescribed value (IC) or NaN for
    periodic-pair rows handled separately.
    """
    rng = rng or np.random.default_rng(1)
    n_ic = int(round(n * ic_frac))
    n_wall = n - n_ic

    x_ic = _uniform(n_ic, x_lo, x_hi, rng)
    t_ic = np.zeros_like(x_ic)
    u_ic = np.cos(np.pi * x_ic)

    XT = np.hstack([x_ic, t_ic])
    U = u_ic

    if n_wall > 0:
        touches_left = np.isclose(x_lo, X_MIN)
        touches_right = np.isclose(x_hi, X_MAX)
        walls = []
        if touches_left:
            walls.append(X_MIN)
        if touches_right:
            walls.append(X_MAX)
        if walls:
            per = n_wall // len(walls)
            for w in walls:
                t_w = _uniform(per, T_MIN, T_MAX, rng)
                x_w = np.full_like(t_w, w)
                XT = np.vstack([XT, np.hstack([x_w, t_w])])
                U = np.vstack([U, np.full_like(t_w, np.nan)])
        else:
            x_extra = _uniform(n_wall, x_lo, x_hi, rng)
            t_extra = np.zeros_like(x_extra)
            XT = np.vstack([XT, np.hstack([x_extra, t_extra])])
            U = np.vstack([U, np.cos(np.pi * x_extra)])

    return XT.astype(np.float64), U.astype(np.float64)


def sample_periodic_pairs(n, rng=None):
    rng = rng or np.random.default_rng(2)
    t = _uniform(n, T_MIN, T_MAX, rng)
    left = np.hstack([np.full_like(t, X_MIN), t])
    right = np.hstack([np.full_like(t, X_MAX), t])
    return left.astype(np.float64), right.astype(np.float64)


def sample_interface(n, x_i=INTERFACE_X, rng=None):
    rng = rng or np.random.default_rng(3)
    t = _uniform(n, T_MIN, T_MAX, rng)
    x = np.full_like(t, x_i)
    return np.hstack([x, t]).astype(np.float64)


def build_pinn_dataset(n_res=18000, n_bnd=914, n_periodic=1000, seed=0):
    rng = np.random.default_rng(seed)
    Xr = sample_residual(n_res, rng=rng)
    Xb, Ub = sample_boundary(n_bnd, rng=rng)
    PL, PR = sample_periodic_pairs(n_periodic, rng=rng)
    mask = ~np.isnan(Ub).ravel()
    return {
        "Xr": Xr,
        "Xb": Xb[mask],
        "Ub": Ub[mask],
        "PL": PL,
        "PR": PR,
    }


def build_xpinn_dataset(n_res1=14000, n_bnd1=646,
                        n_res2=4000, n_bnd2=268,
                        n_iface=10000, n_periodic=1000,
                        x_i=INTERFACE_X, seed=0):
    rng = np.random.default_rng(seed)

    Xr1 = sample_residual(n_res1, x_lo=x_i, x_hi=X_MAX, rng=rng)
    Xb1, Ub1 = sample_boundary(n_bnd1, x_lo=x_i, x_hi=X_MAX, rng=rng)
    m1 = ~np.isnan(Ub1).ravel()

    Xr2 = sample_residual(n_res2, x_lo=X_MIN, x_hi=x_i, rng=rng)
    Xb2, Ub2 = sample_boundary(n_bnd2, x_lo=X_MIN, x_hi=x_i, rng=rng)
    m2 = ~np.isnan(Ub2).ravel()

    XI = sample_interface(n_iface, x_i=x_i, rng=rng)
    PL, PR = sample_periodic_pairs(n_periodic, rng=rng)

    return {
        "Xr1": Xr1, "Xb1": Xb1[m1], "Ub1": Ub1[m1],
        "Xr2": Xr2, "Xb2": Xb2[m2], "Ub2": Ub2[m2],
        "XI": XI, "PL": PL, "PR": PR,
    }


# ----------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class MLP(nn.Module):
    """
    depth = total number of Linear layers (10 => 9 hidden layers + output).
    """

    def __init__(self, in_dim=2, out_dim=1, width=20, depth=10, w0=1.0,
                 lb=(X_MIN, T_MIN), ub=(X_MAX, T_MAX)):
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))

        layers = []
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(Sine(w0))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xt):
        z = 2.0 * (xt - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(z)


def to_tensor(a, requires_grad=False):
    t = torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)
    t.requires_grad_(requires_grad)
    return t


def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True)[0]


def kdv_residual(model, xt, nu=NU):
    xt = xt.clone().requires_grad_(True)
    u = model(xt)
    du = grad(u, xt)
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = grad(u_x, xt)[:, 0:1]
    u_xxx = grad(u_xx, xt)[:, 0:1]
    return u_t + u * u_x - nu * u_xxx


def residual_and_value(model, xt, nu=NU):
    xt = xt.clone().requires_grad_(True)
    u = model(xt)
    du = grad(u, xt)
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = grad(u_x, xt)[:, 0:1]
    u_xxx = grad(u_xx, xt)[:, 0:1]
    return u, u_t + u * u_x - nu * u_xxx


# ----------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------
def train_pinn(data, width=20, depth=10, lr=1e-3, epochs=5000,
               w_res=1.0, w_bnd=1.0, w_per=1.0, seed=0, log_every=500):
    torch.manual_seed(seed)
    model = MLP(width=width, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xr = to_tensor(data["Xr"])
    Xb = to_tensor(data["Xb"])
    Ub = to_tensor(data["Ub"])
    PL = to_tensor(data["PL"])
    PR = to_tensor(data["PR"])

    history = []
    for ep in range(epochs):
        opt.zero_grad()
        r = kdv_residual(model, Xr)
        loss_r = torch.mean(r ** 2)
        loss_b = torch.mean((model(Xb) - Ub) ** 2)
        loss_p = torch.mean((model(PL) - model(PR)) ** 2)
        loss = w_res * loss_r + w_bnd * loss_b + w_per * loss_p
        loss.backward()
        opt.step()

        history.append([loss.item(), loss_r.item(), loss_b.item(), loss_p.item()])
        if log_every and ep % log_every == 0:
            print(f"[PINN] ep {ep:5d} | total {loss.item():.4e} "
                  f"| res {loss_r.item():.4e} | bnd {loss_b.item():.4e} "
                  f"| per {loss_p.item():.4e}")

    return model, np.array(history)


def train_xpinn(data, width=20, depth=10, lr=1e-3, epochs=5000,
                w_res=1.0, w_bnd=1.0, w_iface=1.0, w_rescont=0.0,
                w_per=1.0, x_i=INTERFACE_X, seed=0, log_every=500):
    torch.manual_seed(seed)
    net1 = MLP(width=width, depth=depth, lb=(x_i, T_MIN), ub=(X_MAX, T_MAX)).to(DEVICE)
    net2 = MLP(width=width, depth=depth, lb=(X_MIN, T_MIN), ub=(x_i, T_MAX)).to(DEVICE)
    opt = torch.optim.Adam(list(net1.parameters()) + list(net2.parameters()), lr=lr)

    Xr1, Xb1, Ub1 = to_tensor(data["Xr1"]), to_tensor(data["Xb1"]), to_tensor(data["Ub1"])
    Xr2, Xb2, Ub2 = to_tensor(data["Xr2"]), to_tensor(data["Xb2"]), to_tensor(data["Ub2"])
    XI = to_tensor(data["XI"])
    PL, PR = to_tensor(data["PL"]), to_tensor(data["PR"])

    history = []
    for ep in range(epochs):
        opt.zero_grad()

        r1 = kdv_residual(net1, Xr1)
        r2 = kdv_residual(net2, Xr2)
        loss_r = torch.mean(r1 ** 2) + torch.mean(r2 ** 2)

        loss_b = torch.mean((net1(Xb1) - Ub1) ** 2) + torch.mean((net2(Xb2) - Ub2) ** 2)

        u1_i, res1_i = residual_and_value(net1, XI)
        u2_i, res2_i = residual_and_value(net2, XI)
        u_avg = 0.5 * (u1_i + u2_i)
        loss_i = torch.mean((u1_i - u_avg) ** 2) + torch.mean((u2_i - u_avg) ** 2)
        loss_rc = torch.mean((res1_i - res2_i) ** 2)

        loss_p = torch.mean((net2(PL) - net1(PR)) ** 2)

        loss = (w_res * loss_r + w_bnd * loss_b
                + w_iface * loss_i + w_rescont * loss_rc + w_per * loss_p)
        loss.backward()
        opt.step()

        history.append([loss.item(), loss_r.item(), loss_b.item(),
                        loss_i.item(), loss_rc.item(), loss_p.item()])
        if log_every and ep % log_every == 0:
            print(f"[XPINN] ep {ep:5d} | total {loss.item():.4e} "
                  f"| res {loss_r.item():.4e} | bnd {loss_b.item():.4e} "
                  f"| iface {loss_i.item():.4e} | per {loss_p.item():.4e}")

    return (net1, net2), np.array(history)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_pinn(model, XT, batch=100000):
    out = []
    for i in range(0, len(XT), batch):
        out.append(model(to_tensor(XT[i:i + batch])).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def predict_xpinn(nets, XT, x_i=INTERFACE_X, batch=100000):
    net1, net2 = nets
    XT = np.asarray(XT)
    pred = np.zeros((len(XT), 1))
    m1 = XT[:, 0] > x_i
    m2 = ~m1
    for m, net in ((m1, net1), (m2, net2)):
        if m.sum() == 0:
            continue
        sub = XT[m]
        chunks = []
        for i in range(0, len(sub), batch):
            chunks.append(net(to_tensor(sub[i:i + batch])).cpu().numpy())
        pred[m] = np.vstack(chunks)
    return pred


def test_grid(nx=320, nt=320):
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    XX, TT = np.meshgrid(x, t)
    XT = np.hstack([XX.reshape(-1, 1), TT.reshape(-1, 1)])
    return x, t, XX, TT, XT


def relative_l2(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def spectral_complexity(model):
    """Product of spectral norms times ((sum of (2,1)/spectral ratios)^{2/3})^{3/2}."""
    prod = 1.0
    ratios = 0.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy()
            s = np.linalg.norm(W, 2)
            n21 = np.sum(np.linalg.norm(W, axis=1))
            prod *= s
            ratios += (n21 / s) ** (2.0 / 3.0)
    return prod * ratios ** 1.5