"""Lightweight Q1 Galerkin FEM for 2D coupled Burgers.

Discretisation:
  * Bilinear quadrilateral (Q1) elements on a structured grid.
  * 2x2 Gauss-Legendre quadrature (exact for the bilinear mass and
    stiffness integrands).
  * Lumped mass matrix, which turns the semi-discrete system into an
    explicit ODE and avoids a linear solve per step. Row-sum lumping on
    Q1 elements is standard and preserves the constant mode.
  * Convection assembled at nodes from the Galerkin-projected gradient,
    stabilised by the physical viscosity plus a small streamline
    diffusion term (SUPG-like) sized to the element Peclet number.

Temporal: explicit RK2 (midpoint).
"""
import numpy as np
import burgers2d_common as B

# 2x2 Gauss points on the reference element [-1,1]^2.
_G = 1.0 / np.sqrt(3.0)
_GP = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]
_GW = [1.0, 1.0, 1.0, 1.0]


def _q1_shape(xi, eta):
    """Q1 shape functions and reference-space derivatives."""
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


def _assemble(nx, ny, dx, dy):
    """Assemble lumped mass and the global stiffness (diffusion) matrix.

    Returns (M_lump, K) with K stored densely in COO-like index arrays
    for a structured grid; for the resolutions used here (<= 81x81) the
    sparse assembly below stays well within memory.
    """
    from scipy.sparse import coo_matrix

    nn_ = nx * ny
    nid = np.arange(nn_).reshape(nx, ny)

    rows, cols, vals = [], [], []
    M = np.zeros(nn_)

    # Element geometry is identical for every cell on a uniform grid,
    # so the element matrices are computed once.
    Ke = np.zeros((4, 4))
    Me = np.zeros(4)
    detJ = 0.25 * dx * dy
    for (xi, eta), w in zip(_GP, _GW):
        N, dN_dxi, dN_deta = _q1_shape(xi, eta)
        dN_dx = dN_dxi * (2.0 / dx)
        dN_dy = dN_deta * (2.0 / dy)
        Ke += w * detJ * (np.outer(dN_dx, dN_dx) + np.outer(dN_dy, dN_dy))
        Me += w * detJ * N

    for i in range(nx - 1):
        for j in range(ny - 1):
            e = [nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]]
            M[e] += Me
            for a in range(4):
                for b in range(4):
                    rows.append(e[a]); cols.append(e[b]); vals.append(Ke[a, b])

    K = coo_matrix((vals, (rows, cols)), shape=(nn_, nn_)).tocsr()
    return M, K, nid


def solve_fem(nx=81, ny=81, t_end=1.0, nu=B.NU_DEFAULT, dt=None,
              save_times=(0.0, 0.25, 0.5, 0.75, 1.0), supg=0.5):
    """Return (x, y, snapshots) with snapshots[t] = (u, v)."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    M, K, nid = _assemble(nx, ny, dx, dy)
    Minv = 1.0 / M

    u2d, v2d = B.exact_uv(X, Y, np.zeros_like(X), nu)
    u, v = u2d.ravel().copy(), v2d.ravel().copy()

    umax = max(np.abs(u).max(), np.abs(v).max(), 1e-8)
    if dt is None:
        dt = 0.3 * min(min(dx, dy) / umax, 0.25 * min(dx, dy)**2 / nu)
    nt = int(np.ceil(t_end / dt))
    dt = t_end / nt

    # Streamline diffusion coefficient from the element Peclet number.
    h = min(dx, dy)
    nu_art = supg * h * umax * max(0.0, 1.0 - 2.0 * nu / (h * umax + 1e-12))
    nu_eff = nu + nu_art

    # Boundary node indices.
    bmask = np.zeros((nx, ny), dtype=bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    bidx = nid[bmask]

    def grad_nodal(f):
        """Central-difference nodal gradient (Galerkin-consistent on a
        uniform Q1 grid up to boundary one-sided stencils)."""
        F = f.reshape(nx, ny)
        gx = np.gradient(F, dx, axis=0)
        gy = np.gradient(F, dy, axis=1)
        return gx.ravel(), gy.ravel()

    def rhs(u, v):
        ux, uy = grad_nodal(u)
        vx, vy = grad_nodal(v)
        du = Minv * (-M * (u * ux + v * uy) - nu_eff * (K @ u))
        dv = Minv * (-M * (u * vx + v * vy) - nu_eff * (K @ v))
        return du, dv

    snaps, targets = {}, sorted(save_times)
    if targets and targets[0] == 0.0:
        snaps[0.0] = (u.reshape(nx, ny).copy(), v.reshape(nx, ny).copy())
        targets = targets[1:]

    t = 0.0
    for _ in range(nt):
        k1u, k1v = rhs(u, v)
        um, vm = u + 0.5 * dt * k1u, v + 0.5 * dt * k1v
        k2u, k2v = rhs(um, vm)
        u = u + dt * k2u
        v = v + dt * k2v
        t += dt

        ub, vb = B.exact_uv(X, Y, np.full_like(X, t), nu)
        u[bidx] = ub.ravel()[bidx]
        v[bidx] = vb.ravel()[bidx]

        while targets and t >= targets[0] - 0.5 * dt:
            snaps[targets[0]] = (u.reshape(nx, ny).copy(),
                                 v.reshape(nx, ny).copy())
            targets = targets[1:]

    return x, y, snaps
