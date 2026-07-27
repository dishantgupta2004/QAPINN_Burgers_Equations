"""Finite-difference solver for 2D coupled Burgers.

Spatial discretisation:
  * Diffusion: 2nd-order central differences.
  * Convection: 1st-order upwind, branch-selected on the local sign of
    the advecting velocity. Upwinding is what keeps the scheme stable at
    nu = 0.01, where the cell Peclet number exceeds 2 on any practical
    grid; central differencing on the convective term would oscillate.

Temporal: explicit Euler under the combined CFL/diffusion limit.
Boundaries: Dirichlet, set from the analytic solution each step.
"""
import numpy as np
import burgers2d_common as B


def solve_fdm(nx=101, ny=101, t_end=1.0, nu=B.NU_DEFAULT,
              cfl=0.4, save_times=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Return (x, y, snapshots) where snapshots[t] = (u, v)."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    u, v = B.exact_uv(X, Y, np.zeros_like(X), nu)
    u, v = u.copy(), v.copy()

    umax = max(np.abs(u).max(), np.abs(v).max(), 1e-8)
    dt_conv = cfl * min(dx, dy) / umax
    dt_diff = cfl * 0.25 * min(dx, dy) ** 2 / nu
    dt = min(dt_conv, dt_diff)
    nt = int(np.ceil(t_end / dt))
    dt = t_end / nt

    snaps, targets = {}, sorted(save_times)
    if targets and targets[0] == 0.0:
        snaps[0.0] = (u.copy(), v.copy())
        targets = targets[1:]

    def upwind(f, a, axis, h):
        """First-order upwind derivative of f advected by a."""
        fwd = (np.roll(f, -1, axis) - f) / h      # forward difference
        bwd = (f - np.roll(f, 1, axis)) / h       # backward difference
        return np.where(a > 0, bwd, fwd)

    def lap(f):
        return ((np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / dx**2 +
                (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / dy**2)

    t = 0.0
    for step in range(nt):
        u_x = upwind(u, u, 0, dx)
        u_y = upwind(u, v, 1, dy)
        v_x = upwind(v, u, 0, dx)
        v_y = upwind(v, v, 1, dy)

        u_new = u + dt * (-u * u_x - v * u_y + nu * lap(u))
        v_new = v + dt * (-u * v_x - v * v_y + nu * lap(v))

        t += dt
        # Dirichlet BCs from the analytic solution.
        ub, vb = B.exact_uv(X, Y, np.full_like(X, t), nu)
        for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
            u_new[sl] = ub[sl]
            v_new[sl] = vb[sl]

        u, v = u_new, v_new

        while targets and t >= targets[0] - 0.5 * dt:
            snaps[targets[0]] = (u.copy(), v.copy())
            targets = targets[1:]

    return x, y, snaps
