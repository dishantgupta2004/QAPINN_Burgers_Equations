import os
import numpy
import treelog
from nutils import mesh, function, solver, export, matrix
from nutils.expression_v2 import Namespace

L      = 2.2       # channel length            [m]
H      = 0.41      # channel height            [m]
XC, YC = 0.2, 0.2  # cylinder centre           [m]
R      = 0.05      # cylinder radius (D=0.1)   [m]
RHO    = 1.0       # density                   [kg/m^3]
NU     = 1e-3      # kinematic viscosity       [m^2/s]
UM     = 0.3       # peak inlet velocity       [m/s]   (2D-1 steady case)

UBAR = 2.0 / 3.0 * UM          # mean inlet velocity
RE   = UBAR * (2 * R) / NU     # Reynolds number based on diameter
DEGREE = 2                     # velocity poly degree (pressure = degree-1)


def solve_cylinder(nx=110, ny=24, maxrefine=2, degree=DEGREE, verbose=True):
    xverts = numpy.linspace(0, L, nx + 1)
    yverts = numpy.linspace(0, H, ny + 1)
    domain, geom = mesh.rectilinear([xverts, yverts])
    domain = domain.withboundary(inlet='left', outlet='right', wall='top,bottom')

    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='grad', normal='n', jacobians=('dV', 'dS'))
    ns.delta = function.eye(domain.ndims)
    ns.xc = numpy.array([XC, YC])
    ns.R = R
    ns.hmin = L / nx      
    ns.tau = 0.01            
    ns.rho = RHO
    ns.nu = NU
    ns.H = H
    ns.Um = UM

    levelset = 'sqrt((x_i - xc_i) (x_i - xc_i)) - R' @ ns
    domain = domain.trim(levelset, maxrefine=maxrefine, name='cylinder')

    ubasis = domain.basis('std', degree=degree)
    pbasis = domain.basis('std', degree=degree - 1)
    ns.u = function.field('u', ubasis, shape=(domain.ndims,))
    ns.p = function.field('p', pbasis)
    ns.v = function.field('v', ubasis, shape=(domain.ndims,))
    ns.q = function.field('q', pbasis)
    ns.sigma_ij = 'nu (grad_j(u_i) + grad_i(u_j)) - (p / rho) delta_ij'

    ns.uin_i = '(4 Um x_1 (H - x_1) / H^2) delta_i0'

    sqr  = domain.boundary['inlet'].integral('(u_i - uin_i) (u_i - uin_i) dS' @ ns, degree=2 * degree)
    sqr += domain.boundary['wall'].integral('u_i u_i dS' @ ns, degree=2 * degree)
    sqr += domain.boundary['cylinder'].integral('u_i u_i dS' @ ns, degree=2 * degree)
    ucons = solver.optimize('u', sqr, droptol=1e-12)
    psqr  = domain.boundary['outlet'].integral('p^2 dS' @ ns, degree=2 * degree)
    pcons = solver.optimize('p', psqr, droptol=1e-12)
    dV = function.jacobian(geom, domain.ndims)
    uorphan = domain.integral(ubasis * dV, degree=2 * degree).eval() < 1e-13
    porphan = domain.integral(pbasis * dV, degree=2 * degree).eval() < 1e-13
    ucons = ucons.reshape(-1, domain.ndims)
    ucons[uorphan] = 0.0
    ucons = ucons.ravel()
    pcons = numpy.where(porphan, 0.0, pcons)

    cons = dict(u=ucons, p=pcons)

    virtualwork = domain.integral(
        '(v_i (u_j grad_j(u_i)) + grad_j(v_i) sigma_ij '
        '+ q grad_i(u_i) - tau hmin^2 grad_i(q) grad_i(p)) dV' @ ns,
        degree=2 * degree + 1)
    res_u = virtualwork.derivative('v')
    res_p = virtualwork.derivative('q')

    if verbose:
        treelog.info(f'Re = {RE:.1f}, dofs u={len(ubasis)} p={len(pbasis)}')
    args = solver.newton(('u', 'p'), (res_u, res_p),
                         constrain=cons).solve(tol=1e-10, maxiter=25)

    return domain, ns, args


def sample_solution(domain, ns, args, seed=0):
    bezier = domain.sample('bezier', 3)
    xy   = bezier.eval('x_i' @ ns, arguments=args)
    uv   = bezier.eval('u_i' @ ns, arguments=args)
    pmag = bezier.eval('p'   @ ns, arguments=args)
    umag = numpy.linalg.norm(uv, axis=1)

    return dict(tri=bezier.tri, hull=bezier.hull,
                xy=xy, uv=uv, p=pmag, umag=umag)


def main(nx=110, ny=24, maxrefine=2, outdir='gt_out'):
    os.makedirs(outdir, exist_ok=True)
    domain, ns, args = solve_cylinder(nx=nx, ny=ny, maxrefine=maxrefine)
    S = sample_solution(domain, ns, args)

    npz = os.path.join(outdir, 'cylinder_gt.npz')
    numpy.savez(
        npz,
        x=S['xy'][:, 0], y=S['xy'][:, 1],
        u=S['uv'][:, 0], v=S['uv'][:, 1], p=S['p'],
        L=L, H=H, xc=XC, yc=YC, R=R, rho=RHO, nu=NU, Um=UM, Re=RE,
    )
    treelog.info(f'wrote {npz}  ({len(S["p"])} points)')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), constrained_layout=True)
        for ax, field, title, cmap in [
            (axes[0], S['umag'], '|u|  velocity magnitude', 'viridis'),
            (axes[1], S['p'],    'p  pressure',             'coolwarm')]:
            tpc = ax.tripcolor(S['xy'][:, 0], S['xy'][:, 1], S['tri'], field,
                               shading='gouraud', cmap=cmap)
            ax.set_aspect('equal'); ax.set_title(title)
            ax.set_xlim(0, L); ax.set_ylim(0, H)
            fig.colorbar(tpc, ax=ax, fraction=0.02)
        png = os.path.join(outdir, 'cylinder_gt.png')
        fig.savefig(png, dpi=130); plt.close(fig)
        treelog.info(f'wrote {png}')
    except Exception as e:
        treelog.info(f'(figure skipped: {e})')

    return dict(Re=RE, npoints=len(S['p']),
                umax=float(S['umag'].max()), pmax=float(S['p'].max()))


if __name__ == '__main__':
    import treelog as _tl
    with _tl.set(_tl.StdoutLog()):
        summary = main()
    print('SUMMARY:', summary)