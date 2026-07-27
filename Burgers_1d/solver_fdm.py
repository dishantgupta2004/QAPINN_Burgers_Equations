"""2nd-order central FDM in space + implicit Crank-Nicolson (diffusion) with
explicit-in-nonlinear-term (IMEX). Dirichlet u(+-1,t)=0."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from config import NU, X_MIN, X_MAX, u0_fn

def solve_fdm(nx=1001, nt=2001, nu=NU, t_max=1.0):
    x  = np.linspace(X_MIN, X_MAX, nx)
    dx = x[1]-x[0]
    t  = np.linspace(0.0, t_max, nt)
    dt = t[1]-t[0]
    N  = nx-2                                    # interior unknowns

    main = -2*np.ones(N); off = np.ones(N-1)
    Lap  = sp.diags([off, main, off], [-1,0,1], format="csc")/dx**2
    I    = sp.identity(N, format="csc")
    A    = (I - 0.5*dt*nu*Lap).tocsc()
    B    = (I + 0.5*dt*nu*Lap).tocsc()
    lu   = spla.splu(A)

    U = np.zeros((nt, nx)); U[0] = u0_fn(x)
    u = U[0,1:-1].copy()
    for n in range(1, nt):
        ufull = np.concatenate(([0.0], u, [0.0]))
        ux    = (ufull[2:] - ufull[:-2])/(2*dx)   # central
        rhs   = B.dot(u) - dt*u*ux                # explicit advection
        u     = lu.solve(rhs)
        U[n,1:-1] = u
    return x, t, U

if __name__ == "__main__":
    x, t, U = solve_fdm()
    np.savez("checkpoints/gt_fdm.npz", x=x, t=t, U=U)
    print("fdm:", U.shape, "u range", U.min(), U.max())
