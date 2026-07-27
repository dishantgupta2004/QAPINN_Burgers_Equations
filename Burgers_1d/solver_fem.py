"""P1 Galerkin FEM in space, Crank-Nicolson diffusion + explicit nonlinear
convection. Consistent mass matrix, Dirichlet BCs."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from config import NU, X_MIN, X_MAX, u0_fn

def assemble(nx, dx):
    N = nx
    M = sp.diags([np.ones(N-1), 4*np.ones(N), np.ones(N-1)], [-1,0,1],
                 format="lil")*(dx/6.0)
    M[0,0] = dx/3.0; M[-1,-1] = dx/3.0
    K = sp.diags([-np.ones(N-1), 2*np.ones(N), -np.ones(N-1)], [-1,0,1],
                 format="lil")/dx
    K[0,0] = 1/dx;  K[-1,-1] = 1/dx
    return M.tocsc(), K.tocsc()

def convection(u, dx):
    """Galerkin projection of u u_x = d/dx(u^2/2) using P1 basis."""
    f = 0.5*u**2
    c = np.zeros_like(u)
    c[1:-1] = (f[2:] - f[:-2])/2.0            # int phi_i d/dx(f) = -(f_{i+1}-f_{i-1})/2 (sign folded)
    c[0] = (f[1]-f[0])/2.0; c[-1] = (f[-1]-f[-2])/2.0
    return c

def solve_fem(nx=801, nt=4001, nu=NU, t_max=1.0):
    x = np.linspace(X_MIN, X_MAX, nx); dx = x[1]-x[0]
    t = np.linspace(0.0, t_max, nt);   dt = t[1]-t[0]
    M, K = assemble(nx, dx)
    A = (M + 0.5*dt*nu*K).tolil()
    A[0,:]=0; A[0,0]=1; A[-1,:]=0; A[-1,-1]=1
    A = A.tocsc(); lu = spla.splu(A)
    B = (M - 0.5*dt*nu*K).tocsc()

    U = np.zeros((nt, nx)); U[0] = u0_fn(x)
    u = U[0].copy()
    for n in range(1, nt):
        rhs = B.dot(u) - dt*convection(u, dx)
        rhs[0] = 0.0; rhs[-1] = 0.0
        u = lu.solve(rhs)
        U[n] = u
    return x, t, U

if __name__ == "__main__":
    x, t, U = solve_fem()
    np.savez("checkpoints/gt_fem.npz", x=x, t=t, U=U)
    print("fem:", U.shape, "u range", U.min(), U.max())
