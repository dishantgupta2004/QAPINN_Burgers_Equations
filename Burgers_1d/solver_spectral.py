"""Fourier pseudo-spectral solver for u_t + u u_x = nu u_xx on periodic [-1,1).
Highest-accuracy reference (spectral convergence). Used as PRIMARY ground truth."""
import numpy as np
from scipy.integrate import solve_ivp
from config import NU, X_MIN, X_MAX, u0_fn

def solve_spectral(nx=1024, nt=201, nu=NU, t_max=1.0, rtol=1e-10, atol=1e-12):
    L = X_MAX - X_MIN
    x = X_MIN + L * np.arange(nx) / nx          # periodic grid, excludes endpoint
    k = 2*np.pi*np.fft.fftfreq(nx, d=L/nx)
    k2 = k**2
    u0 = u0_fn(x)

    def rhs(t, u):
        uh  = np.fft.fft(u)
        ux  = np.real(np.fft.ifft(1j*k*uh))
        uxx = np.real(np.fft.ifft(-k2*uh))
        return -u*ux + nu*uxx

    t_eval = np.linspace(0.0, t_max, nt)
    sol = solve_ivp(rhs, (0.0, t_max), u0, t_eval=t_eval,
                    method="Radau", rtol=rtol, atol=atol)
    return x, t_eval, sol.y.T            # U shape (nt, nx)

if __name__ == "__main__":
    x, t, U = solve_spectral()
    np.savez("checkpoints/gt_spectral.npz", x=x, t=t, U=U)
    print("spectral:", U.shape, "u range", U.min(), U.max())
