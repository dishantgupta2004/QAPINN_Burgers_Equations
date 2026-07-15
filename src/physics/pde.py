from __future__ import annotations
from src.fenics_backend import fem, ufl, default_scalar_type


class BurgersPDE:
    def __init__(self, nu: float) -> None:
        self.nu = nu

    def residual(self, u, u_n, v, dt):
        mesh = u.function_space.mesh
        gdim = mesh.geometry.dim
        nu = fem.Constant(mesh, default_scalar_type(self.nu))
        advection = sum(u.dx(i) for i in range(gdim))
        F = ((u - u_n) / dt) * v * ufl.dx
        F += u * advection * v * ufl.dx
        F += nu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        return F
