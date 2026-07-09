"""
src/physics/pde.py
==================

The swappable physics: the UFL weak form of the viscous Burgers equation,
written once for any spatial dimension.

Swapping the PDE later means writing a sibling class with the same ``residual``
signature (e.g. ``HeatPDE`` drops the advection term). The solver depends only
on this interface, not on Burgers specifically.
"""

from __future__ import annotations

from src.fenics_backend import fem, ufl, default_scalar_type


class BurgersPDE:
    """Encapsulates the UFL weak form of the viscous Burgers equation.

    We solve the *scalar* Burgers equation for ``u(x, t)``, which generalizes to
    d dimensions as::

        u_t + u (u_x + u_y + ... )  -  nu * laplacian(u)  =  0

    The advection speed is the field value ``u`` itself, carried along every
    spatial axis — this is the multi-dimensional analogue that keeps the
    unknown scalar (so the exported dataset stays a clean ``[coords, t, u]``
    table). The diffusion term is the full Laplacian, so it is already
    dimension-agnostic once written with ``grad``.
    """

    def __init__(self, nu: float) -> None:
        self.nu = nu

    def residual(self, u, u_n, v, dt):
        """Backward-Euler weak residual F(u; v) = 0, valid in 1D/2D/3D.

        Strong form  : (u - u_n)/dt + u (sum_i u_{,i}) - nu * div(grad u) = 0
        Weak form    : integrate against v, integrate the viscous term by parts.

            F = ∫ (u - u_n)/dt · v dx
              + ∫ u (Σ_i ∂u/∂x_i) · v dx          (advection, non-conservative)
              + nu · ∫ grad(u)·grad(v) dx          (diffusion, after IBP)

        The natural boundary term nu · ∂u/∂n · v on ∂Ω vanishes for the
        do-nothing (homogeneous Neumann) case and is overridden by strongly
        imposed Dirichlet BCs.

        Parameters
        ----------
        u    : fem.Function     unknown at t^{n+1}
        u_n  : fem.Function     known solution at t^n
        v    : ufl.Argument     test function
        dt   : fem.Constant     time-step size
        """
        mesh = u.function_space.mesh
        gdim = mesh.geometry.dim
        nu = fem.Constant(mesh, default_scalar_type(self.nu))

        # ufl.dx is the volume measure. ``u.dx(i)`` is ∂u/∂x_i; summing over the
        # active axes gives the multi-dimensional advection operator. In 1D this
        # reduces to the familiar ``u * u.dx(0)``.
        advection = sum(u.dx(i) for i in range(gdim))

        F = ((u - u_n) / dt) * v * ufl.dx
        F += u * advection * v * ufl.dx
        F += nu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        return F
