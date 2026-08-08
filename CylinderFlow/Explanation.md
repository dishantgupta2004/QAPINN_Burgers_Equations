# Flow Around a Circular Cylinder — Problem Analysis

**Stage 2 benchmark for the HQPINN project (Sedykh et al. 2024 progression: cylinder → Y-mixer)**

This document establishes the physical and mathematical foundations of the *flow around a circular cylinder* benchmark before any simulation or PINN code is written. It follows the same staged pattern used in the earlier Burgers and KdV projects: fix the physics and the sanity-check targets first, then build the independent numerical ground truth, then the classical PINN, then the quantum extension.

---

## 1. Why this benchmark

The cylinder is the single most widely used validation case in laminar incompressible CFD. It is attractive for the HQPINN program for three reasons:

1. **External flow with a curved no-slip boundary.** Unlike the analytic Burgers/KdV cases, there is no closed-form solution, so it forces a genuine mesh-based ground truth and exercises the same "independent solver as truth" discipline established in the earlier projects.
2. **Rich but controllable physics.** By choosing the Reynolds number we move continuously from a single steady recirculation bubble (Re ≈ 20) to periodic vortex shedding — the von Kármán vortex street (Re ≳ 50). We start in the steady regime, consistent with the laminar range (Re ≈ 10–40) confirmed in Stage 1.
3. **Standard reference quantities.** Drag coefficient, lift coefficient, recirculation-bubble length, and (in the unsteady case) the Strouhal number give quantitative validation targets beyond a raw L2 error.

---

## 2. Governing equations

The fluid is Newtonian, incompressible, and (for the first case) in steady state. Writing the velocity field as **u** = (u, v) and the pressure as p, the incompressible Navier–Stokes system in kinematic form is

**Momentum (2 equations):**

$$
(\mathbf{u}\cdot\nabla)\,\mathbf{u} \;=\; -\frac{1}{\rho}\nabla p \;+\; \nu\,\nabla^2 \mathbf{u}
$$

**Continuity (mass conservation, 1 equation):**

$$
\nabla\cdot\mathbf{u} \;=\; 0
$$

Expanded into scalar components in 2D, this is the familiar **4-PDE loss structure** carried over from Stage 1 (2 momentum residuals in the steady case + continuity; the transient case adds the ∂/∂t terms):

$$
u\,u_x + v\,u_y = -\tfrac{1}{\rho}\,p_x + \nu\,(u_{xx}+u_{yy})
$$
$$
u\,v_x + v\,v_y = -\tfrac{1}{\rho}\,p_x' + \nu\,(v_{xx}+v_{yy})
\quad\text{(with } p_y \text{ in the second momentum eqn)}
$$
$$
u_x + v_y = 0
$$

Here:

- **u(x,y), v(x,y)** — velocity components [m/s]
- **p(x,y)** — pressure [Pa] (we solve for p/ρ, the kinematic pressure)
- **ν** — kinematic viscosity [m²/s]; smaller ν ⇒ thinner boundary layers and sharper wake gradients
- **ρ** — density [kg/m³]
- The convective term (**u**·∇)**u** is the source of nonlinearity — the same convection–diffusion tension seen in Burgers, now vector-valued and coupled to a pressure that enforces incompressibility.

The pressure is not a dynamical variable with its own evolution equation. It is a **Lagrange multiplier** that instantaneously enforces ∇·**u** = 0. This is the "saddle-point" character of incompressible flow and the reason the discrete systems need care (see §6).

---

## 3. Geometry (Schäfer–Turek / DFG benchmark)

We adopt the canonical DFG 2D benchmark geometry, which has published reference values:

| Quantity | Symbol | Value |
|---|---|---|
| Channel length | L | 2.2 m |
| Channel height | H | 0.41 m |
| Cylinder centre | (x_c, y_c) | (0.2, 0.2) m |
| Cylinder diameter | D | 0.1 m (radius R = 0.05 m) |

The cylinder is placed **slightly below** the channel centreline (centre at y = 0.2 while the channel spans 0 to 0.41, so the midline is at 0.205). This small asymmetry is deliberate: it seeds the symmetry-breaking that, at higher Re, triggers alternating vortex shedding rather than a perfectly symmetric standing wake.

```
 y=H ┌─────────────────────────────────────────────┐  ← no-slip wall
     │                                             │
     │      ___                                    │
inlet│     /   \        (wake)                     │outlet
 →   │    |  ●  |  ~~~~~~~~~~~~~                    │ →  (do-nothing)
 →   │     \___/                                   │
     │       cylinder                              │
 y=0 └─────────────────────────────────────────────┘  ← no-slip wall
     x=0                                          x=L
```

---

## 4. Boundary conditions

The four boundary types close the system. Note the **complementarity rule** from Stage 1: velocity and pressure conditions must pair up correctly at inflow and outflow, or the problem is ill-posed.

- **Inlet (x = 0): Dirichlet velocity, parabolic profile.**
  $$
  u(0,y) = 4\,U_m\,\frac{y\,(H-y)}{H^2},\qquad v(0,y)=0
  $$
  This is Poiseuille inflow — the exact analytic parabola established as a sanity-check target in Stage 1. Its maximum is U_m at the centreline; its mean is $\bar U = \tfrac{2}{3}U_m$.

- **Channel walls (y = 0 and y = H): no-slip.** $\;\mathbf{u} = \mathbf{0}$.

- **Cylinder surface: no-slip.** $\;\mathbf{u} = \mathbf{0}$. This is the curved boundary that a Cartesian-input PINN must learn to respect and that the FEM ground truth resolves by trimming.

- **Outlet (x = L): "do-nothing" / natural / traction-free.**
  $$
  \big(\nu\,\nabla\mathbf{u} - \tfrac{p}{\rho}\mathbf{I}\big)\cdot \mathbf{n} = \mathbf{0}
  $$
  In the weak (FEM) formulation this term simply vanishes from the boundary integral, which is why it is called "do-nothing." Physically it lets flow leave without imposing an artificial velocity, and it **pins the pressure level** so p is uniquely defined (otherwise pressure is only determined up to an additive constant).

**Complementarity check:** velocity is prescribed at the inlet (Dirichlet-u) while pressure is left free there; at the outlet velocity is free and the traction (pressure-carrying) condition is imposed. Inlet and outlet are complementary — the common silent-error source flagged in Stage 1 is avoided.

---

## 5. Reynolds number and flow regime

The Reynolds number based on the cylinder **diameter** and the **mean** inflow speed is

$$
\mathrm{Re} = \frac{\bar U\,D}{\nu},\qquad \bar U = \tfrac{2}{3}U_m .
$$

Regime map for a cylinder in cross-flow:

| Re range | Behaviour |
|---|---|
| Re ≲ 5 | creeping flow, attached, fore–aft nearly symmetric |
| ~5 – 47 | **steady** laminar with a fixed pair of recirculation vortices behind the cylinder |
| ~47 – 180 | periodic laminar vortex shedding (von Kármán street) |
| ≳ 180 | 3D instabilities, eventually turbulence |

**Chosen starting case — DFG 2D-1 (steady):** $U_m = 0.3$, $\nu = 10^{-3}$, so $\bar U = 0.2$ and

$$
\mathrm{Re} = \frac{0.2 \times 0.1}{10^{-3}} = 20 .
$$

This sits firmly in the steady laminar band. There is **one stationary recirculation bubble**, no time dependence, and no shedding — the correct first rung, matching the Re ≈ 10–40 laminar regime confirmed in Stage 1. It rules out turbulence modelling and keeps the first PINN experiment a *steady* boundary-value problem rather than a time-marching one.

> **Later rungs.** Raising U_m toward ~1.5 (2D-2/2D-3 cases) pushes Re to ~100 and switches on vortex shedding, at which point the Strouhal number and time-periodic lift become the validation targets. That is a deliberate later step, not the starting point.

---

## 6. Sanity-check targets (before trusting any solver)

Consistent with the project's rule of establishing analytic benchmarks first:

1. **Inlet Poiseuille parabola** — the imposed profile is exact; any solver or PINN must reproduce it at x = 0 to machine/what training allows.
2. **Mass conservation** — total flux through any vertical line must equal the inlet flux $\int_0^H u\,dy = \tfrac{2}{3}U_m H$ (incompressibility as a global check).
3. **Front stagnation pressure** — a high-pressure stagnation point on the upstream face of the cylinder, low pressure at the shoulders (top/bottom), consistent with Bernoulli intuition.
4. **Symmetric standing wake at Re = 20** — two counter-rotating vortices of equal size immediately behind the cylinder; strong asymmetry at this Re would indicate a bug.
5. **Reference drag/lift** (optional, quantitative) — the DFG benchmark publishes $c_D \approx 5.58$ and $c_L \approx 0.0106$ for the steady 2D-1 case; matching these is the gold-standard validation.

---

## 7. Numerical ground truth (independent of the PINN)

The truth field is computed by a **completely independent method** — Finite Element Method with **Taylor–Hood P2/P1 elements** (piecewise-quadratic velocity, piecewise-linear pressure) via **nutils**. This is the cylinder-flow analogue of the FEM/spectral/FDM solvers used as ground truth in the Burgers and KdV projects.

Key numerical points (documented rather than silently assumed):

- **Weak form.** Multiply the momentum equation by a velocity test function **v**, the continuity equation by a pressure test function q, integrate by parts. The do-nothing outlet makes the boundary stress term vanish.
- **Nonlinearity.** The convective term makes the discrete system nonlinear; it is solved by Newton iteration.
- **Cut cells.** The cylinder is removed from a background Cartesian mesh by *trimming* against the level set $\sqrt{(x-x_c)^2+(y-y_c)^2} - R$. Trimming can leave a few "orphan" basis functions with almost no support; these are detected and constrained to zero.
- **Pressure gauge.** Because the outlet is the only place pressure is anchored, p is pinned to zero there to remove the constant nullspace.
- **inf–sup / LBB stabilization.** Cut cells can violate the discrete inf–sup condition, producing spurious checkerboard pressure modes. A small **Brezzi–Pitkäranta** term $-\tau h^2\,\nabla q\cdot\nabla p$ is added to the mass residual to regularize the pressure without meaningfully perturbing the solution.
- **Units.** All parameters are pinned in SI, as required for later OpenFOAM cross-checks.

The solver exports a dense point cloud `(x, y, u, v, p)` (with the cylinder interior already excluded) plus metadata `(Re, U_m, ν, geometry)`, which becomes the validation set for the PINN.

---

## 8. Classical PINN formulation

The PINN is a coordinate network $\mathcal{N}_\theta(x,y) \approx (u, v, p)$ trained on a **composite physics-informed loss** — the same scaffolding as the 2D Burgers PINN, adapted to steady Navier–Stokes with a pressure output and a curved interior boundary:

$$
\mathcal{L} = \lambda_{\text{pde}}\,\mathcal{L}_{\text{pde}}
            + \lambda_{\text{bc}}\,\mathcal{L}_{\text{bc}}
            + \lambda_{\text{data}}\,\mathcal{L}_{\text{data}}
$$

- **$\mathcal{L}_{\text{pde}}$** — mean-squared residual of the two momentum equations and continuity, evaluated at interior collocation points via automatic differentiation (first and second derivatives of the network outputs).
- **$\mathcal{L}_{\text{bc}}$** — enforces the parabolic inlet, no-slip walls, no-slip cylinder, and a pressure reference at the outlet.
- **$\mathcal{L}_{\text{data}}$** — optional supervision from the FEM ground truth at a sparse set of points, which stabilizes training for the coupled pressure–velocity system.

Design choices carried over from prior projects:

- **tanh activations** (smooth, infinitely differentiable — needed for clean second derivatives).
- **Xavier/Glorot initialization**, biases zero.
- **Adam warm-up → L-BFGS refinement** (the instrumented two-phase schedule from the 2D Burgers notebook).
- A **pressure gauge** term, because the incompressible pressure is otherwise defined only up to a constant and the network would drift.
- Collocation biased toward the **near-wake and cylinder boundary layer**, the analogue of the shock-biased collocation used in Burgers.
- **Ablation arms remain mandatory** for any later quantum-contribution claim: classical twin and frozen-circuit arms, per the honesty guardrails.

The step-by-step Classical PINN implementation is provided separately in `classical_pinn.py`.

---

## 9. Engineering choices that are underspecified in the source and made explicit here

- **Pipe/channel axis orientation:** the flow axis is **x** (streamwise), with y the cross-stream (wall-normal) direction. This resolves the z-vs-x axis ambiguity flagged as pending in Stage 1 — for this 2D channel benchmark, x is streamwise and there is no z.
- **Inlet-edge velocity discontinuity:** the parabolic inlet is zero at the walls (y = 0, H) and the walls are no-slip, so the two conditions are consistent at the corners — no discontinuity artifact here (unlike a plug inflow, which would need explicit corner handling).
- **Steady vs. transient:** we solve the steady BVP at Re = 20. The transient/shedding case is a deliberate later stage requiring time as a third network input and a different ground-truth (time-stepping) solver.
- **Pressure output vs. stream-function:** we output p directly (three network outputs u, v, p) rather than using a stream-function formulation, to keep the residual structure identical to the FEM weak form and to make pressure directly comparable against the ground truth.

---

## 10. Staging summary

| Sub-stage | Deliverable | Status |
|---|---|---|
| 2.0 Foundations | this document | ✅ |
| 2.1 Ground truth | `ground_truth.py` (nutils FEM, Taylor–Hood P2/P1) | ✅ (Re = 20 solved, validated) |
| 2.2 Classical PINN | `classical_pinn.py` (PyTorch) | ✅ implementation provided |
| 2.3 Quantum extension | HQPINN (sandwich: classical encoder → quantum circuit → decoder) | ⏭ next |
| 2.4 XAI | reuse `xai/` package via `QuantumProbe.from_qapinn()` | ⏭ |
| 2.5 Y-mixer geometry | subsequent geometry step | ⏭ |