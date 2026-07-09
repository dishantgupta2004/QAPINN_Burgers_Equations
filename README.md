# Burgers' Equation: FEniCSx Ground Truth + PyTorch PINN

A two-stage framework for the 1-D viscous Burgers equation:

$$\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}
   = \nu\,\frac{\partial^2 u}{\partial x^2},
   \qquad x\in[x_{\min}, x_{\max}],\; t\in(0, T].$$

1. **High-fidelity data generation** — a modular finite-element solver built on
   **FEniCSx (dolfinx)** produces the ground-truth solution (Backward-Euler in
   time, monolithic PETSc/SNES Newton for the nonlinearity) and exports it as a
   clean `[x, t, u]` dataset.
2. **Physics-Informed Neural Network** — a **PyTorch** PINN
   (`Burgers_PINN.ipynb`) learns the solution field, combining a data-fit term
   against the FEM ground truth with an autograd-based PDE-residual term.

```
  FEniCSx solver  ──►  output/numpy/burgers_dataset.npy  ──►  PyTorch PINN
  (src/, ground truth)     [x, t, u]  contract                (Burgers_PINN.ipynb)
```

The finite-element half is a **PDE-agnostic scaffold**: the same mesh / function
spaces / time loop / I/O / diagnostics can host the Heat equation, Navier–Stokes,
or serve as the data generator for other PINN studies. Swapping the physics is a
one-class change — see [`docs/extending.md`](docs/extending.md).

---

## Project layout

```
.
├── src/                     # the finite-element solver (a proper package)
│   ├── run_burger.py        #   entry point + experiment config
│   ├── fenics_backend.py    #   centralized dolfinx/PETSc/MPI import
│   ├── config/ physics/ solver/ diagnostics/ io/ viz/ utils/ models/
│   └── burgers_equation.py  #   backward-compat shim (old monolith name)
├── Burgers_PINN.ipynb       # PyTorch PINN training notebook
├── Burgers_GMRES_Reference.ipynb
├── docs/
│   ├── architecture.md      # module-by-module breakdown
│   └── extending.md         # how to add BCs / PDEs / solvers / outputs
├── output/                  # generated data + figures (created on first run)
├── requirements.txt
└── README.md
```

A full module-by-module reference is in
[`docs/architecture.md`](docs/architecture.md).

---

## Setup and installation

This project depends on a **heavy scientific stack** that is not fully
pip-installable, most notably the FEniCSx suite (`fenics-basix`,
`fenics-dolfinx`, `fenics-ffcx`, `fenics-ufl`) and PETSc/SLEPc bindings
(`petsc4py`, `slepc4py`). In `requirements.txt` these appear as local
`file:///...` installs — they are provided by a **FEniCSx container image**, not
downloaded from PyPI.

### Recommended: the provided dev container / Docker

The repository ships a `.devcontainer/` that builds on a FEniCSx image with
dolfinx, PETSc, MPI, PyTorch and CUDA already present. Open the folder in VS Code
→ *"Reopen in Container"*, or use the official image directly:

```bash
docker run -ti -v "$PWD":/workspaces/PINN -w /workspaces/PINN \
    dolfinx/dolfinx:stable
```

Verified working versions:

| component | version |
|-----------|---------|
| Python | 3.12.3 |
| dolfinx (FEniCSx) | 0.11 |
| PyTorch | 2.12.1 (`+cu130`) |
| PETSc / mpi4py | via container |
| CUDA toolkit | 13.x (optional — CPU works) |

### Python-only extras

Anything *not* provided by the FEniCSx image (matplotlib, h5py, torch, …) can be
installed with pip. **Do not** blindly `pip install -r requirements.txt` on a
bare machine — the `file:///` FEniCSx/PETSc lines will fail. Install the
FEniCSx stack via the container/conda, then add the rest:

```bash
pip install torch matplotlib h5py numpy scipy
```

### CUDA

The PINN runs on GPU automatically when available (`torch.cuda.is_available()`);
otherwise it falls back to CPU. The FEM solver is CPU/MPI-based and does not
require a GPU.

### Verify the environment

```bash
python -c "import dolfinx, torch; print('dolfinx', dolfinx.__version__, '| torch', torch.__version__)"
```

---

## Step 1 — Generate the ground-truth data

Run from the **repository root**:

```bash
python -m src.run_burger
```

> The `python -m` form is required so the `src` package and its absolute imports
> resolve. Running `python src/run_burger.py` directly will fail.

Edit the experiment in `src/run_burger.py` → `build_config()` (viscosity, mesh,
time step, initial/boundary conditions, which outputs to write). The default
(`nu=0.01`, `nx=300`, `dt=0.002`, `T=1.0`, `sin` IC, homogeneous Dirichlet BCs)
completes in ~15–20 s on CPU and produces, under `output/`:

| file | contents |
|------|----------|
| `numpy/burgers_dataset.npy` | **the PINN dataset** — `(N, 3)` table of `[x, t, u]` |
| `csv/burgers_dataset.csv` | same, as CSV |
| `numpy/<experiment>.npz` | grid arrays `x (nx,)`, `t (nt,)`, `u (nt, nx)` |
| `csv/<experiment>.csv` | wide CSV (time × space) |
| `xdmf/<experiment>.xdmf` (+`.h5`) | ParaView-viewable space-time field |
| `csv/<experiment>_diagnostics.csv` | per-step `umax, umin, L2, energy, mass` |
| `figures/*.png` | snapshots, space-time heatmap, final profile |
| `config.json` | the exact config used (read back by the PINN notebook) |

Running in parallel with MPI is supported for the solve:

```bash
mpirun -n 4 python -m src.run_burger
```

(Snapshot/file output uses serial-accurate rank-0 gather semantics.)

---

## Step 2 — Train the PINN

With the data generated, launch Jupyter **from the repository root** (so the
relative `output/...` paths resolve) and open the notebook:

```bash
jupyter lab           # or: jupyter notebook
# open Burgers_PINN.ipynb and run all cells
```

The notebook:

1. reads `nu`, the domain and `T` from `output/config.json` (guaranteeing the
   PINN's physics matches the data),
2. loads `output/numpy/burgers_dataset.npy` via the `[x, t, u]` contract,
3. trains an MLP `u_θ(x, t)` with a **four-part loss** — data fit + PDE residual
   ($u_t + u\,u_x - \nu\,u_{xx}$, computed with `torch.autograd.grad`) + initial
   condition + Dirichlet boundaries,
4. optimizes with Adam then an optional L-BFGS polish,
5. evaluates against the FEM grid (relative $L^2$ error), plots FEM-vs-PINN
   heatmaps and profiles, and saves the model to `checkpoints/burgers_pinn.pt`.

Hyperparameters are overridable via environment variables (defaults give a full
training run; lower them for a quick smoke test):

```bash
# examples
PINN_EPOCHS=10000 PINN_LBFGS=1500 jupyter lab      # full (default)
PINN_EPOCHS=500  PINN_LBFGS=0   jupyter lab        # fast sanity check
```

| variable | default | meaning |
|----------|---------|---------|
| `PINN_EPOCHS` | `10000` | Adam iterations |
| `PINN_LBFGS` | `1500` | L-BFGS refinement steps (`0` skips) |
| `PINN_NDATA` | `8000` | FEM samples used for the data term |
| `PINN_NCOLL` | `12000` | PDE collocation points |
| `PINN_LR` | `1e-3` | Adam learning rate |

To run headless (CI / batch), execute the notebook with `jupyter nbconvert
--to notebook --execute Burgers_PINN.ipynb` once `nbconvert` is installed.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — every module in `src/`, what
  it does, and how they compose.
- [`docs/extending.md`](docs/extending.md) — recipes for adding boundary
  conditions, changing PDE parameters, swapping the PDE, integrating different
  FEniCSx/PETSc solvers, adding output formats, and going to 2D/3D.

## Troubleshooting

- **`No module named src.run_burger`** — run with `python -m src.run_burger`
  from the repository root, not `python src/run_burger.py`.
- **`ImportError: dolfinx …`** — you are outside the FEniCSx environment; use
  the dev container / Docker image (see *Setup*). The pure-Python config layer
  (`from src import BurgersConfig`) imports fine without dolfinx; only the solver
  needs it.
- **PINN paths not found** — launch Jupyter from the repository root, and run
  Step 1 first so `output/` exists.
