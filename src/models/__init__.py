"""
src.models — PyTorch PINN networks (placeholder).
=================================================

This package is a deliberate, documented placeholder. The finite-element code in
this repository is the *high-fidelity data generator*: it solves the viscous
Burgers equation with FEniCSx and exports the canonical ``[x, t, u]`` table
(see :mod:`src.io.pinn_dataset`) that a Physics-Informed Neural Network consumes.

The PyTorch PINN / QTN-PINN / QAPINN models themselves currently live in the
top-level notebooks (``Burgers_PINN.ipynb``). When that work is productionized,
the ``nn.Module`` definitions and their training loop belong here (models) and in
a sibling ``src/training/`` package respectively — the dataset contract they
depend on is already stable.
"""

__all__: list[str] = []
