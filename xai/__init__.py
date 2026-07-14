"""
xai
===

Explainable-AI & theoretical-expressivity analysis suite for the
**QA-PINN vs. Classical-PINN** comparison on the 1-D viscous Burgers equation.

Sub-modules
-----------
``common``          Paths, physics constants, feature-bundle loader, JSON utils.
``models``          Faithful QA-PINN / Classical-PINN reconstructions + QNodes.
``quantum_metrics`` Circuit depth, Meyer-Wallach & von-Neumann entanglement,
                    effective dimension / SVD information-bottleneck analysis.
``expressivity``    Fourier spectrum of the quantum encoding, gradient dynamics.
``landscapes``      Loss landscapes (Hessian spectrum, 1-D/2-D slices) and
                    barren-plateau (gradient-variance vs. qubits) diagnostics.
``visualizer``      Publication-quality Matplotlib plotting utilities.
``run_analysis``    End-to-end orchestrator over the saved checkpoints.
``physics_guided_burgers``
                    Physics-Guided Design Framework for the Burgers QA-PINN:
                    PDE descriptor, PDE->architecture mapping, physics-guided
                    regularisers, shock-aware sampling, unified loss & analyzer.

Run the full pipeline with::

    python xai/run_analysis.py
"""

from __future__ import annotations

from . import common, models  # noqa: F401

__all__ = [
    "common",
    "models",
    "quantum_metrics",
    "expressivity",
    "landscapes",
    "visualizer",
    "physics_guided_burgers",
]

__version__ = "1.0.0"
