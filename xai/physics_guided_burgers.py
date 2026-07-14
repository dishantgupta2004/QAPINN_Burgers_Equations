r"""
xai.physics_guided_burgers
Scope
-----
**Burgers only.**  Every descriptor default, mapping threshold and regulariser
target below is specialised to the 1-D viscous Burgers equation

.. math::
    u_t + u\,u_x = \nu\,u_{xx}, \qquad
    x\in[0,1],\; t\in[0,1],\; \nu = 0.01,

with homogeneous Dirichlet BCs and a single-period sine initial condition.  The
module deliberately does **not** attempt to generalise to arbitrary PDEs (that is
future work; see the report's Stage VI).

Design principle -- *reuse, never duplicate*
--------------------------------------------
All established explainability metrics come from the existing suite:

* Fourier / spectral primitives   -> :func:`xai.expressivity._fft_power`,
  :func:`xai.expressivity._spectral_summary`
* Entanglement                    -> :func:`xai.quantum_metrics.meyer_wallach`,
  :func:`xai.quantum_metrics.mean_bipartite_entropy`
* Effective dimension / SVD       -> :func:`xai.quantum_metrics.effective_dimension`,
  :func:`xai.quantum_metrics.information_bottleneck`
* Circuit structure               -> :func:`xai.quantum_metrics.circuit_specs`
* Ansatz body (single source)     -> :func:`xai.models._apply_ansatz`
* Hessian / barren plateau        -> :mod:`xai.landscapes`

The framework maps
:math:`\text{PDE} \xrightarrow{\;\Phi\;} \mathcal{D}
       \xrightarrow{\;\Psi\;} \text{Architecture} + \text{Regularisation}`,
following Stages I--IV of the report.

Contents
--------
1.  :class:`BurgersDescriptor`               -- Stage I mathematical descriptor.
2.  Fourier / spectral analysis              -- :func:`power_spectrum`,
    :func:`spectral_complexity`, :func:`shock_frequency`.
3.  :class:`ArchitectureRecommendation`      -- Stage II PDE -> architecture map.
4.  :class:`SpectralRegularizer`             -- Stage III spectral regularisation.
5.  :class:`EntanglementRegularizer`         -- Stage III entanglement regularisation.
6.  :func:`recommend_reuploading_blocks`     -- Stage III encoding regularisation.
7.  :func:`circuit_complexity_metrics`       -- Stage III circuit-complexity regularisation.
8.  :class:`MeasurementRegularizer`          -- Stage III measurement regularisation.
9.  :func:`effective_dimension_report`       -- Stage III effective-dimension regularisation.
10. :func:`sample_collocation`               -- Stage III shock-aware adaptive sampling.
11. :class:`PhysicsGuidedLoss`               -- Stage IV unified loss (switchable terms).
12. :class:`BurgersPhysicsAnalyzer`          -- Stage V orchestrator over existing XAI metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from . import models, quantum_metrics, expressivity
from .common import PHYSICS, FeatureBundle

try:  
    import pennylane as qml

    _HAS_PENNYLANE = True
except Exception: 
    _HAS_PENNYLANE = False


__all__ = [
    "BurgersDescriptor",
    "burgers_descriptor",
    "power_spectrum",
    "spectral_complexity",
    "shock_frequency",
    "ArchitectureRecommendation",
    "recommend_architecture",
    "SpectralRegularizer",
    "EntanglementRegularizer",
    "recommend_reuploading_blocks",
    "circuit_complexity_metrics",
    "MeasurementRegularizer",
    "effective_dimension_report",
    "sample_collocation",
    "PhysicsGuidedLossConfig",
    "PhysicsGuidedLoss",
    "BurgersPhysicsAnalyzer",
]


# =========================================================================== #
# 1. Stage I -- Burgers mathematical descriptor
# =========================================================================== #
@dataclass
class BurgersDescriptor:
    r"""Mathematical descriptor vector :math:`\mathcal{D}` of the 1-D viscous
    Burgers equation (Stage I of the Physics-Guided Design Framework).

    Each field is one coordinate of the descriptor that the Stage II mapping
    :math:`\Psi:\mathcal{D}\to\text{architecture}` consumes.  Scalar descriptors
    that depend only on the *symbolic* form of the PDE (linearity, differential
    order, coupling, boundary, dimensionality) carry Burgers-specific defaults;
    the data-driven descriptors (:attr:`spectral_complexity_value`,
    :attr:`shock_indicator`, :attr:`shock_frequency_value`) are estimated from a
    reference solution field via :func:`burgers_descriptor`.

    Attributes:
        nu: Kinematic viscosity :math:`\nu` (controls shock width :math:`\sim\nu`).
        input_dimensionality: Number of independent variables :math:`d=|\{x,t\}|=2`.
        differential_order: Highest derivative order :math:`p=2` (viscous
            :math:`u_{xx}` term) -- deeper circuits are needed for higher ``p``.
        linearity_index: Nonlinearity strength :math:`\mathcal{L}\in[0,1]`.
            Burgers advection :math:`u\,u_x` is strongly nonlinear
            (:math:`\mathcal{L}\approx0.7`), well above Heat/Poisson
            (:math:`\approx0`) and below Navier-Stokes.
        variable_coupling: Coupling index :math:`\mathcal{C}\in[0,1]`.  Burgers
            couples :math:`u` to its own spatial derivative (self-advection),
            giving moderate coupling that maps to a *ring* entanglement topology.
        boundary_complexity: :math:`\mathcal{B}\in[0,1]`.  Homogeneous Dirichlet
            walls are the lowest-complexity case (:math:`\approx0.1`).
        second_order_operator: Whether a second-order spatial operator is present
            (``True`` for the viscous Laplacian).
        non_separable: Whether the solution is non-separable in :math:`(x,t)`
            (``True``: the nonlinear advection couples space and time).
        spectral_complexity_value: :math:`C_s\in[0,1]`, the normalised spectral
            centroid of the solution's spatial power spectrum (see
            :func:`spectral_complexity`).  Small -> smooth; large -> shocks.
        shock_indicator: :math:`\mathcal{S}\ge 0`, normalised maximum spatial
            steepness :math:`\max|u_x|` -- a data-driven shock strength proxy.
        shock_frequency_value: Characteristic shock wavenumber (cycles per unit
            :math:`x`) that the encoding must be able to represent.

    Complexity:
        Construction is :math:`O(1)` from precomputed scalars; the data-driven
        fields are :math:`O(n_t n_x \log n_x)` (one FFT per time slice).
    """

    nu: float = PHYSICS.nu
    input_dimensionality: int = 2
    differential_order: int = 2
    linearity_index: float = 0.7
    variable_coupling: float = 0.5
    boundary_complexity: float = 0.1
    second_order_operator: bool = True
    non_separable: bool = True
    spectral_complexity_value: float = 0.25
    shock_indicator: float = 1.0
    shock_frequency_value: float = 8.0

    def as_vector(self) -> Dict[str, float]:
        """Return the descriptor as a flat, JSON-serialisable mapping.

        Returns:
            Dict mapping descriptor name -> scalar value (bools cast to ``int``).
        """
        out: Dict[str, float] = {}
        for k, v in asdict(self).items():
            out[k] = int(v) if isinstance(v, bool) else v
        return out

    def summary(self) -> str:
        """Human-readable one-paragraph descriptor summary (for notebooks/logs)."""
        return (
            f"Burgers descriptor (nu={self.nu:g}): "
            f"d={self.input_dimensionality}, order={self.differential_order}, "
            f"linearity L={self.linearity_index:.2f}, coupling C={self.variable_coupling:.2f}, "
            f"boundary B={self.boundary_complexity:.2f}, "
            f"spectral C_s={self.spectral_complexity_value:.3f}, "
            f"shock S={self.shock_indicator:.2f}, "
            f"shock-freq k_s={self.shock_frequency_value:.2f} cyc/x"
        )


def burgers_descriptor(
    field: Optional[np.ndarray] = None,
    x_grid: Optional[np.ndarray] = None,
    nu: float = PHYSICS.nu,
) -> BurgersDescriptor:
    r"""Build the Burgers Stage-I descriptor, estimating data-driven fields.

    The symbolic descriptors (linearity, differential order, coupling, boundary,
    dimensionality) are fixed by the PDE.  When a reference solution ``field`` is
    provided, the spectral complexity :math:`C_s`, the shock indicator
    :math:`\mathcal{S}=\max|u_x|` and the characteristic shock frequency
    :math:`k_s` are estimated from it; otherwise Burgers defaults are used.

    Args:
        field: Reference solution :math:`u(x,t)` of shape ``(n_t, n_x)`` (e.g.
            ``FeatureBundle.U_grid``).  If ``None`` the descriptor keeps its
            defaults.
        x_grid: Spatial grid of length ``n_x`` used to scale the gradient /
            frequency estimates.  Defaults to a uniform :math:`[0,1]` grid.
        nu: Kinematic viscosity.

    Returns:
        A populated :class:`BurgersDescriptor`.

    Complexity:
        :math:`O(n_t\,n_x\log n_x)` (one FFT + one finite difference per row).
    """
    desc = BurgersDescriptor(nu=nu)
    if field is None:
        return desc

    U = np.asarray(field, dtype=np.float64)
    nx = U.shape[-1]
    if x_grid is None:
        x_grid = np.linspace(PHYSICS.xmin, PHYSICS.xmax, nx)
    dx = float(np.mean(np.diff(x_grid)))

    # --- spectral complexity + shock frequency (time-averaged spatial spectrum)
    freqs, P = power_spectrum(U, dx=dx, axis="x")
    desc.spectral_complexity_value = float(spectral_complexity(P, freqs))
    desc.shock_frequency_value = float(shock_frequency(P, freqs))

    # --- shock indicator: max spatial steepness normalised by amplitude/length
    ux = np.gradient(U, x_grid, axis=-1)
    amp = float(np.abs(U).max()) + 1e-12
    length = float(x_grid[-1] - x_grid[0]) + 1e-12
    desc.shock_indicator = float(np.abs(ux).max() * length / amp)
    return desc


# =========================================================================== #
# 2. Stage I -- Fourier / spectral analysis
# =========================================================================== #
def power_spectrum(
    field: np.ndarray,
    dx: float,
    axis: str = "x",
) -> Tuple[np.ndarray, np.ndarray]:
    r"""One-sided spatial power spectrum of a Burgers field, time-averaged.

    Thin, Burgers-specific wrapper around :func:`xai.expressivity._fft_power`
    (the single FFT primitive of the suite) that averages the per-time-slice
    power spectra so a single representative spectrum :math:`P(k)` is returned.

    .. math::
        P(k) = \frac{1}{n_t}\sum_{j} \frac{|\hat u_j(k)|^2}{\sum_k |\hat u_j(k)|^2},

    where :math:`\hat u_j` is the FFT of the :math:`j`-th time slice
    (mean-removed).  ``P`` is normalised to unit total energy.

    Args:
        field: Solution array of shape ``(n_t, n_x)`` (or ``(n_x,)``).
        dx: Sample spacing along the transformed axis.
        axis: ``"x"`` transforms rows (spatial); ``"t"`` transforms columns.

    Returns:
        Tuple ``(freqs, power)`` of matching 1-D arrays; ``freqs`` in cycles per
        unit input, ``power`` normalised to unit sum.

    Complexity:
        :math:`O(n_t n_x\log n_x)`.
    """
    U = np.atleast_2d(np.asarray(field, dtype=np.float64))
    if axis == "t":
        U = U.T  # transform along columns by transposing to rows
    freqs = None
    acc = None
    for row in U:
        f, p = expressivity._fft_power(row, dx)  # reuse suite FFT primitive
        freqs = f
        acc = p if acc is None else acc + p
    power = acc / U.shape[0]
    power = power / (power.sum() + 1e-30)
    return freqs, power


def spectral_complexity(
    power: np.ndarray,
    freqs: np.ndarray,
    domain_length: float = PHYSICS.xmax - PHYSICS.xmin,
) -> float:
    r"""Spectral-complexity descriptor :math:`C_s` (Stage I, report Sec. 3.3).

    Defined as the fraction of spectral energy that lies **above the smooth
    fundamental band** -- i.e. the share of the spatial spectrum carried by
    harmonics beyond the base mode:

    .. math::
        C_s = \frac{\sum_{k:\,f_k > f_0} P(k)}{\sum_k P(k)},
        \qquad f_0 = \frac{3}{2}\,\frac{1}{L},

    with :math:`L` the domain length (so :math:`1/L` is the domain fundamental).
    Unlike a Nyquist-normalised centroid this is **resolution independent**: a
    smooth single-mode sine gives :math:`C_s\!\to\!0`, whereas a viscous shock --
    whose steep front spreads energy into a decaying high-frequency tail -- gives
    a moderate-to-large :math:`C_s` (for the Burgers reference field
    :math:`C_s\approx0.23`, correctly placing it in the *angle + re-uploading*
    regime of Stage II).

    Interpretation (report): small :math:`C_s` -> smooth; large :math:`C_s` ->
    shocks, oscillations, turbulence, multi-scale behaviour.  :math:`C_s` is the
    primary driver of the encoding / re-uploading and decoder recommendations.

    Args:
        power: Normalised power spectrum :math:`P(k)` (from :func:`power_spectrum`).
        freqs: Matching frequency axis (cycles per unit input).
        domain_length: Physical length :math:`L` used to place the fundamental band.

    Returns:
        The scalar spectral complexity :math:`C_s\in[0,1]`.

    Complexity:
        :math:`O(n_x)`.
    """
    P = np.asarray(power, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    total = P.sum() + 1e-30
    f0 = 1.5 / (float(domain_length) + 1e-30)  # 1.5 x domain fundamental
    high = float(P[f > f0].sum())
    return float(np.clip(high / total, 0.0, 1.0))


def shock_frequency(
    power: np.ndarray,
    freqs: np.ndarray,
    energy_thresh: float = 0.95,
) -> float:
    r"""Estimate the characteristic shock wavenumber :math:`k_s`.

    The viscous Burgers shock has width :math:`\sim\nu`, so its energy occupies a
    band up to a finite cutoff.  We define :math:`k_s` as the smallest frequency
    capturing ``energy_thresh`` of the cumulative spectral energy -- the "knee"
    beyond which the physically meaningful shock content ends and (for a faithful
    solution) the spectrum decays.  The encoding must reach at least :math:`k_s`.

    This mirrors the ``max_frequency`` logic of
    :func:`xai.expressivity._spectral_summary` but exposes the physical
    shock-band interpretation used by the Stage II / Stage III recommendations.

    Args:
        power: Normalised power spectrum :math:`P(k)`.
        freqs: Matching frequency axis.
        energy_thresh: Cumulative-energy fraction defining the shock band cutoff.

    Returns:
        The shock frequency :math:`k_s` in cycles per unit input.

    Complexity:
        :math:`O(n_x)`.
    """
    P = np.asarray(power, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    cum = np.cumsum(P) / (P.sum() + 1e-30)
    k = int(np.searchsorted(cum, energy_thresh))
    k = min(k, len(f) - 1)
    return float(f[k])


# =========================================================================== #
# 3. Stage II -- PDE -> quantum architecture mapping
# =========================================================================== #
@dataclass
class ArchitectureRecommendation:
    r"""Recommended QA-PINN architecture derived from a Burgers descriptor.

    This is the output of the Stage II deterministic map :math:`\Psi`.  It
    encodes the report's central hypothesis -- *the PDE determines the quantum
    architecture* -- and, crucially, the report's top recommendations for
    Burgers: enrich the encoding via **data re-uploading**, use a **ring**
    entanglement topology on **every** layer (not even-layers-only), and size the
    decoder to the spectral richness.

    Attributes:
        encoding: One of ``"angle"``, ``"angle+reuploading"``, ``"iqp"``,
            ``"amplitude"`` (report Sec. 4 decision rules).
        reuploading_blocks: Number of encode-re-uploading repetitions :math:`L`.
            By the Schuld-Sweke-Meyer theorem the accessible Fourier degree grows
            with :math:`L`, so this is the single highest-leverage knob.
        n_qubits: Recommended qubit count (Burgers -> 4, report Sec. 5).
        circuit_depth: Number of variational layers (report Sec. 6).
        entanglement_topology: ``"ring"`` for Burgers (report Sec. 7).
        entangle_every_layer: If ``True``, apply entanglers on every layer rather
            than even-layers-only (the report's stronger-entanglement fix).
        decoder_hidden: Recommended decoder hidden width (report Sec. 8).
        rationale: Per-field human-readable justification.
    """

    encoding: str
    reuploading_blocks: int
    n_qubits: int
    circuit_depth: int
    entanglement_topology: str
    entangle_every_layer: bool
    decoder_hidden: int
    rationale: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Compact multi-line rendering of the recommendation."""
        return (
            "Physics-guided architecture recommendation\n"
            f"  encoding            : {self.encoding} (L={self.reuploading_blocks} re-uploading blocks)\n"
            f"  qubits              : {self.n_qubits}\n"
            f"  circuit depth       : {self.circuit_depth} variational layers\n"
            f"  entanglement        : {self.entanglement_topology} "
            f"({'every layer' if self.entangle_every_layer else 'even layers only'})\n"
            f"  decoder hidden width: {self.decoder_hidden}"
        )


def recommend_architecture(desc: BurgersDescriptor) -> ArchitectureRecommendation:
    r"""Map a Burgers descriptor to a QA-PINN architecture (Stage II).

    Deterministic mapping :math:`\Psi:\mathcal{D}\to\text{architecture}` using the
    report's decision rules, specialised to Burgers:

    * **Encoding** (Sec. 4): driven by :math:`C_s`.  Burgers' moderate-to-high
      spectral complexity selects *angle + data re-uploading* -- accessible
      frequencies come from the encoding, not the qubit count.
    * **Re-uploading depth** :math:`L` (Sec. 11): from :math:`C_s` and the shock
      frequency, see :func:`recommend_reuploading_blocks`.
    * **Qubits** (Sec. 5): from input dimensionality + coupling; Burgers -> 4.
    * **Depth** (Sec. 6): from differential order + nonlinearity; strong
      nonlinear PDE -> 6-8 layers.
    * **Entanglement** (Sec. 7): from :math:`\mathcal{C}`; Burgers -> ring, on
      every layer (the report's stronger-entanglement fix).
    * **Decoder** (Sec. 8): width scales with spectral richness.

    Args:
        desc: The Stage-I :class:`BurgersDescriptor`.

    Returns:
        An :class:`ArchitectureRecommendation`.

    Complexity:
        :math:`O(1)`.
    """
    Cs = desc.spectral_complexity_value
    rationale: Dict[str, str] = {}

    # --- encoding (Sec. 4) --------------------------------------------------
    if Cs < 0.1:
        encoding = "angle"
        rationale["encoding"] = "low C_s: single-shot angle encoding suffices."
    elif Cs < 0.5:
        encoding = "angle+reuploading"
        rationale["encoding"] = (
            "moderate C_s: angle + data re-uploading enlarges the accessible "
            "Fourier set (Perez-Salinas 2020); qubit count cannot (Schuld-Sweke-Meyer)."
        )
    else:
        encoding = "iqp"
        rationale["encoding"] = "high C_s: IQP-style encoding for rich high-frequency content."

    # --- re-uploading depth (Sec. 11) --------------------------------------
    L = recommend_reuploading_blocks(Cs, desc.shock_indicator, desc.shock_frequency_value)
    rationale["reuploading"] = (
        f"L={L}: degree of the accessible Fourier series must cover the shock "
        f"band k_s={desc.shock_frequency_value:.1f} cyc/x."
    )

    # --- qubits (Sec. 5) ----------------------------------------------------
    # base = input dimensionality; +coupling bonus; Burgers lands on 4.
    n_qubits = int(desc.input_dimensionality + round(2 * desc.variable_coupling) + 1)
    n_qubits = int(np.clip(n_qubits, 2, 8))
    rationale["qubits"] = (
        f"n={n_qubits} from input dim {desc.input_dimensionality} + coupling; "
        "adding qubits without spectral richness only inflates the 2^n space."
    )

    # --- circuit depth (Sec. 6) --------------------------------------------
    depth = int(round(2 * desc.differential_order + 3 * desc.linearity_index))
    depth = int(np.clip(depth, 2, 8))
    rationale["depth"] = (
        f"depth={depth} from order {desc.differential_order} + nonlinearity "
        f"{desc.linearity_index:.2f}; capped at 8 to avoid barren plateaus."
    )

    # --- entanglement topology (Sec. 7) ------------------------------------
    if desc.variable_coupling < 0.05:
        topology, every = "none", False
    elif desc.variable_coupling < 0.3:
        topology, every = "chain", True
    elif desc.variable_coupling < 0.7:
        topology, every = "ring", True
    else:
        topology, every = "all_to_all", True
    rationale["entanglement"] = (
        f"coupling C={desc.variable_coupling:.2f} -> {topology} topology on every "
        "layer (the report's fix for even-layers-only entanglement suppression)."
    )

    # --- decoder width (Sec. 8) --------------------------------------------
    # Widen beyond the current 2^(n-1) pinch point in proportion to spectral
    # richness so genuine high-frequency shock content reaches the output.
    base_hidden = 2 ** (n_qubits - 1)
    decoder_hidden = int(base_hidden * (1 + min(math.ceil(4 * Cs), 4)))
    rationale["decoder"] = (
        f"hidden={decoder_hidden}: widened from the 2^(n-1)={base_hidden} pinch "
        f"point by spectral richness C_s={Cs:.2f} so genuine high frequencies "
        "reach the output instead of collapsing."
    )

    return ArchitectureRecommendation(
        encoding=encoding,
        reuploading_blocks=L,
        n_qubits=n_qubits,
        circuit_depth=depth,
        entanglement_topology=topology,
        entangle_every_layer=every,
        decoder_hidden=decoder_hidden,
        rationale=rationale,
    )


# =========================================================================== #
# 6. Stage III -- Encoding regularisation (re-uploading recommendation)
# =========================================================================== #
def recommend_reuploading_blocks(
    spectral_complexity_value: float,
    shock_indicator: float,
    shock_frequency_value: float = 8.0,
    max_blocks: int = 8,
) -> int:
    r"""Recommend the number of data-re-uploading blocks :math:`L` (Sec. 11).

    By the Fourier-expressivity theorem (Schuld-Sweke-Meyer), :math:`L` repeated
    encodings of a datum give accessible integer frequencies
    :math:`\Omega=\{-L,\dots,L\}`.  To represent the Burgers shock band the model
    must reach :math:`k_s`, so the recommendation grows with both the shock
    frequency and the (spectral-complexity, shock-strength) pair:

    .. math::
        L = \operatorname{clip}\Big(
            \big\lceil k_s\big\rceil
            + \operatorname{round}\big(2\,C_s + \tanh(\mathcal{S}/10)\big),
            \; 1,\; L_{\max}\Big).

    The dominant term is :math:`\lceil k_s\rceil` (the degree needed to reach the
    shock band); the shock strength enters through a **saturating** :math:`\tanh`
    so that even very steep fronts add only a bounded increment rather than
    blowing :math:`L` up to the cap.  For the Burgers reference
    (:math:`k_s\approx3,\,C_s\approx0.23,\,\mathcal{S}\approx39`) this yields
    :math:`L\approx4` -- moderate re-uploading, as the report prescribes.

    Args:
        spectral_complexity_value: :math:`C_s\in[0,1]`.
        shock_indicator: :math:`\mathcal{S}\ge0` (normalised max steepness).
        shock_frequency_value: :math:`k_s`, characteristic shock wavenumber.
        max_blocks: Upper bound on :math:`L` (barren-plateau / cost guard).

    Returns:
        The recommended integer re-uploading depth :math:`L\ge1`.

    Complexity:
        :math:`O(1)`.
    """
    base = math.ceil(max(shock_frequency_value, 1.0))
    bonus = int(round(2.0 * spectral_complexity_value + math.tanh(max(shock_indicator, 0.0) / 10.0)))
    return int(np.clip(base + bonus, 1, max_blocks))


# =========================================================================== #
# 4. Stage III -- Spectral regularisation
# =========================================================================== #
class SpectralRegularizer(nn.Module):
    r"""PDE-aware spectral penalty on the spatial output (Sec. 9).

    Every PDE has an intrinsic spatial spectrum; the learned Burgers solution
    should be low-frequency-dominated with a sharp cutoff and **no** spurious
    high-frequency ringing (the report's key QA-PINN failure mode).  Given the
    network output :math:`u(\cdot, t)` sampled on a uniform spatial line, we form
    the one-sided power spectrum :math:`P(k)=|\hat u(k)|^2` via a differentiable
    ``torch.fft.rfft`` and penalise

    .. math::
        \mathcal{L}_{\mathrm{spec}} =
            \frac{\sum_k w(k)\,P(k)}{\sum_k P(k) + \varepsilon},

    where the weighting :math:`w(k)` depends on ``mode``:

    * ``"highpass"``   -- :math:`w(k)=(k/k_{\mathrm{Nyq}})^p`; suppress all high
      frequencies (Heat/Poisson-like strong suppression).
    * ``"bandpass"``   -- penalise energy **outside** :math:`[k_{lo}, k_{hi}]`.
    * ``"shock_preserve"`` -- **Burgers default**: zero penalty inside the shock
      band :math:`[0, k_s(1+\text{margin})]`, quadratic penalty above it, so the
      physically meaningful shock frequencies survive while decoder ringing is
      suppressed.
    * ``"adaptive"``   -- like ``shock_preserve`` but with the penalty strength
      scaled by an externally supplied factor (see :meth:`set_adaptive_scale`),
      enabling a spectral-penalty schedule during training.

    Args:
        mode: One of the four modes above.
        shock_frequency: :math:`k_s`, the shock-band cutoff (cycles per unit x).
        margin: Fractional widening of the preserved shock band.
        power: Exponent :math:`p` for the ``highpass`` ramp.
        k_lo, k_hi: Band edges for ``bandpass`` mode.
        domain_length: Physical spatial length used to convert samples to
            cycles-per-unit-x (defaults to the Burgers domain length).

    Complexity:
        :math:`O(B\,n\log n)` for a batch of :math:`B` lines of :math:`n` samples.
    """

    def __init__(
        self,
        mode: str = "shock_preserve",
        shock_frequency: float = 8.0,
        margin: float = 0.25,
        power: float = 2.0,
        k_lo: float = 0.0,
        k_hi: float = 8.0,
        domain_length: float = PHYSICS.xmax - PHYSICS.xmin,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.shock_frequency = float(shock_frequency)
        self.margin = float(margin)
        self.power = float(power)
        self.k_lo = float(k_lo)
        self.k_hi = float(k_hi)
        self.domain_length = float(domain_length)
        self._adaptive_scale = 1.0

    def set_adaptive_scale(self, scale: float) -> None:
        """Set the multiplicative penalty scale used in ``adaptive`` mode."""
        self._adaptive_scale = float(scale)

    def _weights(self, freqs: torch.Tensor) -> torch.Tensor:
        f_nyq = float(freqs.max().item()) + 1e-30
        if self.mode == "highpass":
            return (freqs / f_nyq) ** self.power
        if self.mode == "bandpass":
            out = torch.zeros_like(freqs)
            out[(freqs < self.k_lo) | (freqs > self.k_hi)] = 1.0
            return out
        # shock_preserve / adaptive: keep the shock band, penalise above it
        cutoff = self.shock_frequency * (1.0 + self.margin)
        excess = torch.clamp(freqs - cutoff, min=0.0)
        w = (excess / f_nyq) ** self.power
        if self.mode == "adaptive":
            w = w * self._adaptive_scale
        return w

    def forward(self, u_lines: torch.Tensor) -> torch.Tensor:
        r"""Compute the spectral penalty for a batch of spatial output lines.

        Args:
            u_lines: Tensor of shape ``(B, n)`` -- the network output evaluated at
                ``n`` uniformly spaced spatial points, for ``B`` time slices (or
                ``(n,)`` for a single line).

        Returns:
            Scalar penalty :math:`\mathcal{L}_{\mathrm{spec}}\ge 0`.
        """
        if u_lines.dim() == 1:
            u_lines = u_lines.unsqueeze(0)
        n = u_lines.shape[1]
        u = u_lines - u_lines.mean(dim=1, keepdim=True)
        fft = torch.fft.rfft(u, dim=1)
        power = (fft.real ** 2 + fft.imag ** 2)  # (B, n//2+1)
        dx = self.domain_length / (n - 1)
        freqs = torch.fft.rfftfreq(n, d=dx).to(u.device)
        w = self._weights(freqs).unsqueeze(0)  # (1, F)
        num = (w * power).sum(dim=1)
        den = power.sum(dim=1) + 1e-30
        return (num / den).mean()


# =========================================================================== #
# 5. Stage III -- Entanglement regularisation
# =========================================================================== #
def _make_state_qnode_torch(n_qubits: int, n_layers: int):
    """Torch-interface state QNode reusing the suite's single ansatz body.

    Returns a differentiable QNode ``(angles, weights) -> statevector`` built from
    :func:`xai.models._apply_ansatz`, so entanglement can enter the loss.
    """
    if not _HAS_PENNYLANE:  # pragma: no cover
        raise RuntimeError("pennylane is required for the entanglement regulariser")
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles, weights):
        models._apply_ansatz(angles, weights, n_qubits, n_layers)
        return qml.state()

    return circuit


def meyer_wallach_torch(states: torch.Tensor, n_qubits: int) -> torch.Tensor:
    r"""Differentiable batched Meyer-Wallach entanglement :math:`Q`.

    Torch counterpart of :func:`xai.quantum_metrics.meyer_wallach` (identical
    definition), vectorised over a batch of pure states so it can be used inside
    an autograd loss:

    .. math::
        Q(|\psi\rangle) = 2\Big(1 - \tfrac1n\sum_{k=1}^{n}\operatorname{Tr}\rho_k^2\Big),

    with :math:`\operatorname{Tr}\rho_k^2 = \sum_{a,a'}|(\rho_k)_{aa'}|^2` for the
    single-qubit reduced density matrix :math:`\rho_k`.

    Args:
        states: Complex tensor of shape ``(B, 2**n)`` -- batched statevectors.
        n_qubits: Number of qubits :math:`n`.

    Returns:
        Real tensor of shape ``(B,)`` with :math:`Q\in[0,1]` per state.

    Complexity:
        :math:`O(B\,n\,2^{n})`.
    """
    B = states.shape[0]
    psi = states.reshape((B,) + (2,) * n_qubits)
    purities = []
    for k in range(n_qubits):
        # bring qubit k to axis 1, flatten the rest
        perm = [0, k + 1] + [ax + 1 for ax in range(n_qubits) if ax != k]
        pk = psi.permute(*perm).reshape(B, 2, -1)  # (B, 2, 2**(n-1))
        rho = torch.einsum("bam,bcm->bac", pk, pk.conj())  # (B, 2, 2)
        purities.append((rho.real ** 2 + rho.imag ** 2).sum(dim=(1, 2)))
    mean_purity = torch.stack(purities, dim=1).mean(dim=1)
    return 2.0 * (1.0 - mean_purity)


class EntanglementRegularizer(nn.Module):
    r"""Meyer-Wallach entanglement penalty with a target + schedule (Sec. 10).

    The report's "smoking gun": training drives entanglement *down*, collapsing
    the circuit toward classicality.  This regulariser breaks that incentive by
    penalising the shortfall of the mean Meyer-Wallach :math:`\bar Q` below a
    physically motivated target :math:`Q^\star` (Burgers -> moderate):

    .. math::
        \mathcal{L}_{\mathrm{ent}} =
            \big(\max(0,\; Q^\star - \bar Q)\big)^2 ,
        \qquad \bar Q = \tfrac1B\sum_b Q(|\psi_b\rangle).

    Only the *deficit* is penalised, so the optimiser is rewarded for *using*
    entanglement but not forced to over-entangle.  An optional annealing schedule
    ramps :math:`Q^\star` from ``target_start`` to ``target`` over training
    (:meth:`set_progress`), matching the report's "adaptive entanglement schedule".

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers (defines the state QNode).
        target: Final target entanglement :math:`Q^\star` (Burgers default 0.5).
        target_start: Initial target for the schedule (defaults to ``target``).
        n_samples: Max number of domain points evaluated per loss call.

    Complexity:
        :math:`O(n_{\mathrm{samples}}\, n\,2^{n})` per call (state simulation).
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = models.LAYERS_PAPER,
        target: float = 0.5,
        target_start: Optional[float] = None,
        n_samples: int = 64,
    ) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.target = float(target)
        self.target_start = float(target if target_start is None else target_start)
        self.n_samples = int(n_samples)
        self._progress = 1.0
        self._qnode = _make_state_qnode_torch(n_qubits, n_layers)

    def set_progress(self, progress: float) -> None:
        """Set schedule progress :math:`\\in[0,1]` (0=start target, 1=final)."""
        self._progress = float(np.clip(progress, 0.0, 1.0))

    @property
    def current_target(self) -> float:
        """The scheduled target :math:`Q^\\star` at the current progress."""
        return self.target_start + (self.target - self.target_start) * self._progress

    def forward(self, angles: torch.Tensor, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Entanglement penalty over a batch of encoded states.

        Args:
            angles: Encoding angles of shape ``(B, n_qubits)`` (e.g. from
                ``QAPINN.angles(x, t)``).
            weights: Variational weights of shape ``(n_layers, n_qubits)``.

        Returns:
            Dict with ``"loss"`` (scalar penalty), ``"mean_Q"`` (batch-mean
            Meyer-Wallach) and ``"target"`` (current :math:`Q^\star`).
        """
        if angles.shape[0] > self.n_samples:
            idx = torch.randperm(angles.shape[0], device=angles.device)[: self.n_samples]
            angles = angles[idx]
        states = self._qnode(angles, weights)
        states = torch.as_tensor(states)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        Q = meyer_wallach_torch(states, self.n_qubits)
        mean_Q = Q.mean()
        deficit = torch.clamp(self.current_target - mean_Q, min=0.0)
        return {"loss": deficit ** 2, "mean_Q": mean_Q, "target": torch.tensor(self.current_target)}


# =========================================================================== #
# 7. Stage III -- Circuit-complexity regularisation
# =========================================================================== #
def circuit_complexity_metrics(
    n_qubits: int,
    n_layers: int,
    reuploading_blocks: int = 1,
    effective_dim: Optional[float] = None,
    target_depth: Optional[int] = None,
) -> Dict[str, float]:
    r"""Circuit-complexity descriptors + depth penalty (Sec. 12).

    Combines the structural facts from
    :func:`xai.quantum_metrics.circuit_specs` (reused, not re-derived) with the
    Physics-Guided efficiency ratios that the framework introduces:

    * **Depth penalty** :math:`\mathcal{L}_{\mathrm{depth}} =
      \big(\max(0, D - D^\star)\big)^2` -- discourage circuits deeper than the
      Stage II recommendation :math:`D^\star`.
    * **Parameter efficiency** :math:`d_{\mathrm{eff}} / N_{\mathrm{params}}` --
      representational return per trainable parameter (needs ``effective_dim``).
    * **Hilbert-space utilisation** :math:`d_{\mathrm{eff}} / 2^{n}` -- fraction
      of the exponential space actually recruited (the report's core deficiency).
    * **Qubit efficiency** :math:`d_{\mathrm{eff}} / n` -- effective dimensions per
      qubit; flat/decreasing values flag wasted qubits.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers :math:`D`.
        reuploading_blocks: Encoding repetitions :math:`L` (added to the RX count).
        effective_dim: Measured :math:`d_{\mathrm{eff}}` of ``quantum_probs``
            (e.g. from :func:`effective_dimension_report`); enables the efficiency
            ratios when supplied.
        target_depth: Recommended depth :math:`D^\star`; defaults to ``n_layers``.

    Returns:
        Dict of structural facts, efficiency ratios and ``depth_penalty``.

    Complexity:
        Dominated by :func:`circuit_specs` (:math:`O(\text{gates})`).
    """
    specs = quantum_metrics.circuit_specs(n_qubits, n_layers)  # reuse
    hilbert = float(2 ** n_qubits)
    n_params = float(specs["trainable_rx_params"] + reuploading_blocks * n_qubits)
    D_star = int(n_layers if target_depth is None else target_depth)
    depth_penalty = float(max(0, n_layers - D_star) ** 2)

    out: Dict[str, float] = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "reuploading_blocks": reuploading_blocks,
        "hilbert_dim": hilbert,
        "circuit_depth": specs.get("circuit_depth"),
        "n_trainable_params": n_params,
        "target_depth": D_star,
        "depth_penalty": depth_penalty,
    }
    if effective_dim is not None:
        out["effective_dim"] = float(effective_dim)
        out["parameter_efficiency"] = float(effective_dim / (n_params + 1e-30))
        out["hilbert_utilisation"] = float(effective_dim / hilbert)
        out["qubit_efficiency"] = float(effective_dim / n_qubits)
    return out


# =========================================================================== #
# 8. Stage III -- Measurement regularisation
# =========================================================================== #
class MeasurementRegularizer(nn.Module):
    r"""Measurement-entropy penalty against probability collapse (Sec. 13).

    The :math:`2^n` basis probabilities :math:`p_i(x,t)` should not collapse onto
    a few basis states (which would waste the Hilbert space -- the report's
    low-effective-dimension pathology).  We use the Shannon entropy of each
    measurement distribution,

    .. math::
        H(p) = -\sum_{i=1}^{2^n} p_i \log p_i,\qquad
        \mathcal{L}_{\mathrm{meas}} = H_{\max} - \bar H,\quad H_{\max}=\log 2^{n},

    penalising the entropy *shortfall* so richer Hilbert-space utilisation is
    encouraged.  Also reports **measurement diversity** :math:`\bar H/H_{\max}`
    and a **collapse fraction** (share of samples whose max probability exceeds a
    threshold).

    Args:
        n_qubits: Number of qubits (sets :math:`H_{\max}=\log 2^n`).
        collapse_threshold: :math:`\max_i p_i` above which a sample is "collapsed".

    Complexity:
        :math:`O(B\,2^{n})`.
    """

    def __init__(self, n_qubits: int, collapse_threshold: float = 0.9) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.h_max = float(math.log(2 ** n_qubits))
        self.collapse_threshold = float(collapse_threshold)

    def forward(self, probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Compute the measurement penalty + diagnostics.

        Args:
            probs: Basis-probability tensor of shape ``(B, 2**n)`` (e.g. the
                ``quantum_probs`` feature of :class:`xai.models.QAPINN`).

        Returns:
            Dict with ``"loss"`` (entropy shortfall), ``"mean_entropy"``,
            ``"diversity"`` and ``"collapse_fraction"``.
        """
        p = probs.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=1)  # (B,)
        mean_h = entropy.mean()
        loss = torch.clamp(self.h_max - mean_h, min=0.0)
        diversity = mean_h / (self.h_max + 1e-30)
        collapse = (probs.max(dim=1).values > self.collapse_threshold).float().mean()
        return {
            "loss": loss,
            "mean_entropy": mean_h,
            "diversity": diversity,
            "collapse_fraction": collapse,
        }


# =========================================================================== #
# 9. Stage III -- Effective-dimension analysis + regularisation
# =========================================================================== #
def effective_dimension_report(features: np.ndarray) -> Dict[str, float]:
    r"""Effective-dimension / intrinsic-dimension report (Sec. 14).

    Wraps :func:`xai.quantum_metrics.effective_dimension` (reused, not rewritten)
    and augments it with the covariance eigenvalue spectrum and the participation
    ratio / intrinsic-dimension diagnostics the framework calls for.

    Given the centred feature matrix :math:`X\in\mathbb{R}^{N\times d}` with
    covariance :math:`\Sigma=\tfrac1N X^\top X` and eigenvalues
    :math:`\{\lambda_i\}`:

    .. math::
        d_{\mathrm{eff}} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
        \quad(\text{participation ratio}),\qquad
        d_{\mathrm{intr}} = \#\{\text{PCs for }90\%\text{ variance}\}.

    (The participation ratio of the covariance eigenvalues equals that of the
    squared singular values used by :func:`effective_dimension`, so the two agree.)

    Args:
        features: Representation matrix of shape ``(N, d)`` (e.g.
            ``quantum_probs``).

    Returns:
        Dict with ``effective_dimension``, ``participation_ratio``,
        ``intrinsic_dimension``, ``covariance_eigenvalues``,
        ``eigenvalue_spectrum`` (normalised) and passthrough SVD fields.

    Complexity:
        :math:`O(N d^2 + d^3)`.
    """
    base = quantum_metrics.effective_dimension(features)  # reuse (SVD-based)
    X = np.asarray(features, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    N = max(X.shape[0], 1)
    cov = (Xc.T @ Xc) / N
    evals = np.linalg.eigvalsh(cov)[::-1]
    evals = np.clip(evals, 0.0, None)
    total = float(evals.sum()) + 1e-30
    pr = float((evals.sum() ** 2) / (np.sum(evals ** 2) + 1e-30))
    cum = np.cumsum(evals) / total
    intrinsic = int(np.searchsorted(cum, 0.90) + 1)
    return {
        "ambient_dim": int(X.shape[1]),
        "effective_dimension": base["effective_dimension"],
        "participation_ratio": pr,
        "intrinsic_dimension": intrinsic,
        "covariance_eigenvalues": evals.tolist(),
        "eigenvalue_spectrum": (evals / total).tolist(),
        "singular_values": base["singular_values"],
        "rank": base["rank"],
        "n_components_90pct": base["n_components_90pct"],
    }


def _effective_dimension_torch(features: torch.Tensor) -> torch.Tensor:
    """Differentiable participation-ratio effective dimension of ``(B, d)``."""
    Xc = features - features.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    energy = s ** 2
    return (energy.sum() ** 2) / (torch.sum(energy ** 2) + 1e-30)


# =========================================================================== #
# 10. Stage III -- Shock-aware adaptive sampling
# =========================================================================== #
def sample_collocation(
    n_points: int,
    strategy: str = "shock_aware",
    model: Optional[nn.Module] = None,
    nu: float = PHYSICS.nu,
    oversample: int = 8,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generate normalised collocation points with a chosen sampling strategy (Sec. 15).

    For nonlinear hyperbolic PDEs such as Burgers, collocation points should be
    concentrated where the physics is hard (the steep front), not spread
    uniformly.  Four strategies are provided, in increasing physical awareness:

    * ``"uniform"``   -- i.i.d. uniform on the normalised domain :math:`[-1,1]^2`.
    * ``"gradient"``  -- importance-sample by :math:`|\partial u/\partial x|`
      (needs ``model``): points cluster where the solution is steep.
    * ``"residual"``  -- importance-sample by the squared PDE residual
      :math:`(u_t + u u_x - \nu u_{xx})^2` (needs ``model``): points cluster
      where the model currently violates the physics.
    * ``"shock_aware"`` -- **Burgers default**: a mixture that draws
      :math:`\tfrac12` uniformly (domain coverage) and :math:`\tfrac12` by the
      gradient weight (front resolution), giving robust shock densification.

    Importance strategies use rejection over an ``oversample``:math:`\times`
    candidate pool.  All coordinates are returned in the normalised training
    frame :math:`[-1,1]` with ``requires_grad=True`` so they feed the PDE residual.

    Args:
        n_points: Number of collocation points to return.
        strategy: One of ``{"uniform","gradient","residual","shock_aware"}``.
        model: A QA-PINN / PINN taking normalised ``(x, t)``; required for the
            ``gradient``/``residual``/``shock_aware`` strategies.
        nu: Viscosity for the residual strategy.
        oversample: Candidate-pool multiplier for importance sampling.
        seed: RNG seed.
        device: Torch device for the returned tensors.

    Returns:
        Tuple ``(xc, tc)`` of shape ``(n_points, 1)`` normalised tensors with
        ``requires_grad=True``.

    Complexity:
        ``uniform``: :math:`O(n)`; importance strategies:
        :math:`O(\text{oversample}\cdot n)` model evaluations.
    """
    rng = np.random.default_rng(seed)

    def _uniform(m: int) -> Tuple[np.ndarray, np.ndarray]:
        return rng.uniform(-1, 1, (m, 1)), rng.uniform(-1, 1, (m, 1))

    def _to_t(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)

    if strategy == "uniform" or model is None:
        xc, tc = _uniform(n_points)
        return _to_t(xc), _to_t(tc)

    # ---- build a candidate pool and score it ------------------------------
    m = n_points * max(oversample, 1)
    xc, tc = _uniform(m)

    def _gradient_weight() -> np.ndarray:
        xt = torch.tensor(xc, dtype=torch.float32, device=device, requires_grad=True)
        tt = torch.tensor(tc, dtype=torch.float32, device=device, requires_grad=True)
        u = model(xt, tt)
        u_x = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=False)[0]
        return np.abs(u_x.detach().cpu().numpy().reshape(-1))

    def _residual_weight() -> np.ndarray:
        xt = torch.tensor(xc, dtype=torch.float32, device=device, requires_grad=True)
        tt = torch.tensor(tc, dtype=torch.float32, device=device, requires_grad=True)
        u = model(xt, tt)
        u_t = torch.autograd.grad(u, tt, torch.ones_like(u), create_graph=True)[0] * PHYSICS.st
        u_xh = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
        u_x = u_xh * PHYSICS.sx
        u_xx = torch.autograd.grad(u_xh, xt, torch.ones_like(u_xh), create_graph=True)[0] * PHYSICS.sx ** 2
        res = u_t + u * u_x - nu * u_xx
        return (res.detach().cpu().numpy().reshape(-1)) ** 2

    if strategy == "gradient":
        w = _gradient_weight()
    elif strategy == "residual":
        w = _residual_weight()
    elif strategy == "shock_aware":
        w = _gradient_weight()
    else:
        raise ValueError(f"unknown sampling strategy: {strategy!r}")

    w = w + 1e-8
    p = w / w.sum()

    if strategy == "shock_aware":
        # half uniform (coverage) + half importance (front resolution)
        n_imp = n_points // 2
        n_uni = n_points - n_imp
        idx_imp = rng.choice(m, size=n_imp, replace=False, p=p)
        xu, tu = _uniform(n_uni)
        x_sel = np.concatenate([xc[idx_imp], xu], axis=0)
        t_sel = np.concatenate([tc[idx_imp], tu], axis=0)
    else:
        idx = rng.choice(m, size=n_points, replace=False, p=p)
        x_sel, t_sel = xc[idx], tc[idx]

    return _to_t(x_sel), _to_t(t_sel)


# =========================================================================== #
# 11. Stage IV -- Unified Physics-Guided Loss
# =========================================================================== #
@dataclass
class PhysicsGuidedLossConfig:
    r"""Configuration for :class:`PhysicsGuidedLoss` -- every term switchable.

    The unified objective (report Stage IV) is

    .. math::
        \mathcal{L} = w_d\mathcal{L}_{\mathrm{data}}
                    + w_p\mathcal{L}_{\mathrm{PDE}}
                    + w_{ic}\mathcal{L}_{\mathrm{IC}}
                    + w_{bc}\mathcal{L}_{\mathrm{BC}}
                    + \lambda_{\mathrm{spec}}\mathcal{L}_{\mathrm{spec}}
                    + \lambda_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}}
                    + \lambda_{\mathrm{enc}}\mathcal{L}_{\mathrm{enc}}
                    + \lambda_{\mathrm{depth}}\mathcal{L}_{\mathrm{depth}}
                    + \lambda_{\mathrm{meas}}\mathcal{L}_{\mathrm{meas}}
                    + \lambda_{\mathrm{dim}}\mathcal{L}_{\mathrm{dim}} .

    Each ``use_*`` flag toggles a term independently; each ``w_*`` / ``lambda_*``
    is its coefficient.  The base PINN weights match the suite convention
    (``landscapes.W_*`` = ``1,1,10,10``).
    """

    # -- base PINN terms -----------------------------------------------------
    use_data: bool = True
    use_pde: bool = True
    use_ic: bool = True
    use_bc: bool = True
    w_data: float = 1.0
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 10.0
    # -- physics-guided regularisers ----------------------------------------
    use_spectral: bool = True
    use_entanglement: bool = False
    use_encoding: bool = True
    use_depth: bool = True
    use_measurement: bool = False
    use_effdim: bool = False
    lambda_spectral: float = 1e-2
    lambda_entanglement: float = 1e-1
    lambda_encoding: float = 1e-3
    lambda_depth: float = 1e-3
    lambda_measurement: float = 1e-2
    lambda_effdim: float = 1e-2
    # -- regulariser hyper-parameters ---------------------------------------
    effdim_target: float = 4.0
    target_depth: Optional[int] = None
    reuploading_blocks: int = 1
    recommended_reuploading: int = 4
    spectral_line_points: int = 128
    spectral_line_times: int = 8


class PhysicsGuidedLoss(nn.Module):
    r"""Unified Physics-Guided Loss for the Burgers QA-PINN (Stage IV).

    Assembles the four base PINN terms (data, PDE residual, IC, BC) and the six
    physics-guided regularisers (spectral, entanglement, encoding, circuit-depth,
    measurement, effective-dimension) into a single objective whose every term is
    independently switchable via :class:`PhysicsGuidedLossConfig`.  Each
    coefficient has a direct physical interpretation (report Stage IV), not an
    arbitrary tuning role.

    The module is built around the suite's :class:`xai.models.QAPINN` interface
    (``forward(x, t, return_features=True)`` and ``angles(x, t)``) but the base
    PINN terms work with any ``model(x, t)`` (e.g. the classical PINN).

    Args:
        model: The network being trained (QA-PINN or classical PINN).
        config: The loss configuration.
        shock_frequency: Shock band cutoff :math:`k_s` for the spectral term.
        entanglement_target: Target :math:`Q^\star` if the entanglement term is on.

    Complexity:
        Per call, dominated by the PDE residual (double autograd) and, if enabled,
        the entanglement term's state simulation (:math:`O(n_{\mathrm{ent}} 2^n)`).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[PhysicsGuidedLossConfig] = None,
        shock_frequency: float = 8.0,
        entanglement_target: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = config or PhysicsGuidedLossConfig()
        self._mse = nn.MSELoss()

        is_quantum = hasattr(model, "n_qubits")
        self.is_quantum = is_quantum

        self.spectral = SpectralRegularizer(
            mode="shock_preserve", shock_frequency=shock_frequency
        )
        self.entangler: Optional[EntanglementRegularizer] = None
        self.measurement: Optional[MeasurementRegularizer] = None
        if is_quantum:
            self.measurement = MeasurementRegularizer(model.n_qubits)
            if self.cfg.use_entanglement:
                self.entangler = EntanglementRegularizer(
                    model.n_qubits, model.n_layers, target=entanglement_target
                )

    # -- individual physics-guided terms ------------------------------------
    def _spectral_term(self) -> torch.Tensor:
        """Evaluate the model on a batch of spatial lines and penalise the tail."""
        n = self.cfg.spectral_line_points
        nt = self.cfg.spectral_line_times
        device = next(self.model.parameters()).device
        xs = torch.linspace(-1, 1, n, device=device)
        ts = torch.linspace(-1, 1, nt, device=device)
        XX, TT = torch.meshgrid(xs, ts, indexing="ij")  # (n, nt)
        x = XX.reshape(-1, 1)
        t = TT.reshape(-1, 1)
        u = self.model(x, t).reshape(n, nt).T  # (nt, n): one line per time
        return self.spectral(u)

    def _encoding_term(self) -> torch.Tensor:
        r"""Encoding regulariser: penalise a re-uploading depth below the
        physics-recommended value (report Sec. 11).  Constant w.r.t. weights but
        included so ablations can switch the term on/off consistently."""
        deficit = max(0, self.cfg.recommended_reuploading - self.cfg.reuploading_blocks)
        return torch.tensor(float(deficit) ** 2)

    def _depth_term(self) -> torch.Tensor:
        """Circuit-depth penalty (report Sec. 12)."""
        if not self.is_quantum:
            return torch.tensor(0.0)
        D_star = self.cfg.target_depth if self.cfg.target_depth is not None else self.model.n_layers
        pen = max(0, self.model.n_layers - int(D_star)) ** 2
        return torch.tensor(float(pen))

    # -- full objective ------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        r"""Compute the total loss and a per-term breakdown.

        Args:
            batch: Dict of normalised tensors with keys ``xd,td,ud`` (supervised),
                ``xc,tc`` (collocation, ``requires_grad=True``), ``xic,tic,uic``
                (initial condition) and ``xbc,tbc,ubc`` (boundary condition).
                Any block may be omitted if its term is disabled.

        Returns:
            Tuple ``(total_loss, components)`` where ``components`` maps each term
            name to its (unweighted) scalar value plus quantum diagnostics
            (``mean_Q``, ``diversity``, ``collapse_fraction``) when available.
        """
        cfg = self.cfg
        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        comp: Dict[str, float] = {}

        # ---- base PINN terms ----
        if cfg.use_data and "xd" in batch:
            l_data = self._mse(self.model(batch["xd"], batch["td"]), batch["ud"])
            total = total + cfg.w_data * l_data
            comp["L_data"] = float(l_data.item())

        if cfg.use_ic and "xic" in batch:
            l_ic = self._mse(self.model(batch["xic"], batch["tic"]), batch["uic"])
            total = total + cfg.w_ic * l_ic
            comp["L_ic"] = float(l_ic.item())

        if cfg.use_bc and "xbc" in batch:
            l_bc = self._mse(self.model(batch["xbc"], batch["tbc"]), batch["ubc"])
            total = total + cfg.w_bc * l_bc
            comp["L_bc"] = float(l_bc.item())

        if cfg.use_pde and "xc" in batch:
            xc, tc = batch["xc"], batch["tc"]
            u = self.model(xc, tc)
            u_th = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True)[0]
            u_xh = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
            u_xxh = torch.autograd.grad(u_xh, xc, torch.ones_like(u_xh), create_graph=True)[0]
            res = u_th * PHYSICS.st + u * (u_xh * PHYSICS.sx) - PHYSICS.nu * (u_xxh * PHYSICS.sx ** 2)
            l_pde = (res ** 2).mean()
            total = total + cfg.w_pde * l_pde
            comp["L_pde"] = float(l_pde.item())

        # ---- physics-guided regularisers ----
        if cfg.use_spectral:
            l_spec = self._spectral_term()
            total = total + cfg.lambda_spectral * l_spec
            comp["L_spectral"] = float(l_spec.item())

        if cfg.use_encoding:
            l_enc = self._encoding_term().to(device)
            total = total + cfg.lambda_encoding * l_enc
            comp["L_encoding"] = float(l_enc.item())

        if cfg.use_depth:
            l_depth = self._depth_term().to(device)
            total = total + cfg.lambda_depth * l_depth
            comp["L_depth"] = float(l_depth.item())

        if self.is_quantum and (cfg.use_measurement or cfg.use_effdim or cfg.use_entanglement):
            # one feature pass shared by the quantum regularisers
            xc = batch.get("xc")
            tc = batch.get("tc")
            if xc is None:
                xc, tc = sample_collocation(256, "uniform", device=device)
            _, feats = self.model(xc, tc, return_features=True)
            probs = feats["quantum_probs"]

            if cfg.use_measurement and self.measurement is not None:
                mres = self.measurement(probs)
                total = total + cfg.lambda_measurement * mres["loss"]
                comp["L_measurement"] = float(mres["loss"].item())
                comp["diversity"] = float(mres["diversity"].item())
                comp["collapse_fraction"] = float(mres["collapse_fraction"].item())

            if cfg.use_effdim:
                d_eff = _effective_dimension_torch(probs)
                l_dim = torch.clamp(torch.tensor(cfg.effdim_target, device=device) - d_eff, min=0.0) ** 2
                total = total + cfg.lambda_effdim * l_dim
                comp["L_effdim"] = float(l_dim.item())
                comp["effective_dim"] = float(d_eff.item())

            if cfg.use_entanglement and self.entangler is not None:
                angles = self.model.angles(xc, tc)
                eres = self.entangler(angles, self.model.qlayer.weights)
                total = total + cfg.lambda_entanglement * eres["loss"]
                comp["L_entanglement"] = float(eres["loss"].item())
                comp["mean_Q"] = float(eres["mean_Q"].item())

        comp["total"] = float(total.item())
        return total, comp


# =========================================================================== #
# 12. Stage V -- Unified Burgers analyzer (orchestrates existing XAI metrics)
# =========================================================================== #
class BurgersPhysicsAnalyzer:
    r"""Single-interface orchestrator over the existing XAI suite (Stage V).

    This class does **not** re-implement any explainability metric; it wires the
    already-implemented routines in :mod:`xai.expressivity`,
    :mod:`xai.quantum_metrics` and :mod:`xai.landscapes` behind one Burgers-aware
    interface, and adds the Physics-Guided descriptor / spectral diagnostics from
    this module.

    Typical use (see ``Physics_Guided_Burgers_QAPINN.ipynb``)::

        az = BurgersPhysicsAnalyzer.from_checkpoints()
        desc = az.descriptor()
        rec  = az.recommend()
        out  = az.run_all()          # every analysis, guarded

    Args:
        fb: The QA-PINN :class:`xai.common.FeatureBundle`.
        qapinn: A loaded QA-PINN model (defaults to the best/3-qubit checkpoint).
        classical: A loaded classical PINN (optional, for comparison).
    """

    def __init__(
        self,
        fb: FeatureBundle,
        qapinn: Optional[nn.Module] = None,
        classical: Optional[nn.Module] = None,
    ) -> None:
        self.fb = fb
        self.qapinn = qapinn
        self.classical = classical
        self._desc: Optional[BurgersDescriptor] = None

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_checkpoints(cls, n_qubits: Optional[int] = None) -> "BurgersPhysicsAnalyzer":
        """Load the feature bundle + best QA-PINN + classical PINN from disk."""
        from .common import load_features

        fb = load_features()
        q = n_qubits or fb.n_qubits
        qmodel, _ = models.load_qapinn(q)
        try:
            cmodel, _ = models.load_classical()
        except Exception:  # pragma: no cover
            cmodel = None
        return cls(fb, qmodel, cmodel)

    # -- Stage I / II --------------------------------------------------------
    def descriptor(self) -> BurgersDescriptor:
        """Stage-I descriptor computed from the FEM reference field (cached)."""
        if self._desc is None:
            self._desc = burgers_descriptor(self.fb.U_grid, self.fb.x_grid, nu=PHYSICS.nu)
        return self._desc

    def recommend(self) -> ArchitectureRecommendation:
        """Stage-II architecture recommendation from the descriptor."""
        return recommend_architecture(self.descriptor())

    # -- Stage V diagnostics (each delegates to the existing suite) ----------
    def spectral_diagnostics(self) -> Dict:
        """Spectral complexity, shock frequency and per-model output spectra.

        Reuses :func:`xai.expressivity.compare_spectra` for the model spectra and
        this module's :func:`spectral_complexity` / :func:`shock_frequency` for the
        physics descriptors.
        """
        freqs, P = power_spectrum(self.fb.U_grid, dx=float(np.mean(np.diff(self.fb.x_grid))))
        d = self.descriptor()
        out = {
            "spectral_complexity": d.spectral_complexity_value,
            "shock_frequency": d.shock_frequency_value,
            "shock_indicator": d.shock_indicator,
            "reference_freqs": freqs.tolist(),
            "reference_power": P.tolist(),
        }
        try:
            out["model_spectra"] = expressivity.compare_spectra(
                qubit_range=(self.fb.n_qubits,), axis="x", fixed=0.5, n=512
            )
        except Exception as exc:  # pragma: no cover
            out["model_spectra"] = {"error": str(exc)}
        return out

    def fourier_analysis(self, axis: str = "x") -> Dict:
        """Fourier spectra of every model (reuses ``expressivity.compare_spectra``)."""
        return expressivity.compare_spectra(qubit_range=(3, 4, 5), axis=axis, fixed=0.5, n=512)

    def pca(self, max_points: int = 4000, seed: int = 0) -> Dict:
        """2-component PCA of the quantum-probability features (reuses sklearn)."""
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(seed)
        X = self.fb.quantum_probs
        u = self.fb.output.reshape(-1)
        idx = rng.choice(X.shape[0], size=min(max_points, X.shape[0]), replace=False)
        p = PCA(n_components=2).fit(X[idx])
        return {
            "embedding": p.transform(X[idx]).tolist(),
            "color_u": u[idx].tolist(),
            "explained_variance_ratio": p.explained_variance_ratio_.tolist(),
        }

    def tsne(self, max_points: int = 1500, seed: int = 0) -> Dict:
        """2-D t-SNE of the quantum-probability features (reuses sklearn)."""
        from sklearn.manifold import TSNE

        rng = np.random.default_rng(seed)
        X = self.fb.quantum_probs
        u = self.fb.output.reshape(-1)
        idx = rng.choice(X.shape[0], size=min(max_points, X.shape[0]), replace=False)
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   learning_rate="auto", random_state=seed).fit_transform(X[idx])
        return {"embedding": emb.tolist(), "color_u": u[idx].tolist()}

    def hessian(self, **kwargs) -> Dict:
        """Hessian spectrum at convergence (reuses ``landscapes.hessian_spectrum``)."""
        from . import landscapes

        model = self.qapinn or self.classical
        return landscapes.hessian_spectrum(model, self.fb, **kwargs)

    def entanglement(self, n_samples: int = 256, seed: int = 0) -> Dict:
        """Meyer-Wallach + bipartite entropy over the domain (reuses ``quantum_metrics``)."""
        if self.qapinn is None:
            return {"error": "no QA-PINN loaded"}
        return quantum_metrics.entanglement_over_domain(
            self.qapinn, self.fb, n_samples=n_samples, seed=seed
        )

    def effective_dimension(self, seed: int = 0) -> Dict:
        """Layer-wise information bottleneck + augmented eff-dim report (reused)."""
        ib = quantum_metrics.information_bottleneck(self.fb, seed=seed)
        ib["quantum_probs_report"] = effective_dimension_report(self.fb.quantum_probs)
        return ib

    def gradient_variance(self, seed: int = 0) -> Dict:
        """Barren-plateau gradient-variance scan (reuses ``landscapes``)."""
        from . import landscapes

        return landscapes.barren_plateau_scan((3, 4, 5), seed=seed)

    def circuit_complexity(self) -> Dict:
        """Circuit-complexity + efficiency metrics for the loaded QA-PINN."""
        if self.qapinn is None:
            return {"error": "no QA-PINN loaded"}
        ed = effective_dimension_report(self.fb.quantum_probs)["effective_dimension"]
        rec = self.recommend()
        return circuit_complexity_metrics(
            self.qapinn.n_qubits, self.qapinn.n_layers,
            reuploading_blocks=1, effective_dim=ed, target_depth=rec.circuit_depth,
        )

    def run_all(self, seed: int = 0, skip_hessian: bool = False) -> Dict:
        """Run the whole Stage-V pipeline, guarding each step.

        Returns:
            Dict keyed by analysis name; failures are captured as ``{"error": ...}``
            so one broken step never aborts the report.
        """
        def _guard(fn, *a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # pragma: no cover
                return {"error": str(exc)}

        out: Dict = {
            "descriptor": self.descriptor().as_vector(),
            "recommendation": asdict(self.recommend()),
            "spectral_diagnostics": _guard(self.spectral_diagnostics),
            "fourier": _guard(self.fourier_analysis),
            "pca": _guard(self.pca),
            "tsne": _guard(self.tsne),
            "entanglement": _guard(self.entanglement),
            "effective_dimension": _guard(self.effective_dimension, seed),
            "gradient_variance": _guard(self.gradient_variance, seed),
            "circuit_complexity": _guard(self.circuit_complexity),
        }
        if not skip_hessian:
            out["hessian"] = _guard(self.hessian, seed=seed)
        return out
