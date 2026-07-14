r"""
xai.visualizer
=============

Publication-quality Matplotlib rendering for every XAI analysis.  All functions
take the *numerical results* produced by the analysis modules and write a PNG to
``output/xai/``; they never re-run models, so plotting is fast and decoupled.

Design notes
------------
* A single muted, colour-blind-safe categorical palette is used across all
  figures so the QA-PINN qubit series and the classical baseline are visually
  consistent from plot to plot (classical is always the dashed grey reference).
* Figures are saved at 150 dpi with tight bounding boxes.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Shared style
# --------------------------------------------------------------------------- #
PALETTE = {
    "3q": "#4C72B0",   # blue
    "4q": "#55A868",   # green
    "5q": "#C44E52",   # red
    "classical": "#4D4D4D",  # grey (dashed reference)
    "accent": "#8172B3",
}
QUBIT_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _color_for(label: str, idx: int = 0) -> str:
    low = label.lower()
    if "classical" in low:
        return PALETTE["classical"]
    for k in ("3q", "4q", "5q"):
        if k in low:
            return PALETTE[k]
    return QUBIT_COLORS[idx % len(QUBIT_COLORS)]


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1. Fourier spectra
# --------------------------------------------------------------------------- #
def plot_fourier_spectra(spectra: Dict, out_dir: str, axis_label: str = "x") -> str:
    """Overlay output power spectra + quantum-probe spectra for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (label, r) in enumerate(spectra.items()):
        if label.startswith("_") or "error" in r:
            continue
        freqs = np.asarray(r["freqs"])
        p = np.asarray(r["output_power"])
        color = _color_for(label, i)
        style = "--" if "classical" in label.lower() else "-"
        ax1.semilogy(freqs, p + 1e-12, style, color=color, label=label, lw=2)
        if "quantum_probs_power" in r:
            ax2.semilogy(freqs, np.asarray(r["quantum_probs_power"]) + 1e-12,
                         style, color=color, label=label, lw=2)

    ax1.set_title(f"Network output power spectrum (sweep in {axis_label})")
    ax1.set_xlabel(f"frequency [cycles per unit {axis_label}]")
    ax1.set_ylabel("normalised power")
    ax1.legend(fontsize=9)
    ax1.set_ylim(1e-8, 2)

    ax2.set_title("Quantum feature-map power spectrum (probs)")
    ax2.set_xlabel(f"frequency [cycles per unit {axis_label}]")
    ax2.set_ylabel("normalised power")
    ax2.legend(fontsize=9)
    ax2.set_ylim(1e-8, 2)

    fig.suptitle("Fourier Expressivity: accessible frequencies", fontsize=13, y=1.02)
    return _save(fig, out_dir, f"fourier_spectra_{axis_label}.png")


def plot_max_frequency_bar(spectra: Dict, out_dir: str) -> str:
    """Bar chart of max output frequency + quantum-probe max frequency."""
    table = spectra.get("_summary_table", [])
    if not table:
        return ""
    labels = [row["model"] for row in table]
    out_f = [row["max_frequency"] for row in table]
    q_f = [row.get("quantum_max_frequency", np.nan) for row in table]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38
    ax.bar(x - w / 2, out_f, w, label="output spectrum", color=PALETTE["3q"])
    ax.bar(x + w / 2, q_f, w, label="quantum feature-map", color=PALETTE["5q"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("max accessible frequency (99% energy)")
    ax.set_title("Maximum Accessible Frequency vs. Model")
    ax.legend()
    return _save(fig, out_dir, "max_frequency_bar.png")


# --------------------------------------------------------------------------- #
# 2. Entanglement
# --------------------------------------------------------------------------- #
def plot_entanglement_trends(ent: Dict, out_dir: str) -> str:
    """Meyer-Wallach Q and mean bipartite entropy vs. qubit count."""
    qs = sorted(int(k) for k in ent.keys() if k.isdigit() and "error" not in ent[k])
    mw = [ent[str(q)]["meyer_wallach_mean"] for q in qs]
    mw_e = [ent[str(q)]["meyer_wallach_std"] for q in qs]
    se = [ent[str(q)]["bipartite_entropy_mean"] for q in qs]
    se_e = [ent[str(q)]["bipartite_entropy_std"] for q in qs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(qs, mw, yerr=mw_e, marker="o", lw=2, capsize=4,
                color=PALETTE["3q"], label="Meyer-Wallach $Q$")
    ax.errorbar(qs, se, yerr=se_e, marker="s", lw=2, capsize=4,
                color=PALETTE["5q"], label="mean bipartite $S(\\rho_A)$ [bits]")
    ax.set_xlabel("number of qubits $n$")
    ax.set_ylabel("entanglement")
    ax.set_xticks(qs)
    ax.set_title("Entanglement of the Encoded State over the Burgers Domain")
    ax.legend()
    return _save(fig, out_dir, "entanglement_trends.png")


def plot_entanglement_dynamics(dyn: Dict, out_dir: str) -> str:
    """Entanglement vs. encoder-homotopy factor alpha (optimisation proxy)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (q, r) in enumerate(sorted(dyn.items())):
        if "error" in r:
            continue
        a = r["alphas"]
        ax.plot(a, r["meyer_wallach_curve"], marker="o", lw=2,
                color=QUBIT_COLORS[i], label=f"{q} qubits ($Q$)")
    ax.axvline(1.0, color="k", ls=":", alpha=0.6, label="trained operating point")
    ax.set_xlabel(r"encoder homotopy factor $\alpha$  ($W=\alpha W^\star$)")
    ax.set_ylabel("mean Meyer-Wallach $Q$")
    ax.set_title("Entanglement Generation as the Classical Encoder Scales")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "entanglement_dynamics.png")


# --------------------------------------------------------------------------- #
# 3. Information bottleneck (SVD / effective dimension)
# --------------------------------------------------------------------------- #
def plot_information_bottleneck(ib: Dict, out_dir: str) -> str:
    """Two panels: singular-value spectra per layer + effective-dim pipeline."""
    layers = ib["layers"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, name in enumerate(layers):
        s = np.asarray(ib["per_layer"][name]["singular_values"])
        s = s / (s.max() + 1e-30)
        ax1.semilogy(range(1, len(s) + 1), s + 1e-12, marker="o",
                     color=QUBIT_COLORS[i % len(QUBIT_COLORS)], label=name)
    ax1.set_xlabel("singular-value index")
    ax1.set_ylabel("normalised singular value")
    ax1.set_title("Layer-wise Singular-Value Spectra")
    ax1.legend(fontsize=9)

    d_eff = ib["effective_dimension_sequence"]
    norm = ib["normalised_effective_dimension"]
    x = np.arange(len(layers))
    ax2.plot(x, d_eff, "-o", color=PALETTE["3q"], lw=2, label="effective dim $d_{eff}$")
    ax2b = ax2.twinx()
    ax2b.plot(x, norm, "--s", color=PALETTE["5q"], lw=2, label="$d_{eff}$ / ambient")
    bi = layers.index(ib["bottleneck_layer"])
    ax2.axvline(bi, color="k", ls=":", alpha=0.6)
    ax2.annotate("bottleneck", (bi, d_eff[bi]), textcoords="offset points",
                 xytext=(6, 10), fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, rotation=20, fontsize=9)
    ax2.set_ylabel("effective dimension")
    ax2b.set_ylabel("normalised effective dimension")
    ax2.set_title("Information Flow through the Hybrid Pipeline")
    ax2.grid(False)
    ax2b.grid(False)
    lines1, l1 = ax2.get_legend_handles_labels()
    lines2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, l1 + l2, fontsize=9, loc="upper right")

    fig.suptitle("Measurement-Operator Information Bottleneck", fontsize=13, y=1.02)
    return _save(fig, out_dir, "information_bottleneck.png")


# --------------------------------------------------------------------------- #
# 4. Feature embeddings (PCA / t-SNE)
# --------------------------------------------------------------------------- #
def plot_feature_embeddings(fb, out_dir: str, max_points: int = 4000, seed: int = 0) -> str:
    """PCA and t-SNE of the quantum-probability features coloured by prediction u.

    Shows how the ``2**n`` measurement channels organise the (x,t) domain into a
    low-dimensional manifold -- the quantum layer acting as a non-linear feature
    extractor.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    probs = fb.quantum_probs
    u = fb.output.reshape(-1)
    N = probs.shape[0]
    idx = rng.choice(N, size=min(max_points, N), replace=False)
    X = probs[idx]
    c = u[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    p2 = PCA(n_components=2).fit(X)
    XY = p2.transform(X)
    sc0 = axes[0].scatter(XY[:, 0], XY[:, 1], c=c, s=6, cmap="coolwarm")
    axes[0].set_title(
        f"PCA of quantum probs  (EVR={p2.explained_variance_ratio_[:2].sum():.2f})")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    fig.colorbar(sc0, ax=axes[0], label="predicted $u$")

    try:
        from sklearn.manifold import TSNE

        sub = min(2000, len(idx))
        sidx = rng.choice(len(idx), size=sub, replace=False)
        ts = TSNE(n_components=2, perplexity=30, init="pca",
                  learning_rate="auto", random_state=seed).fit_transform(X[sidx])
        sc1 = axes[1].scatter(ts[:, 0], ts[:, 1], c=c[sidx], s=6, cmap="coolwarm")
        axes[1].set_title(f"t-SNE of quantum probs (n={sub})")
        axes[1].set_xlabel("dim 1"); axes[1].set_ylabel("dim 2")
        fig.colorbar(sc1, ax=axes[1], label="predicted $u$")
    except Exception as exc:  # pragma: no cover
        axes[1].text(0.5, 0.5, f"t-SNE unavailable:\n{exc}", ha="center", va="center")

    fig.suptitle("Quantum Measurement Layer as a Non-linear Feature Extractor",
                 fontsize=13, y=1.02)
    return _save(fig, out_dir, "feature_embeddings.png")


# --------------------------------------------------------------------------- #
# 5. Barren plateaus
# --------------------------------------------------------------------------- #
def plot_barren_plateau(bp: Dict, out_dir: str) -> str:
    """Gradient variance vs. qubits (log scale) with the 2^-n reference law."""
    qs = np.asarray(bp["qubits"], dtype=float)
    var = np.asarray(bp["variances"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(qs, var, "o-", color=PALETTE["5q"], lw=2, ms=9,
                label=r"empirical Var$[\partial_\theta \langle Z_0\rangle]$")

    # ideal barren-plateau reference anchored at the first point
    ref = var[0] * (0.5 ** (qs - qs[0]))
    ax.semilogy(qs, ref, "k--", lw=1.8, label=r"$\mathcal{O}(2^{-n})$ reference")

    # fitted decay
    fit = bp["fit"]
    fitted = np.exp(fit["intercept"] + fit["log_slope"] * qs)
    ax.semilogy(qs, fitted, ":", color=PALETTE["3q"], lw=1.8,
                label=f"fit (decay {fit['decay_factor_per_qubit']:.2f}/qubit)")

    ax.set_xlabel("number of qubits $n$")
    ax.set_ylabel("gradient variance")
    ax.set_xticks(qs)
    ax.set_title("Barren-Plateau Diagnostic: Gradient Variance vs. Qubit Count")
    ax.legend()
    return _save(fig, out_dir, "barren_plateau.png")


# --------------------------------------------------------------------------- #
# 6. Loss landscapes
# --------------------------------------------------------------------------- #
def plot_loss_slices_1d(slices: Dict, out_dir: str) -> str:
    """Overlay 1-D filter-normalised loss profiles for all models."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (label, r) in enumerate(slices.items()):
        if "error" in r:
            continue
        a = np.asarray(r["alphas"])
        L = np.asarray(r["losses"])
        color = _color_for(label, i)
        style = "--" if "classical" in label.lower() else "-"
        ax.plot(a, L, style, color=color, lw=2,
                label=f"{label} (curv={r['local_curvature']:.2g})")
    ax.set_xlabel(r"filter-normalised displacement $\alpha$")
    ax.set_ylabel("composite PINN loss")
    ax.set_yscale("log")
    ax.set_title("1-D Loss-Landscape Slices (filter-normalised)")
    ax.legend(fontsize=9)
    return _save(fig, out_dir, "loss_slices_1d.png")


def plot_loss_surfaces_2d(surfaces: Dict, out_dir: str) -> str:
    """Grid of 2-D loss-surface contour maps, one per model."""
    items = [(k, v) for k, v in surfaces.items() if "error" not in v]
    n = len(items)
    if n == 0:
        return ""
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.2 * rows), squeeze=False)
    for ax_i, (label, r) in enumerate(items):
        ax = axes[ax_i // cols][ax_i % cols]
        g = np.asarray(r["grid"])
        Z = np.asarray(r["loss_surface"])
        X, Y = np.meshgrid(g, g, indexing="ij")
        cf = ax.contourf(X, Y, np.log10(Z + 1e-12), levels=25, cmap="viridis")
        ax.contour(X, Y, np.log10(Z + 1e-12), levels=12, colors="white",
                   linewidths=0.4, alpha=0.5)
        ax.plot(0, 0, "r*", ms=14)
        ax.set_title(f"{label}   (center L={r['center_loss']:.3g})")
        ax.set_xlabel(r"$\alpha$ (dir 1)")
        ax.set_ylabel(r"$\beta$ (dir 2)")
        fig.colorbar(cf, ax=ax, label=r"$\log_{10}$ loss")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("2-D Loss Landscapes (filter-normalised): QA-PINN vs. Classical",
                 fontsize=13, y=1.01)
    return _save(fig, out_dir, "loss_surfaces_2d.png")


def plot_hessian_spectrum(hess: Dict, out_dir: str) -> str:
    """Bar chart comparing extreme Hessian eigenvalues / sharpness across models."""
    items = [(k, v) for k, v in hess.items() if "error" not in v]
    if not items:
        return ""
    labels = [k for k, _ in items]
    lam_max = [v["lambda_max"] for _, v in items]
    lam_min = [v["lambda_min"] for _, v in items]
    cond = [v["condition_number"] for _, v in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, lam_max, 0.4, label=r"$\lambda_{max}$", color=PALETTE["5q"])
    ax1.bar(x + 0.2, lam_min, 0.4, label=r"$\lambda_{min}$", color=PALETTE["3q"])
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("eigenvalue")
    ax1.set_title("Extreme Hessian Eigenvalues (curvature)")
    ax1.legend()

    ax2.bar(x, cond, 0.5, color=PALETTE["accent"])
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel(r"$|\lambda_{max}| / |\lambda_{min}|$")
    ax2.set_title("Hessian Condition Number (ill-conditioning)")

    fig.suptitle("Loss-Curvature Comparison at Convergence", fontsize=13, y=1.02)
    return _save(fig, out_dir, "hessian_spectrum.png")


# --------------------------------------------------------------------------- #
# 7. Qubit scaling vs. expressivity
# --------------------------------------------------------------------------- #
def plot_qubit_scaling(sweep: Dict, extra: Optional[Dict], out_dir: str) -> str:
    """Convergence / residual loss / training cost vs. qubit count."""
    results = sweep.get("results", [])
    if not results:
        return ""
    qs = [r["n_qubits"] for r in results]
    total = [r["final_total"] for r in results]
    pde = [r["L_pde"] for r in results]
    rel = [r["rel_l2_vs_fem"] for r in results]
    tt = [r["train_time_s"] for r in results]
    nparams = [r["n_params"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(qs, total, "o-", color=PALETTE["3q"], lw=2, label="total loss")
    axes[0].plot(qs, pde, "s--", color=PALETTE["5q"], lw=2, label="PDE residual")
    axes[0].set_xlabel("qubits $n$"); axes[0].set_ylabel("final loss")
    axes[0].set_title("Converged Loss vs. Qubits"); axes[0].legend()
    axes[0].set_xticks(qs)

    axes[1].plot(qs, rel, "o-", color=PALETTE["accent"], lw=2)
    axes[1].set_xlabel("qubits $n$"); axes[1].set_ylabel(r"relative $L_2$ vs FEM")
    axes[1].set_title("Solution Accuracy vs. Qubits")
    axes[1].set_xticks(qs)

    ax2b = axes[2].twinx()
    axes[2].plot(qs, tt, "o-", color=PALETTE["4q"], lw=2, label="train time [s]")
    ax2b.plot(qs, nparams, "s--", color="#888", lw=2, label="parameters")
    axes[2].set_xlabel("qubits $n$")
    axes[2].set_ylabel("train time [s]", color=PALETTE["4q"])
    ax2b.set_ylabel("parameter count", color="#888")
    axes[2].set_title("Training Cost / Model Size vs. Qubits")
    axes[2].set_xticks(qs)
    axes[2].grid(False); ax2b.grid(False)

    fig.suptitle("Qubit Scaling vs. Problem Expressivity", fontsize=13, y=1.03)
    return _save(fig, out_dir, "qubit_scaling.png")


def plot_solution_fields(fb, out_dir: str) -> str:
    """Reference FEM field vs. QA-PINN field vs. pointwise error heatmaps."""
    X, T = np.meshgrid(fb.x_grid, fb.t_grid)
    err = np.abs(fb.U_qapinn - fb.U_grid)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, field, title, cmap in [
        (axes[0], fb.U_grid, "FEM reference $u(x,t)$", "viridis"),
        (axes[1], fb.U_qapinn, "QA-PINN prediction", "viridis"),
        (axes[2], err, "absolute error", "magma"),
    ]:
        pc = ax.pcolormesh(X, T, field, shading="auto", cmap=cmap)
        ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(title)
        fig.colorbar(pc, ax=ax)
    fig.suptitle("QA-PINN Solution Field vs. FEM Reference", fontsize=13, y=1.03)
    return _save(fig, out_dir, "solution_fields.png")
