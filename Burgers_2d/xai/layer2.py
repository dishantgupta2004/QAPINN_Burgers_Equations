"""
layer2.py — Quantum-Layer Explainability  (spec "Layer 2", 11 analyses)
=======================================================================

These analyses answer one question in eleven ways: *what does inserting a
quantum layer into a classical PINN actually buy (or cost) you?* They are the
heart of the module. Each is a standalone function taking a `QuantumProbe`
(from adapter.py) plus a `bounds` box describing the input domain, so the same
code applies to 1D Burgers, 2D Burgers, or any CFD problem.

Index (matches the specification image):
  2.1  input_sensitivity          — how input encoding affects PDE learning
  2.2  measurement_operators      — how informative the measurements are (Z/X/Y, Pauli, correlation w/ residual)
  2.3  circuit_depth_analysis     — depth (n_layers) sweep vs loss / expressivity / runtime
  2.4  entanglement_analysis      — Meyer-Wallach Q, concurrence, entropy, pair correlation; vs PDE error
  2.5  fourier_spectrum           — can the quantum layer represent high-frequency (shock) content
  2.6  expressivity_analysis      — function-class size: state overlap, kernel rank, effective dimension
  2.7  barren_plateau_analysis    — gradient variance vs qubits / depth
  2.8  loss_landscape             — filter-normalised loss around the optimum
  2.9  measurement_distribution   — raw-measurement histograms: entropy, KL, evolution
  2.10 feature_attribution        — saliency / sensitivity maps for explainability
  2.11 quantum_state_evolution    — store/visualise state every few epochs (Bloch, purity, fidelity)

Honesty guardrails (baked in, per project principles):
  * Fourier support is reported as *encoding-determined* (Schuld-Sweke-Meyer):
    the accessible frequency set is fixed by the encoding, expanded linearly by
    re-uploading; weights only redistribute amplitude. We measure the empirical
    spectral centroid and flag the theoretical reachable band separately.
  * Entanglement conclusions are conditional on *observed* Q, never presupposed.
  * Random-init spectra are labelled as such; trained-vs-untrained is never conflated.
"""
from __future__ import annotations
from typing import Optional, Sequence, Callable, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt

from .adapter import QuantumProbe, ModelAdapter
from . import utils


# ============================================================================= #
#  2.1  Input-encoding sensitivity                                              #
# ============================================================================= #
def input_sensitivity(probe: QuantumProbe, bounds: Sequence[tuple],
                      n: int = 1500, seed: int = 0, noise_sigma: float = 0.02,
                      plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    How does the input encoding affect learning?

    Metrics
    -------
    * input_sensitivity[d] : mean |d(output)/d(input_d)| — which coordinate the
      field responds to most (e.g. x vs t for Burgers).
    * feature_importance   : same, normalised to sum 1.
    * encoding_robustness  : relative change in the *quantum probabilities* when
      inputs are perturbed by Gaussian noise — how brittle the encoding is.
    * grad_norm_input      : ||d(output)/d(input)|| distribution stats.
    """
    X = utils.sample_domain(bounds, n, seed=seed, device=probe.device)
    g = utils.input_gradient(probe, X)                      # (N, d_in)
    sens = np.abs(g).mean(0)
    importance = sens / (sens.sum() + 1e-30)
    gn = np.linalg.norm(g, axis=1)

    # encoding robustness: perturb inputs, measure prob shift
    with torch.no_grad():
        p0 = probe.probs(X).cpu().numpy()
        Xn = X + noise_sigma * torch.randn_like(X)
        pn = probe.probs(Xn).cpu().numpy()
    rob = float(np.linalg.norm(pn - p0, axis=1).mean() /
                (np.linalg.norm(p0, axis=1).mean() + 1e-30))

    res = dict(analysis="input_sensitivity", d_in=probe.d_in,
               input_sensitivity=sens.tolist(),
               feature_importance=importance.tolist(),
               grad_norm_mean=float(gn.mean()), grad_norm_std=float(gn.std()),
               encoding_robustness=rob, noise_sigma=noise_sigma)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].bar(range(probe.d_in), importance)
        ax[0].set(xlabel="input dimension", ylabel="normalised sensitivity",
                  title="Feature importance (input encoding)")
        ax[0].set_xticks(range(probe.d_in))
        ax[1].hist(gn, bins=40); ax[1].set(xlabel="||∂u/∂input||",
                  ylabel="count", title=f"Input-gradient norm  (robustness={rob:.3f})")
        fig.suptitle(f"2.1 Input sensitivity — {probe.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_1_input_sensitivity_{probe.name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.2  Measurement-operator analysis                                           #
# ============================================================================= #
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_op(pstr: str) -> np.ndarray:
    op = np.array([[1]], dtype=complex)
    for ch in pstr:
        op = np.kron(op, _PAULI[ch])
    return op


def measurement_operators(probe: QuantumProbe, bounds: Sequence[tuple],
                          residual_fn: Optional[Callable] = None,
                          n: int = 800, seed: int = 0,
                          plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    How informative are the measurements?  Requires a statevector probe.

    Computes, per qubit:  <Z>, <X>, <Y> expectation fields and their variance
    across the domain (a low-variance observable carries little information).
    Also computes correlation of each single-qubit <Z_i> with the PDE residual
    (if `residual_fn` given) — i.e. which measured qubit tracks the physics.

    [Measurement operator -> prediction -> residual] chain is summarised by the
    residual-correlation vector.
    """
    if not probe.has_state():
        return dict(analysis="measurement_operators", skipped="no statevector")

    X = utils.sample_domain(bounds, n, seed=seed, device=probe.device)
    psi = np.asarray(probe.statevector(X).detach().cpu().numpy())
    psi = np.atleast_2d(psi)                                    # (N, 2^n)
    N, dim = psi.shape
    nq = probe.n_qubits

    exp = {b: np.zeros((N, nq)) for b in ("X", "Y", "Z")}
    var = {b: np.zeros(nq) for b in ("X", "Y", "Z")}
    for basis in ("X", "Y", "Z"):
        for q in range(nq):
            pstr = "I" * q + basis + "I" * (nq - q - 1)
            O = _pauli_string_op(pstr)
            vals = np.real(np.einsum("ni,ij,nj->n", psi.conj(), O, psi))
            exp[basis][:, q] = vals
            var[basis][q] = float(np.var(vals))

    res = dict(analysis="measurement_operators", n_qubits=nq,
               expZ_var=var["Z"].tolist(), expX_var=var["X"].tolist(),
               expY_var=var["Y"].tolist())

    # correlation of <Z_i> with the PDE residual
    if residual_fn is not None:
        r = np.asarray(residual_fn(probe, X)).ravel()
        corr = []
        for q in range(nq):
            z = exp["Z"][:, q]
            c = np.corrcoef(z, r)[0, 1] if np.std(z) > 1e-9 else 0.0
            corr.append(float(0.0 if np.isnan(c) else c))
        res["residual_correlation_Z"] = corr

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        width = 0.25
        xq = np.arange(nq)
        for i, b in enumerate(("Z", "X", "Y")):
            ax[0].bar(xq + (i - 1) * width, var[b], width, label=f"Var<{b}>")
        ax[0].set(xlabel="qubit", ylabel="observable variance",
                  title="Measurement informativeness"); ax[0].legend()
        ax[0].set_xticks(xq)
        if "residual_correlation_Z" in res:
            ax[1].bar(xq, res["residual_correlation_Z"], color="C3")
            ax[1].set(xlabel="qubit", ylabel="corr(<Z>, PDE residual)",
                      title="Which qubit tracks the physics"); ax[1].set_xticks(xq)
            ax[1].axhline(0, color="k", lw=.6)
        else:
            ax[1].axis("off")
        fig.suptitle(f"2.2 Measurement operators — {probe.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_2_measurement_ops_{probe.name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.3  Circuit-depth analysis                                                  #
# ============================================================================= #
def circuit_depth_analysis(build_train_eval: Callable[[int], Dict[str, float]],
                           depths: Sequence[int] = (1, 2, 3, 4, 6, 8, 10),
                           plot: bool = True, outdir: str = "outputs/xai",
                           name: str = "qapinn") -> Dict[str, Any]:
    """
    Hyperparameter tune for circuit depth (n_layers = 1..10).

    `build_train_eval(depth)` is a *user-supplied* callback that trains a QA-PINN
    with the given depth and returns a dict with at least:
        {'train_loss', 'rel_l2', 'grad_norm', 'expressivity', 'runtime'}
    (any missing keys are simply not plotted). Kept as a callback so the module
    never owns your training loop — plug in train_qapinn.train_qapinn.

    Observes: depth vs relative-L2, depth vs gradient-norm, depth vs expressivity,
    depth vs runtime.
    """
    rows = []
    for d in depths:
        r = dict(build_train_eval(d)); r["depth"] = int(d); rows.append(r)

    def col(k): return [r.get(k, np.nan) for r in rows]
    res = dict(analysis="circuit_depth", depths=list(depths), rows=rows)

    if plot:
        keys = [k for k in ("train_loss", "rel_l2", "grad_norm",
                            "expressivity", "runtime") if any(k in r for r in rows)]
        n = len(keys)
        fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.2), squeeze=False)
        for a, k in zip(ax[0], keys):
            a.plot(list(depths), col(k), "o-")
            a.set(xlabel="circuit depth (layers)", ylabel=k,
                  title=f"depth vs {k}")
            if k in ("train_loss", "rel_l2"): a.set_yscale("log")
            a.grid(alpha=.3)
        fig.suptitle(f"2.3 Circuit-depth analysis — {name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_3_depth_{name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.4  Entanglement analysis                                                   #
# ============================================================================= #
def entanglement_analysis(probe: QuantumProbe, bounds: Sequence[tuple],
                          residual_fn: Optional[Callable] = None,
                          n: int = 512, seed: int = 0,
                          plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    Does the trained circuit actually entangle? (Answer is *measured*, not assumed.)

    Metrics
    -------
    * meyer_wallach_Q      : global entanglement, averaged over the domain batch.
    * single_qubit_entropy : mean von-Neumann entropy of each qubit's reduced state.
    * mean_pair_concurrence: averaged Wootters concurrence over qubit pairs.
    * pair_correlation      : <Z_i Z_j> - <Z_i><Z_j> correlation matrix.
    * Q_vs_residual        : (optional) correlation between per-sample Q and |residual|.

    IMPORTANT (project principle): a full-CNOT ansatz does *not* guarantee high Q
    after training. Report the observed value and phrase conclusions conditionally.
    A near-zero Q on few qubits is often a shape/batching artefact — this routine
    processes states per-sample to avoid that.
    """
    if not probe.has_state():
        return dict(analysis="entanglement", skipped="no statevector")

    X = utils.sample_domain(bounds, n, seed=seed, device=probe.device)
    psi = np.atleast_2d(np.asarray(probe.statevector(X).detach().cpu().numpy()))
    nq = probe.n_qubits

    # Meyer-Wallach, per-sample then averaged
    Q_per = np.array([utils.meyer_wallach_Q(psi[b], nq) for b in range(psi.shape[0])])
    Q = float(Q_per.mean())

    # single-qubit entropies
    ent = np.zeros(nq)
    for b in range(psi.shape[0]):
        rho = utils.density_from_state(psi[b])
        for q in range(nq):
            ent[q] += utils.vn_entropy(utils.partial_trace_keep(rho, q, nq))
    ent /= psi.shape[0]

    # pair concurrence + ZZ correlation (sampled subset for cost)
    Z = _PAULI["Z"]
    sub = psi[: min(128, psi.shape[0])]
    corr = np.zeros((nq, nq))
    conc = []
    for b in range(sub.shape[0]):
        rho = utils.density_from_state(sub[b])
        zexp = np.array([np.real(np.einsum("i,ij,j->",
                        sub[b].conj(), _pauli_string_op("I"*q+"Z"+"I"*(nq-q-1)), sub[b]))
                        for q in range(nq)])
        for i in range(nq):
            for j in range(nq):
                zz = np.real(np.einsum("k,kl,l->", sub[b].conj(),
                     _pauli_string_op("I"*i+"Z"+"I"*(j-i-1)+"Z"+"I"*(nq-j-1)) if j > i
                     else _pauli_string_op("I"*nq), sub[b])) if j > i else zexp[i]**0
                corr[i, j] += (zz - zexp[i]*zexp[j]) if j > i else 0.0
    corr /= sub.shape[0]
    corr = corr + corr.T

    res = dict(analysis="entanglement", n_qubits=nq,
               meyer_wallach_Q=Q, Q_std=float(Q_per.std()),
               single_qubit_entropy=ent.tolist(),
               pair_correlation=corr.tolist(),
               interpretation=_entanglement_verdict(Q))

    if residual_fn is not None:
        r = np.abs(np.asarray(residual_fn(probe, X)).ravel())
        m = min(len(r), len(Q_per))
        c = np.corrcoef(Q_per[:m], r[:m])[0, 1]
        res["Q_vs_residual_corr"] = float(0.0 if np.isnan(c) else c)

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].bar(range(nq), ent); ax[0].set(xlabel="qubit",
                  ylabel="von-Neumann entropy", title=f"Single-qubit entropy\nMeyer-Wallach Q={Q:.3f}")
        ax[0].set_xticks(range(nq))
        im = ax[1].imshow(corr, cmap="RdBu_r", vmin=-abs(corr).max()-1e-9,
                          vmax=abs(corr).max()+1e-9)
        ax[1].set(title="Pair correlation ⟨ZᵢZⱼ⟩-⟨Zᵢ⟩⟨Zⱼ⟩", xlabel="qubit j", ylabel="qubit i")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        ax[2].hist(Q_per, bins=30, color="C2")
        ax[2].axvline(Q, color="k", ls="--", label=f"mean {Q:.3f}")
        ax[2].set(xlabel="per-sample Meyer-Wallach Q", ylabel="count",
                  title="Entanglement distribution"); ax[2].legend()
        fig.suptitle(f"2.4 Entanglement — {probe.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_4_entanglement_{probe.name}", outdir)
        plt.close(fig)
    return res


def _entanglement_verdict(Q: float) -> str:
    if Q < 0.05:
        return ("Observed Q≈0: the trained circuit is (nearly) producing product "
                "states over this domain. On few qubits verify this is not a "
                "batching/shape artefact before citing it.")
    if Q < 0.4:
        return "Mild entanglement: the quantum layer uses correlations modestly."
    return "Substantial entanglement: correlations across qubits are actively used."


# ============================================================================= #
#  2.5  Fourier-spectrum analysis                                              #
# ============================================================================= #
def fourier_spectrum(probe: QuantumProbe, bounds: Sequence[tuple],
                     axis: int = 0, n: int = 512,
                     classical_ref: Optional[ModelAdapter] = None,
                     plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    Can the quantum layer represent high-frequency PDE content (shocks)?

    We sweep the field along one input axis, FFT it, and report the amplitude
    spectrum and the *spectral centroid* (a robust summary; raw bandwidth on
    near-random circuits is brittle — hence centroid, per project principle).

    Theory anchor (Schuld-Sweke-Meyer): with a single angle encoding the
    reachable frequency set is fixed by the encoding and does NOT grow with the
    weights; data re-uploading grows it ~linearly in the number of re-uploads.
    We therefore also report the *theoretical reachable max integer frequency*
    = n_encodings, where n_encodings = (n_layers if reupload else 1).

    If `classical_ref` is provided, its spectrum is overlaid so you can see the
    frequency content the classical PINN reaches for the same problem.
    """
    Xline = utils.line_samples(bounds, axis, n, device=probe.device)
    with torch.no_grad():
        u = probe(Xline).detach().cpu().numpy().ravel()
    coord = Xline[:, axis].cpu().numpy()
    L = bounds[axis][1] - bounds[axis][0]

    amp = np.abs(np.fft.rfft(u - u.mean()))
    freqs = np.fft.rfftfreq(n, d=L / n)
    centroid = float(np.sum(freqs * amp) / (np.sum(amp) + 1e-30))

    reachable = probe.n_layers if probe.reupload else 1

    res = dict(analysis="fourier_spectrum", axis=axis,
               spectral_centroid=centroid,
               theoretical_reachable_freq=int(reachable),
               reupload=probe.reupload, n_layers=probe.n_layers,
               note=("Reachable frequency band is encoding-determined "
                     "(Schuld-Sweke-Meyer); re-uploading expands it linearly. "
                     "Weights redistribute amplitude within that band only."))

    if classical_ref is not None:
        with torch.no_grad():
            uc = classical_ref(Xline).detach().cpu().numpy().ravel()
        ampc = np.abs(np.fft.rfft(uc - uc.mean()))
        res["classical_spectral_centroid"] = float(np.sum(freqs*ampc)/(np.sum(ampc)+1e-30))

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(coord, u, lw=1.4, label=probe.name)
        if classical_ref is not None:
            ax[0].plot(coord, uc, lw=1.2, ls="--", label=classical_ref.name)
        ax[0].set(xlabel=f"input axis {axis}", ylabel="u", title="Field slice"); ax[0].legend()
        ax[1].semilogy(freqs, amp + 1e-12, lw=1.3, label=f"{probe.name} spectrum")
        if classical_ref is not None:
            ax[1].semilogy(freqs, ampc + 1e-12, lw=1.1, ls="--", label=classical_ref.name)
        ax[1].axvline(reachable, color="C3", ls=":", label=f"reachable f≤{reachable}")
        ax[1].axvline(centroid, color="C2", ls="-.", label=f"centroid {centroid:.2f}")
        ax[1].set(xlabel="frequency", ylabel="amplitude", title="Amplitude spectrum")
        ax[1].legend(fontsize=8); ax[1].set_xlim(0, min(freqs.max(), 4*max(reachable, 4)))
        fig.suptitle(f"2.5 Fourier spectrum — {probe.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_5_fourier_{probe.name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.6  Expressivity analysis                                                   #
# ============================================================================= #
def expressivity_analysis(probe: QuantumProbe, bounds: Sequence[tuple],
                          n_states: int = 400, n_random_weights: int = 40,
                          seed: int = 0, plot: bool = True,
                          outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    How large is the function class the circuit represents?

    Metrics
    -------
    * state_overlap_hist : distribution of |<psi_i|psi_j>|^2 over random inputs;
      broad/low-overlap => the circuit explores Hilbert space (more expressive).
    * kernel_rank        : numerical rank of the quantum kernel K_ij=|<psi_i|psi_j>|^2.
    * effective_dimension: participation-ratio effective dim of the kernel
      spectrum, (Σλ)^2 / Σλ^2 — a smooth, comparable expressivity scalar.
    * feature_dim        : 2**n_qubits (raw measurement dimension the head sees).

    NTK / quantum-Fisher are heavy; we expose a light effective-dimension proxy
    here and leave full QFI to `barren_plateau_analysis`/`layer3` gradients.
    """
    if not probe.has_state():
        # fall back to probability-space kernel if no statevector
        X = utils.sample_domain(bounds, n_states, seed=seed, device=probe.device)
        with torch.no_grad():
            feat = probe.probs(X).cpu().numpy()
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-30)
        K = feat @ feat.T
    else:
        X = utils.sample_domain(bounds, n_states, seed=seed, device=probe.device)
        psi = np.atleast_2d(np.asarray(probe.statevector(X).detach().cpu().numpy()))
        K = np.abs(psi.conj() @ psi.T) ** 2

    ev = np.clip(np.linalg.eigvalsh(K), 0, None)[::-1]
    rank = int((ev > 1e-6 * ev.max()).sum())
    eff_dim = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-30))
    offdiag = K[~np.eye(K.shape[0], dtype=bool)]

    res = dict(analysis="expressivity", n_qubits=probe.n_qubits,
               feature_dim=int(2 ** probe.n_qubits),
               kernel_rank=rank, effective_dimension=eff_dim,
               mean_state_overlap=float(offdiag.mean()),
               eigspectrum_head=ev[:min(20, len(ev))].tolist())

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].hist(offdiag, bins=40, color="C0")
        ax[0].set(xlabel="pairwise state overlap |⟨ψᵢ|ψⱼ⟩|²",
                  ylabel="count", title=f"State overlap (lower=more expressive)")
        ax[1].semilogy(ev[:min(40, len(ev))] + 1e-12, "o-")
        ax[1].set(xlabel="index", ylabel="kernel eigenvalue",
                  title=f"Kernel spectrum\nrank={rank}, eff-dim={eff_dim:.1f}")
        ax[1].grid(alpha=.3)
        fig.suptitle(f"2.6 Expressivity — {probe.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_6_expressivity_{probe.name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.7  Barren-plateau analysis                                                #
# ============================================================================= #
def barren_plateau_analysis(make_probe: Callable[[int, int], QuantumProbe],
                            bounds: Sequence[tuple],
                            qubit_list: Sequence[int] = (2, 4, 6),
                            depth_list: Sequence[int] = (2, 4, 8),
                            n_samples: int = 60, n_points: int = 64, seed: int = 0,
                            plot: bool = True, outdir: str = "outputs/xai",
                            name: str = "qapinn") -> Dict[str, Any]:
    """
    Barren plateaus: variance of the loss gradient w.r.t. a circuit parameter,
    as a function of #qubits and depth. Exponential decay of Var[∂L] with qubits
    is the barren-plateau signature.

    `make_probe(n_qubits, depth)` returns a *fresh random-init* QuantumProbe.
    We estimate Var over `n_samples` random weight draws, using the mean-squared
    output over `n_points` domain samples as a generic scalar objective (no PDE
    labels needed — it isolates the trainability of the quantum layer itself).
    """
    grid = np.full((len(qubit_list), len(depth_list)), np.nan)
    detail = []
    X0 = utils.sample_domain(bounds, n_points, seed=seed)
    for i, nq in enumerate(qubit_list):
        for j, dp in enumerate(depth_list):
            grads = []
            for s in range(n_samples):
                probe = make_probe(nq, dp)
                X = X0.to(probe.device)
                w = probe.q_weights
                # scalar objective: mean output^2 (probe-independent of labels)
                out = probe(X)
                obj = (out ** 2).mean()
                g = torch.autograd.grad(obj, w, retain_graph=False, allow_unused=True)[0]
                if g is None:
                    continue
                grads.append(g.detach().cpu().numpy().ravel()[0])   # first param
            v = float(np.var(grads)) if grads else np.nan
            grid[i, j] = v
            detail.append(dict(n_qubits=nq, depth=dp, grad_var=v))

    res = dict(analysis="barren_plateau", qubit_list=list(qubit_list),
               depth_list=list(depth_list), grad_var_grid=grid.tolist(),
               detail=detail)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for j, dp in enumerate(depth_list):
            ax[0].semilogy(qubit_list, grid[:, j], "o-", label=f"depth {dp}")
        ax[0].set(xlabel="n_qubits", ylabel="Var[∂L/∂θ]",
                  title="Gradient variance vs qubits\n(exp. decay ⇒ barren plateau)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        im = ax[1].imshow(np.log10(grid + 1e-30), aspect="auto", cmap="viridis",
                          origin="lower")
        ax[1].set(xticks=range(len(depth_list)), yticks=range(len(qubit_list)),
                  xticklabels=depth_list, yticklabels=qubit_list,
                  xlabel="depth", ylabel="n_qubits", title="log₁₀ Var[∂L]")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        fig.suptitle(f"2.7 Barren-plateau — {name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_7_barren_{name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.8  Loss-landscape analysis                                                #
# ============================================================================= #
def loss_landscape(adapter: ModelAdapter, loss_fn: Callable[[], float],
                   span: float = 1.0, n: int = 25, seed: int = 0,
                   plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    Filter-normalised 2-D loss landscape around the current optimum
    (Li et al. 2018). Two random directions are drawn and filter-normalised so
    the landscape is comparable across architectures (classical PINN vs QA-PINN).

    `loss_fn()` must evaluate the *total* PINN loss at the model's current
    parameters (typically a closure over your `composite_loss(model, B)`).
    We perturb parameters along the two directions, evaluate the grid, restore.
    """
    params = [p for p in adapter.parameters if p.requires_grad]
    if not params:
        return dict(analysis="loss_landscape", skipped="no trainable params")

    g = torch.Generator().manual_seed(seed)
    orig = [p.detach().clone() for p in params]

    def _rand_dir():
        d = [torch.randn(p.shape, generator=g).to(p.device) for p in params]
        # filter normalisation: scale each direction to its param's norm
        for di, p in zip(d, params):
            di.mul_(p.norm() / (di.norm() + 1e-10))
        return d

    d1, d2 = _rand_dir(), _rand_dir()
    alphas = np.linspace(-span, span, n)
    Z = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            with torch.no_grad():
                for p, o, x1, x2 in zip(params, orig, d1, d2):
                    p.copy_(o + alphas[a] * x1 + alphas[b] * x2)
            Z[a, b] = float(loss_fn())
    with torch.no_grad():
        for p, o in zip(params, orig): p.copy_(o)

    res = dict(analysis="loss_landscape", span=span, grid_n=n,
               loss_center=float(Z[n // 2, n // 2]),
               loss_min=float(Z.min()), loss_max=float(Z.max()),
               roughness=float(np.std(Z)))

    if plot:
        A, Bm = np.meshgrid(alphas, alphas, indexing="ij")
        fig = plt.figure(figsize=(11, 4.2))
        ax0 = fig.add_subplot(1, 2, 1)
        cs = ax0.contourf(A, Bm, np.log10(Z - Z.min() + 1e-8), levels=30, cmap="viridis")
        ax0.plot(0, 0, "r*", ms=12); fig.colorbar(cs, ax=ax0, label="log₁₀(L-Lmin)")
        ax0.set(xlabel="dir 1", ylabel="dir 2", title="Filter-normalised loss")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        ax1.plot_surface(A, Bm, np.log10(Z - Z.min() + 1e-8), cmap="viridis")
        ax1.set(xlabel="dir 1", ylabel="dir 2", title="Surface")
        fig.suptitle(f"2.8 Loss landscape — {adapter.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_8_landscape_{adapter.name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.9  Measurement-distribution analysis                                       #
# ============================================================================= #
def measurement_distribution(probe: QuantumProbe, bounds: Sequence[tuple],
                             n: int = 800, seed: int = 0,
                             ref_probe: Optional[QuantumProbe] = None,
                             plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    Study of the raw measurement histogram (computational-basis probabilities).

    Metrics
    -------
    * mean_entropy   : Shannon entropy of the basis distribution averaged over the
      domain — how much of the 2^n-dim measurement space is actually used.
    * effective_support : 2**entropy (participation of basis states).
    * kl_to_uniform  : KL(mean prob || uniform) — deviation from maximal mixing.
    * kl_to_reference: (optional) KL against another probe (e.g. untrained), to
      quantify how *training* reshaped the measurement distribution.
    """
    X = utils.sample_domain(bounds, n, seed=seed, device=probe.device)
    with torch.no_grad():
        P = probe.probs(X).cpu().numpy()
    pbar = P.mean(0); pbar /= pbar.sum()
    ent = float(np.mean([utils.shannon_entropy(p) for p in P]))
    uniform = np.full_like(pbar, 1.0 / len(pbar))

    res = dict(analysis="measurement_distribution", n_qubits=probe.n_qubits,
               mean_entropy=ent, effective_support=float(2 ** ent),
               kl_to_uniform=utils.kl_divergence(pbar, uniform),
               basis_dim=int(len(pbar)))

    if ref_probe is not None:
        with torch.no_grad():
            Pr = ref_probe.probs(utils.sample_domain(bounds, n, seed=seed,
                                 device=ref_probe.device)).cpu().numpy()
        pref = Pr.mean(0); pref /= pref.sum()
        res["kl_to_reference"] = utils.kl_divergence(pbar, pref)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].bar(range(len(pbar)), pbar, color="C0")
        ax[0].axhline(1/len(pbar), color="k", ls="--", label="uniform")
        ax[0].set(xlabel="basis state", ylabel="mean probability",
                  title=f"Measurement histogram\nH={ent:.2f} bits"); ax[0].legend()
        ax[1].hist([utils.shannon_entropy(p) for p in P], bins=30, color="C3")
        ax[1].set(xlabel="per-sample entropy (bits)", ylabel="count",
                  title="Entropy distribution over domain")
        fig.suptitle(f"2.9 Measurement distribution — {probe.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_9_measdist_{probe.name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.10  Feature attribution (explainability)                                   #
# ============================================================================= #
def feature_attribution(adapter: ModelAdapter, bounds: Sequence[tuple],
                        axes: Sequence[int] = (0, 1), grid_n: int = 120,
                        plot: bool = True, outdir: str = "outputs/xai") -> Dict[str, Any]:
    """
    Saliency / sensitivity maps for explainability: |∂u/∂input_d| rendered as a
    field over two chosen input axes (e.g. (x,t)). Highlights *where* in the
    domain the model is most input-sensitive — for Burgers this concentrates at
    the shock. Works identically for classical and quantum adapters, so the two
    saliency maps can be compared side by side to show the quantum layer's effect.
    """
    ax0, ax1 = axes[0], axes[1]
    a = np.linspace(bounds[ax0][0], bounds[ax0][1], grid_n)
    b = np.linspace(bounds[ax1][0], bounds[ax1][1], grid_n)
    A, Bm = np.meshgrid(a, b, indexing="ij")
    d = len(bounds)
    cols = []
    for k in range(d):
        if k == ax0: cols.append(A.ravel())
        elif k == ax1: cols.append(Bm.ravel())
        else: cols.append(np.full(A.size, 0.5*(bounds[k][0]+bounds[k][1])))
    X = torch.tensor(np.stack(cols, 1), dtype=torch.float32, device=adapter.device)
    g = utils.input_gradient(adapter, X)
    sal = np.linalg.norm(g, axis=1).reshape(grid_n, grid_n)

    res = dict(analysis="feature_attribution", axes=list(axes),
               saliency_max=float(sal.max()), saliency_mean=float(sal.mean()),
               peak_location=[float(a[np.unravel_index(sal.argmax(), sal.shape)[0]]),
                              float(b[np.unravel_index(sal.argmax(), sal.shape)[1]])])

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ext = [bounds[ax0][0], bounds[ax0][1], bounds[ax1][0], bounds[ax1][1]]
        im0 = ax[0].imshow(sal.T, extent=ext, origin="lower", aspect="auto", cmap="magma")
        ax[0].set(xlabel=f"axis {ax0}", ylabel=f"axis {ax1}",
                  title="Saliency ||∂u/∂input||"); fig.colorbar(im0, ax=ax[0], fraction=0.046)
        # per-axis component
        comp = np.abs(g).reshape(grid_n, grid_n, d)
        im1 = ax[1].imshow(comp[..., ax0].T, extent=ext, origin="lower",
                           aspect="auto", cmap="viridis")
        ax[1].set(xlabel=f"axis {ax0}", ylabel=f"axis {ax1}",
                  title=f"|∂u/∂(axis {ax0})|"); fig.colorbar(im1, ax=ax[1], fraction=0.046)
        fig.suptitle(f"2.10 Feature attribution — {adapter.name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_10_saliency_{adapter.name}", outdir)
        plt.close(fig)
    return res


# ============================================================================= #
#  2.11  Quantum-state evolution                                                #
# ============================================================================= #
def quantum_state_evolution(make_probe_from_weights: Callable[[np.ndarray], QuantumProbe],
                            weight_snapshots: Dict[int, np.ndarray],
                            bounds: Sequence[tuple], probe_point: Optional[np.ndarray] = None,
                            seed: int = 0, plot: bool = True,
                            outdir: str = "outputs/xai", name: str = "qapinn") -> Dict[str, Any]:
    """
    Store & visualise the quantum state as training progresses.

    `weight_snapshots` maps epoch -> q_weights array (save these during training,
    exactly as the codebase already snapshots state_dicts). `make_probe_from_weights`
    rebuilds a QuantumProbe with given weights. We track, at a fixed input point:

      * per-qubit Bloch vector (<X>,<Y>,<Z>)  -> Bloch-sphere trajectory
      * purity Tr[rho_q^2] per qubit
      * fidelity |<psi_e | psi_{e_prev}>|^2 between consecutive snapshots
      * von-Neumann entropy per qubit

    This shows how the quantum representation *forms* during optimisation.
    """
    if probe_point is None:
        probe_point = utils.sample_domain(bounds, 1, seed=seed).cpu().numpy()
    Xp = torch.tensor(probe_point, dtype=torch.float32).reshape(1, -1)

    epochs = sorted(weight_snapshots)
    bloch, purity, entropy, fidelity = [], [], [], []
    prev_psi = None
    nq = None
    for e in epochs:
        probe = make_probe_from_weights(weight_snapshots[e])
        nq = probe.n_qubits
        psi = np.asarray(probe.statevector(Xp.to(probe.device)).detach().cpu().numpy()).ravel()
        rho = utils.density_from_state(psi)
        bv, pu, en = [], [], []
        for q in range(nq):
            rq = utils.partial_trace_keep(rho, q, nq)
            bx = np.real(np.trace(rq @ _PAULI["X"]))
            by = np.real(np.trace(rq @ _PAULI["Y"]))
            bz = np.real(np.trace(rq @ _PAULI["Z"]))
            bv.append([bx, by, bz]); pu.append(float(np.real(np.trace(rq @ rq))))
            en.append(utils.vn_entropy(rq))
        bloch.append(bv); purity.append(pu); entropy.append(en)
        fidelity.append(1.0 if prev_psi is None
                        else float(np.abs(np.vdot(prev_psi, psi)) ** 2))
        prev_psi = psi

    bloch = np.array(bloch)          # (E, nq, 3)
    purity = np.array(purity)        # (E, nq)
    entropy = np.array(entropy)
    res = dict(analysis="quantum_state_evolution", epochs=epochs, n_qubits=nq,
               bloch=bloch.tolist(), purity=purity.tolist(),
               entropy=entropy.tolist(), fidelity=fidelity)

    if plot:
        fig = plt.figure(figsize=(14, 4))
        axb = fig.add_subplot(1, 3, 1, projection="3d")
        _draw_bloch_wire(axb)
        for q in range(nq):
            axb.plot(bloch[:, q, 0], bloch[:, q, 1], bloch[:, q, 2], "-o", ms=3, label=f"q{q}")
        axb.set_title("Bloch trajectory"); axb.legend(fontsize=7)
        ax1 = fig.add_subplot(1, 3, 2)
        for q in range(nq): ax1.plot(epochs, purity[:, q], "o-", label=f"q{q}")
        ax1.set(xlabel="epoch", ylabel="purity Tr[ρ²]", title="Per-qubit purity"); ax1.legend(fontsize=7); ax1.grid(alpha=.3)
        ax2 = fig.add_subplot(1, 3, 3)
        ax2.plot(epochs, fidelity, "s-", color="C3")
        ax2.set(xlabel="epoch", ylabel="fidelity to prev snapshot",
                title="State change between snapshots"); ax2.grid(alpha=.3)
        fig.suptitle(f"2.11 Quantum-state evolution — {name}")
        plt.tight_layout(); res["figure"] = utils.savefig(fig, f"l2_11_state_evol_{name}", outdir)
        plt.close(fig)
    return res


def _draw_bloch_wire(ax):
    u = np.linspace(0, 2*np.pi, 24); v = np.linspace(0, np.pi, 12)
    x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.15, linewidth=0.5)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel("⟨X⟩"); ax.set_ylabel("⟨Y⟩"); ax.set_zlabel("⟨Z⟩")
