import numpy as np, matplotlib.pyplot as plt, os, json

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

def load_run(run_dir):
    d = np.load(os.path.join(run_dir, "history.npz"))
    tr = {k[6:]: d[k] for k in d.files if k.startswith("train_")}
    va = {k[4:]: d[k] for k in d.files if k.startswith("val_")}
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    return tr, va, meta


def plot_training_curves(run_dir, save=True):
    tr, va, meta = load_run(run_dir)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    a = ax[0, 0]
    a.semilogy(tr["it"], tr["loss"], lw=1.2, label="total")
    a.semilogy(tr["it"], tr["L_r"],  lw=1.0, label=r"$\mathcal{L}_r$")
    a.semilogy(tr["it"], tr["L_ic"], lw=1.0, label=r"$\mathcal{L}_{ic}$")
    a.semilogy(tr["it"], tr["L_bc"], lw=1.0, label=r"$\mathcal{L}_{bc}$")
    a.set_xlabel("iteration"); a.set_ylabel("loss"); a.legend(fontsize=8)
    a.set_title("Loss components")

    a = ax[0, 1]
    a.semilogy(tr["it"], tr["L_ru"], lw=1.0, label=r"$\|r_u\|^2$")
    a.semilogy(tr["it"], tr["L_rv"], lw=1.0, label=r"$\|r_v\|^2$")
    a.set_xlabel("iteration"); a.set_ylabel("residual MSE"); a.legend(fontsize=8)
    a.set_title("Per-component residual (coupling balance)")

    a = ax[1, 0]
    a.semilogy(va["it"], va["rel_u"], "o-", ms=3, lw=1.1, label="u")
    a.semilogy(va["it"], va["rel_v"], "s-", ms=3, lw=1.1, label="v")
    a.set_xlabel("iteration"); a.set_ylabel(r"rel. $L^2$"); a.legend(fontsize=8)
    a.set_title("Validation error (uniform grid)")

    a = ax[1, 1]
    a.semilogy(tr["it"], tr["gnorm"], lw=0.9, color="C3")
    a.set_xlabel("iteration"); a.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    a2 = a.twinx(); a2.plot(tr["it"], tr["lr"], lw=0.9, color="C7", ls="--")
    a2.set_ylabel("lr", color="C7"); a2.grid(False)
    a.set_title("Gradient norm & learning rate")

    fig.suptitle(f"{meta['tag']}  |  params={meta['n_params']}  "
                 f"|  best mean rel$L^2$={meta['best_mean_relL2']:.2e}", y=1.01)
    plt.tight_layout()
    if save: plt.savefig(os.path.join(run_dir, "training_curves.png"))
    plt.show()


def plot_loss_vs_walltime(run_dirs, labels=None, save=None):
    """Fair QA-PINN vs classical comparison needs BOTH axes."""
    labels = labels or [os.path.basename(d) for d in run_dirs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, lab in zip(run_dirs, labels):
        tr, va, _ = load_run(d)
        ax[0].semilogy(tr["it"], tr["loss"], lw=1.1, label=lab)
        ax[1].semilogy(tr["wall"]/60., tr["loss"], lw=1.1, label=lab)
    ax[0].set_xlabel("iteration"); ax[1].set_xlabel("wall-clock (min)")
    for a in ax: a.set_ylabel("total loss"); a.legend(fontsize=8)
    ax[0].set_title("Matched epochs"); ax[1].set_title("Matched compute")
    plt.tight_layout()
    if save: plt.savefig(save)
    plt.show()


def plot_convergence_band(run_dirs, key="rel_u", label="", save=None, ax=None):
    """Multi-seed: median + min/max band."""
    curves, its = [], None
    for d in run_dirs:
        _, va, _ = load_run(d)
        its = va["it"]; curves.append(va[key])
    C = np.stack(curves)
    own = ax is None
    if own: fig, ax = plt.subplots(figsize=(6, 4))
    med = np.median(C, 0)
    ax.semilogy(its, med, lw=1.4, label=f"{label} (median, n={len(run_dirs)})")
    ax.fill_between(its, C.min(0), C.max(0), alpha=0.22)
    ax.set_xlabel("iteration"); ax.set_ylabel(f"rel. $L^2$ ({key[-1]})")
    ax.legend(fontsize=8)
    if own:
        plt.tight_layout()
        if save: plt.savefig(save)
        plt.show()
    return ax
