import numpy as np, torch, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from config import DEVICE

def save(fig, name):
    p = f"outputs/{name}.png"; fig.savefig(p, dpi=160, bbox_inches="tight")
    print("saved", p); return p

# ---------- 1. Heatmaps ----------
def heatmap_triptych(res, title, name):
    x, t, Up, Ug = res["x"], res["t"], res["U_pred"], res["U_true"]
    err = np.abs(Up-Ug)
    fig, ax = plt.subplots(1,3, figsize=(16,4.2))
    ext = [x.min(), x.max(), t.min(), t.max()]
    for a, D, ttl, cm in zip(ax, [Ug, Up, err],
        ["Ground truth (spectral)", "Prediction", "|error|"],
        ["RdBu_r","RdBu_r","magma"]):
        vm = np.abs(Ug).max()
        im = a.imshow(D, extent=ext, origin="lower", aspect="auto", cmap=cm,
                      vmin=(-vm if cm=="RdBu_r" else None),
                      vmax=(vm if cm=="RdBu_r" else None))
        a.set_xlabel("x"); a.set_ylabel("t"); a.set_title(ttl)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(title, y=1.03)
    plt.tight_layout(); save(fig, name); plt.show()

def snapshots(res, ts=(0.0,0.25,0.5,0.75,1.0), title="", name="snap"):
    fig, ax = plt.subplots(1, len(ts), figsize=(4*len(ts), 3.4), sharey=True)
    for a, t0 in zip(np.atleast_1d(ax), ts):
        i = int(np.argmin(np.abs(res["t"]-t0)))
        a.plot(res["x"], res["U_true"][i], "k-", lw=2.2, label="ground truth")
        a.plot(res["x"], res["U_pred"][i], "r--", lw=1.8, label="model")
        a.set_title(f"t={res['t'][i]:.2f}  L2={res['l2_per_t'][i]:.2e}")
        a.set_xlabel("x"); a.grid(alpha=.3)
    np.atleast_1d(ax)[0].set_ylabel("u"); np.atleast_1d(ax)[0].legend()
    fig.suptitle(title, y=1.05); plt.tight_layout(); save(fig, name); plt.show()

# ---------- 2. Loss curves: vs epoch AND vs wall-clock ----------
def loss_curves(hists: dict, name="loss_curves"):
    fig, ax = plt.subplots(2,2, figsize=(14,8))
    for lbl,h in hists.items():
        ax[0,0].semilogy(h["iter"], h["total"], lw=1.2, label=lbl)
        ax[0,1].semilogy(h["wall"], h["total"], lw=1.2, label=lbl)
        ax[1,0].semilogy(h["iter"], h["pde"], lw=1.0, label=f"{lbl} pde")
        ax[1,0].semilogy(h["iter"], h["ic"],  lw=1.0, ls="--", label=f"{lbl} ic")
        ax[1,1].plot(h["wall"], h["iter"], lw=1.4, label=lbl)
    ax[0,0].set(xlabel="iteration", ylabel="total loss", title="Loss vs epoch")
    ax[0,1].set(xlabel="wall-clock (s)", ylabel="total loss", title="Loss vs wall-clock time")
    ax[1,0].set(xlabel="iteration", ylabel="component loss", title="Loss components")
    ax[1,1].set(xlabel="wall-clock (s)", ylabel="iteration", title="Throughput (iters vs time)")
    for a in ax.ravel(): a.grid(alpha=.3); a.legend(fontsize=7)
    plt.tight_layout(); save(fig, name); plt.show()

# ---------- 3. Hidden-layer visualisation ----------
@torch.no_grad()
def hidden_layer_maps(model, layer_keys=None, ts=(0.0,0.25,0.5,0.75,1.0),
                      nx=200, n_units=6, title="", name="hidden_maps"):
    """Per-layer, per-time-step: value of first n_units neurons across x."""
    xs = np.linspace(-1,1,nx)
    acts_by_t = {}
    for t0 in ts:
        X = torch.tensor(np.stack([xs, np.full(nx, t0)],1), dtype=torch.float32, device=DEVICE)
        acts_by_t[t0] = model.layer_activations(X)
    keys = layer_keys or [k for k in acts_by_t[ts[0]] if k not in ("input",)]
    fig, ax = plt.subplots(len(keys), len(ts), figsize=(3.1*len(ts), 2.4*len(keys)),
                           squeeze=False, sharex=True)
    for r,k in enumerate(keys):
        for c,t0 in enumerate(ts):
            A = acts_by_t[t0][k]
            for u in range(min(n_units, A.shape[1])):
                ax[r,c].plot(xs, A[:,u], lw=1.1)
            ax[r,c].grid(alpha=.25)
            if r==0: ax[r,c].set_title(f"t={t0}")
            if c==0: ax[r,c].set_ylabel(k, fontsize=9)
    for c in range(len(ts)): ax[-1,c].set_xlabel("x")
    fig.suptitle(f"Hidden-layer activations — {title}", y=1.01)
    plt.tight_layout(); save(fig, name); plt.show()
    return acts_by_t

@torch.no_grad()
def hidden_layer_heatmaps(model, nx=200, nt=100, layer_keys=None,
                          n_units=4, title="", name="hidden_heat"):
    """Neuron activation as a full (x,t) heatmap — shows where each unit fires."""
    xs = np.linspace(-1,1,nx); ts = np.linspace(0,1,nt)
    X,T = np.meshgrid(xs, ts, indexing="xy")
    P = torch.tensor(np.stack([X.ravel(),T.ravel()],1), dtype=torch.float32, device=DEVICE)
    acts = model.layer_activations(P)
    keys = layer_keys or [k for k in acts if k != "input"]
    fig, ax = plt.subplots(len(keys), n_units, figsize=(3.0*n_units, 2.5*len(keys)),
                           squeeze=False)
    for r,k in enumerate(keys):
        A = acts[k]
        for u in range(n_units):
            if u >= A.shape[1]: ax[r,u].axis("off"); continue
            M = A[:,u].reshape(nt,nx)
            im = ax[r,u].imshow(M, extent=[-1,1,0,1], origin="lower",
                                aspect="auto", cmap="coolwarm")
            ax[r,u].set_title(f"{k}[{u}]", fontsize=8)
            fig.colorbar(im, ax=ax[r,u], fraction=0.046)
    fig.suptitle(f"Per-neuron (x,t) activation maps — {title}", y=1.01)
    plt.tight_layout(); save(fig, name); plt.show()

@torch.no_grad()
def hidden_pca_trajectory(model, ts=np.linspace(0,1,6), nx=200,
                          layer="h3", title="", name="hidden_pca"):
    """PCA of a hidden layer's representation, coloured by x, one panel per t."""
    xs = np.linspace(-1,1,nx)
    fig, ax = plt.subplots(1, len(ts), figsize=(3.0*len(ts), 3.0))
    for a,t0 in zip(np.atleast_1d(ax), ts):
        X = torch.tensor(np.stack([xs, np.full(nx,t0)],1), dtype=torch.float32, device=DEVICE)
        A = model.layer_activations(X)[layer]
        Z = PCA(2).fit_transform(A) if A.shape[1] > 2 else A[:,:2]
        s = a.scatter(Z[:,0], Z[:,1], c=xs, cmap="viridis", s=8)
        a.set_title(f"t={t0:.2f}", fontsize=9); a.grid(alpha=.25)
    fig.colorbar(s, ax=np.atleast_1d(ax).tolist(), label="x", fraction=0.02)
    fig.suptitle(f"PCA of {layer} — {title}", y=1.06)
    save(fig, name); plt.show()

@torch.no_grad()
def layer_evolution_during_training(model_cls, snaps, ctor_kw, layer="h3",
                                    t0=0.5, nx=200, name="layer_evol"):
    """Same hidden layer, at several TRAINING epochs — shows feature formation."""
    xs = np.linspace(-1,1,nx)
    X = torch.tensor(np.stack([xs, np.full(nx,t0)],1), dtype=torch.float32, device=DEVICE)
    eps = sorted(snaps)
    fig, ax = plt.subplots(1, len(eps), figsize=(3.0*len(eps), 2.8), sharey=True)
    for a,e in zip(np.atleast_1d(ax), eps):
        m = model_cls(**ctor_kw).to(DEVICE); m.load_state_dict(snaps[e]); m.eval()
        A = m.layer_activations(X)[layer]
        for u in range(min(8, A.shape[1])): a.plot(xs, A[:,u], lw=1.0)
        a.set_title(f"epoch {e}", fontsize=9); a.set_xlabel("x"); a.grid(alpha=.25)
    np.atleast_1d(ax)[0].set_ylabel(layer)
    fig.suptitle(f"Evolution of {layer} during training (t={t0})", y=1.06)
    save(fig, name); plt.show()

# ---------- 4. Solver comparison ----------
def solver_comparison(gts, ts=(0.25,0.5,0.75,1.0), nx=401, name="solver_cmp"):
    xq = np.linspace(-1,1,nx)
    fig, ax = plt.subplots(1, len(ts), figsize=(4*len(ts),3.4), sharey=True)
    styles = {"spectral":("k-",2.2),"fdm":("C0--",1.6),"fem":("C3:",1.8)}
    for a,t0 in zip(ax, ts):
        for m,g in gts.items():
            st,lw = styles[m]; a.plot(xq, g.slice(t0,xq), st, lw=lw, label=m)
        a.set_title(f"t={t0}"); a.set_xlabel("x"); a.grid(alpha=.3)
    ax[0].set_ylabel("u"); ax[0].legend()
    fig.suptitle("Classical reference solvers: spectral vs FDM vs FEM", y=1.05)
    plt.tight_layout(); save(fig, name); plt.show()
