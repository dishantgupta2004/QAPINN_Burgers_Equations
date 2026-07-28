import time, numpy as np, torch
from config import DEVICE, set_seed
from physics import build_batches, composite_loss
from models import QAPINN

def train_qapinn(n_qubits=3, n_layers=2, hidden=8, reupload=True,
                 epochs=600, lr=2e-3, seed=0, n_pde=256, n_ic=128, n_bc=128,
                 log_every=200, snapshot_epochs=()):
    set_seed(seed)
    model = QAPINN(n_qubits, hidden=hidden, n_layers=n_layers,
                   reupload=reupload).to(DEVICE)
    B = build_batches(n_pde=n_pde, n_ic=n_ic, n_bc=n_bc, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-5)

    hist = {"iter": [], "wall": [], "total": [], "pde": [], "ic": [], "bc": []}
    snaps = {}
    t0 = time.perf_counter()
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss, parts = composite_loss(model, B)
        loss.backward(); opt.step(); sch.step()
        hist["iter"].append(ep); hist["wall"].append(time.perf_counter()-t0)
        for k in ("total","pde","ic","bc"): hist[k].append(parts[k])
        if ep in snapshot_epochs:
            snaps[ep] = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep % log_every == 0 or ep == epochs-1:
            print(f"[{n_qubits}q L{n_layers} ru={int(reupload)}] {ep:5d} | "
                  f"{parts['total']:.4e} | pde {parts['pde']:.2e} "
                  f"ic {parts['ic']:.2e} bc {parts['bc']:.2e} | {hist['wall'][-1]:.1f}s")
    hist = {k: np.asarray(v) for k,v in hist.items()}
    return model, hist, snaps
