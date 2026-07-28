import time, numpy as np, torch
from config import DEVICE, set_seed
from physics import build_batches, composite_loss
from models import ClassicalPINN

def train_classical(depth=4, width=8, adam_epochs=600, lbfgs_iter=0,
                    lr=2e-3, seed=0, log_every=500, snapshot_epochs=()):
    set_seed(seed)
    model = ClassicalPINN(depth, width).to(DEVICE)
    B = build_batches(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_epochs, 1e-5)

    hist = {"iter": [], "wall": [], "total": [], "pde": [], "ic": [], "bc": [], "phase": []}
    snaps = {}
    t0 = time.perf_counter(); it = 0

    for ep in range(adam_epochs):
        opt.zero_grad(set_to_none=True)
        loss, parts = composite_loss(model, B)
        loss.backward(); opt.step(); sch.step()
        hist["iter"].append(it); hist["wall"].append(time.perf_counter()-t0)
        for k in ("total","pde","ic","bc"): hist[k].append(parts[k])
        hist["phase"].append(0); it += 1
        if ep in snapshot_epochs:
            snaps[ep] = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep % log_every == 0 or ep == adam_epochs-1:
            print(f"[adam ] {ep:6d} | {parts['total']:.4e} | pde {parts['pde']:.2e} "
                  f"ic {parts['ic']:.2e} bc {parts['bc']:.2e} | {hist['wall'][-1]:.1f}s")

    opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iter,
                             history_size=100, tolerance_grad=1e-11,
                             tolerance_change=1e-14, line_search_fn="strong_wolfe")
    state = {"it": it}
    def closure():
        opt2.zero_grad(set_to_none=True)
        loss, parts = composite_loss(model, B)
        loss.backward()
        hist["iter"].append(state["it"]); hist["wall"].append(time.perf_counter()-t0)
        for k in ("total","pde","ic","bc"): hist[k].append(parts[k])
        hist["phase"].append(1); state["it"] += 1
        return loss
    print("[lbfgs] polishing ...")
    opt2.step(closure)
    print(f"[lbfgs] final {hist['total'][-1]:.4e} | wall {hist['wall'][-1]:.1f}s")
    hist = {k: np.asarray(v) for k,v in hist.items()}
    hist["adam_epochs"] = adam_epochs
    return model, hist, snaps
