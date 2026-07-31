import torch, torch.nn as nn, numpy as np, time, json, os
import burgers2d_common as B
from train2d import _val_errors, _grad_norm

def train_qapinn2d(model, iters=5000, n_r=2000, n_i=400, n_b=200,
                   lam_r=1.0, lam_i=10.0, lam_b=10.0, nu=B.NU_DEFAULT,
                   lr=5e-3, device="cuda", log_every=25, val_every=250,
                   tag="qapinn2d", outdir="runs", seed=1234,
                   chunk=512, extra_meta=None,
                   snapshot_every=0, snapshot_cb=None, resample_every=0,
                   return_best=False):
    """Train a QPINN2D for the 2D Burgers system.

    chunk : QNode micro-batch size. Tune to fit CPU memory of the double-backward
            graph; 512 is safe for n_qubits<=6.
    snapshot_every : int
        If >0, call snapshot_cb(it, model) every snapshot_every iterations (plus
        once at it==0 and at the final iter). Captures q_weights DURING a single
        continuous run, so Adam momentum and the cosine LR schedule are never
        reset (unlike splitting into multiple train_qapinn2d calls).
    snapshot_cb : callable(it:int, model) -> None
        User hook. Copy what you need out of model here (cheap).
    resample_every : int
        If >0, resample interior collocation points every N iters. 0 = fixed set.
    return_best : bool
        If True, reload best (lowest-val) weights into model before returning.
    """
    run_dir = os.path.join(outdir, tag); os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-4)
    mse = nn.MSELoss()

    H = {k: [] for k in ["it", "loss", "L_r", "L_ru", "L_rv", "L_ic", "L_bc",
                         "lr", "gnorm", "wall", "qw_norm"]}
    V = {k: [] for k in ["it", "rel_u", "rel_v", "wall"]}

    xr, yr, tr_ = B.sample_interior(n_r, device)
    xi, yi, ti, ui, vi = B.sample_initial(n_i, device, nu)
    xb, yb, tb, ub, vb = B.sample_boundary(n_b, device, nu)

    if snapshot_every and snapshot_cb is not None:
        snapshot_cb(0, model)                       # snapshot at init

    best = float("inf"); best_path = os.path.join(run_dir, "best.pt")
    t0 = time.time()
    for it in range(iters + 1):
        if resample_every and it % resample_every == 0 and it > 0:
            xr, yr, tr_ = B.sample_interior(n_r, device)

        opt.zero_grad(set_to_none=True)
        L_ru = torch.zeros((), device=device); L_rv = torch.zeros((), device=device)
        nchunk = (n_r + chunk - 1)//chunk
        for c in range(nchunk):
            sl = slice(c*chunk, min((c+1)*chunk, n_r))
            ru, rv = B.pde_residual(model, xr[sl].clone(), yr[sl].clone(),
                                    tr_[sl].clone(), nu)
            L_ru = L_ru + (ru**2).sum(); L_rv = L_rv + (rv**2).sum()
        L_ru = L_ru/n_r; L_rv = L_rv/n_r
        L_r = L_ru + L_rv

        pi = model(torch.cat([xi, yi, ti], 1))
        L_i = mse(pi[:, 0:1], ui) + mse(pi[:, 1:2], vi)
        pb = model(torch.cat([xb, yb, tb], 1))
        L_b = mse(pb[:, 0:1], ub) + mse(pb[:, 1:2], vb)

        loss = lam_r*L_r + lam_i*L_i + lam_b*L_b
        loss.backward()
        gn = _grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if it % log_every == 0:
            H["it"].append(it); H["loss"].append(loss.item())
            H["L_r"].append(L_r.item()); H["L_ru"].append(L_ru.item())
            H["L_rv"].append(L_rv.item()); H["L_ic"].append(L_i.item())
            H["L_bc"].append(L_b.item()); H["lr"].append(sched.get_last_lr()[0])
            H["gnorm"].append(gn); H["wall"].append(time.time()-t0)
            H["qw_norm"].append(model.q_weights.detach().norm().item())

        if it % val_every == 0:
            ru_, rv_ = _val_errors(model, nu, device, n=41)
            V["it"].append(it); V["rel_u"].append(ru_); V["rel_v"].append(rv_)
            V["wall"].append(time.time()-t0)
            s = 0.5*(ru_+rv_)
            if s < best:
                best = s
                torch.save({"state_dict": model.state_dict(),
                            "n_qubits": model.n_qubits,
                            "n_layers": model.n_layers, "reupload": model.reupload,
                            "it": it, "rel_u": ru_, "rel_v": rv_}, best_path)
            print(f"{it:5d}  L={loss.item():.3e}  relL2 u={ru_:.3e} v={rv_:.3e} "
                  f"| {time.time()-t0:.0f}s")

        if (snapshot_every and snapshot_cb is not None
                and it > 0 and (it % snapshot_every == 0 or it == iters)):
            snapshot_cb(it, model)

    np.savez(os.path.join(run_dir, "history.npz"),
             **{f"train_{k}": np.array(v) for k, v in H.items()},
             **{f"val_{k}": np.array(v) for k, v in V.items()})
    meta = {"tag": tag, "seed": seed, "iters": iters, "n_r": n_r, "chunk": chunk,
            "n_qubits": model.n_qubits, "n_layers": model.n_layers,
            "reupload": model.reupload, "nu": nu, "lr": lr,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_q_params": model.q_weights.numel(),
            "best_mean_relL2": best, "wall_total_s": time.time()-t0}
    if extra_meta: meta.update(extra_meta)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "meta": meta},
               os.path.join(run_dir, "final.pt"))

    if return_best and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[return_best] reloaded best.pt from it={ckpt['it']} "
              f"(rel_u={ckpt['rel_u']:.3e} rel_v={ckpt['rel_v']:.3e})")

    return {"train": H, "val": V}, run_dir