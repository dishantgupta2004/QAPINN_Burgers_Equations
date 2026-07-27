import torch, torch.nn as nn, numpy as np, time, json, os
import burgers2d_common as B


def _val_errors(model, nu, device, n=81, t_list=(0.0, 0.25, 0.5, 1.0)):
    """Relative L2 for u and v, averaged over t_list. No grad."""
    eu, ev = [], []
    with torch.no_grad():
        for te in t_list:
            X, Y, up, vp, ue, ve = B.eval_grid(model, te, n=n, nu=nu, device=device)
            eu.append(B.rel_l2(up, ue)); ev.append(B.rel_l2(vp, ve))
    return float(np.mean(eu)), float(np.mean(ev))


def _grad_norm(model):
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.detach().norm().item()**2
    return s**0.5


def train_pinn2d(model, iters=20000, n_r=8000, n_i=1000, n_b=500,
                 lam_r=1.0, lam_i=10.0, lam_b=10.0, nu=B.NU_DEFAULT,
                 lr=1e-3, resample_every=0, device="cuda",
                 log_every=100, val_every=500, tag="pinn2d",
                 outdir="runs", seed=1234, extra_meta=None):
    """Returns (history dict, path to run directory)."""
    os.makedirs(outdir, exist_ok=True)
    run_dir = os.path.join(outdir, tag)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-5)
    mse = nn.MSELoss()

    H = {k: [] for k in ["it", "loss", "L_r", "L_ru", "L_rv", "L_ic", "L_bc",
                         "lr", "gnorm", "wall"]}
    V = {k: [] for k in ["it", "rel_u", "rel_v", "wall"]}

    xr, yr, tr = B.sample_interior(n_r, device)
    xi, yi, ti, ui, vi = B.sample_initial(n_i, device, nu)
    xb, yb, tb, ub, vb = B.sample_boundary(n_b, device, nu)

    best = float("inf"); t0 = time.time()
    for it in range(iters + 1):
        if resample_every and it % resample_every == 0 and it > 0:
            xr, yr, tr = B.sample_interior(n_r, device)

        opt.zero_grad(set_to_none=True)
        r_u, r_v = B.pde_residual(model, xr, yr, tr, nu)
        L_ru = (r_u**2).mean(); L_rv = (r_v**2).mean()
        L_r = L_ru + L_rv

        pi = model(torch.cat([xi, yi, ti], 1))
        L_i = mse(pi[:, 0:1], ui) + mse(pi[:, 1:2], vi)
        pb = model(torch.cat([xb, yb, tb], 1))
        L_b = mse(pb[:, 0:1], ub) + mse(pb[:, 1:2], vb)

        loss = lam_r*L_r + lam_i*L_i + lam_b*L_b
        loss.backward()
        gn = _grad_norm(model)
        opt.step(); sched.step()

        if it % log_every == 0:
            H["it"].append(it); H["loss"].append(loss.item())
            H["L_r"].append(L_r.item()); H["L_ru"].append(L_ru.item())
            H["L_rv"].append(L_rv.item()); H["L_ic"].append(L_i.item())
            H["L_bc"].append(L_b.item()); H["lr"].append(sched.get_last_lr()[0])
            H["gnorm"].append(gn); H["wall"].append(time.time()-t0)

        if it % val_every == 0:
            ru_, rv_ = _val_errors(model, nu, device)
            V["it"].append(it); V["rel_u"].append(ru_); V["rel_v"].append(rv_)
            V["wall"].append(time.time()-t0)
            score = 0.5*(ru_+rv_)
            if score < best:
                best = score
                torch.save({"state_dict": model.state_dict(),
                            "it": it, "rel_u": ru_, "rel_v": rv_},
                           os.path.join(run_dir, "best.pt"))
            print(f"{it:6d}  L={loss.item():.3e}  r={L_r.item():.3e} "
                  f"ic={L_i.item():.3e} bc={L_b.item():.3e} | "
                  f"relL2 u={ru_:.3e} v={rv_:.3e} | {time.time()-t0:.0f}s")

    hist = {"train": H, "val": V}
    np.savez(os.path.join(run_dir, "history.npz"),
             **{f"train_{k}": np.array(v) for k, v in H.items()},
             **{f"val_{k}": np.array(v) for k, v in V.items()})

    meta = {"tag": tag, "seed": seed, "iters": iters, "n_r": n_r, "n_i": n_i,
            "n_b": n_b, "lam_r": lam_r, "lam_i": lam_i, "lam_b": lam_b,
            "nu": nu, "lr": lr, "resample_every": resample_every,
            "device": str(device), "best_mean_relL2": best,
            "n_params": sum(p.numel() for p in model.parameters()),
            "wall_total_s": time.time()-t0,
            "torch": torch.__version__}
    if extra_meta: meta.update(extra_meta)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    torch.save({"state_dict": model.state_dict(), "meta": meta},
               os.path.join(run_dir, "final.pt"))
    return hist, run_dir


def lbfgs_polish(model, iters=500, n_r=8000, n_i=1000, n_b=500,
                 lam_r=1.0, lam_i=10.0, lam_b=10.0, nu=B.NU_DEFAULT,
                 device="cuda", seed=1234):
    """Second-order polish after Adam. Full-batch, fixed collocation set."""
    torch.manual_seed(seed)
    model.to(device)
    mse = nn.MSELoss()
    xr, yr, tr = B.sample_interior(n_r, device)
    xi, yi, ti, ui, vi = B.sample_initial(n_i, device, nu)
    xb, yb, tb, ub, vb = B.sample_boundary(n_b, device, nu)

    opt = torch.optim.LBFGS(model.parameters(), max_iter=iters,
                            history_size=50, tolerance_grad=1e-9,
                            tolerance_change=1e-11,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        r_u, r_v = B.pde_residual(model, xr, yr, tr, nu)
        L_r = (r_u**2).mean() + (r_v**2).mean()
        pi = model(torch.cat([xi, yi, ti], 1))
        L_i = mse(pi[:, 0:1], ui) + mse(pi[:, 1:2], vi)
        pb = model(torch.cat([xb, yb, tb], 1))
        L_b = mse(pb[:, 0:1], ub) + mse(pb[:, 1:2], vb)
        loss = lam_r*L_r + lam_i*L_i + lam_b*L_b
        loss.backward()
        return loss

    opt.step(closure)
    return model
