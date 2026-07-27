import numpy as np, torch
from config import DEVICE
from ground_truth import rel_l2

@torch.no_grad()
def predict_grid(model, x, t, batch=4096):
    X, T = np.meshgrid(x, t, indexing="xy")
    P = np.stack([X.ravel(), T.ravel()], 1).astype(np.float32)
    out = []
    for i in range(0, len(P), batch):
        xb = torch.from_numpy(P[i:i+batch]).to(DEVICE)
        out.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(out).reshape(len(t), len(x))

def evaluate(model, gt, nx=401, nt=101, t_max=1.0):
    x = np.linspace(-1,1,nx); t = np.linspace(0,t_max,nt)
    Up = predict_grid(model, x, t)
    Ug = gt.interp(np.stack(np.meshgrid(t,x,indexing="ij"),-1)[...,[0,1]])
    Ug = np.array([gt.slice(tt, x) for tt in t])
    return dict(x=x, t=t, U_pred=Up, U_true=Ug,
                l2_global=rel_l2(Up, Ug),
                l2_per_t=np.array([rel_l2(Up[i], Ug[i]) for i in range(nt)]),
                linf=float(np.abs(Up-Ug).max()))

def l2_table(models: dict, gt, **kw):
    rows = {}
    for name, m in models.items():
        r = evaluate(m, gt, **kw)
        rows[name] = r
        print(f"{name:24s} | rel-L2 {r['l2_global']:.4e} | Linf {r['linf']:.4e}")
    return rows
