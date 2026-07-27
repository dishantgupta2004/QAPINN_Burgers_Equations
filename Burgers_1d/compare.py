import numpy as np, matplotlib.pyplot as plt, pandas as pd
from ground_truth import rel_l2

def summary_table(res_dict, hists, param_counts):
    rows = []
    for k, r in res_dict.items():
        h = hists[k]
        rows.append(dict(model=k, params=param_counts[k],
                         rel_L2=r["l2_global"], Linf=r["linf"],
                         final_loss=h["total"][-1],
                         iters=len(h["iter"]), wall_s=h["wall"][-1],
                         s_per_1k_iter=1000*h["wall"][-1]/len(h["iter"])))
    df = pd.DataFrame(rows).set_index("model")
    return df

def l2_vs_time_plot(res_dict, name="l2_vs_t"):
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    for k,r in res_dict.items():
        ax.semilogy(r["t"], r["l2_per_t"], lw=1.8, label=k)
    ax.set(xlabel="t", ylabel="relative $L_2$ vs spectral GT",
           title="Error growth through the shock formation")
    ax.grid(alpha=.3); ax.legend(); plt.tight_layout()
    fig.savefig(f"outputs/{name}.png", dpi=160, bbox_inches="tight"); plt.show()
    return fig