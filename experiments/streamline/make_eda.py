# -*- coding: utf-8 -*-
"""EDA report on the 184-dataset streamlined benchmark (construction characteristics)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
f = pd.read_csv(os.path.join(D, "STREAM_FINAL_SET.csv")); f = f[f.n_eff >= 100].copy()
src = pd.read_csv(os.path.join(D, "STREAM_SOURCE2.csv"))[["corpus", "dataset", "d"]]
f = f.merge(src, on=["corpus", "dataset"], how="left")
f["base_rate"] = f.n_hard / (f.n_hard + f.n_norm)
CORP = ["ovrbench", "oddbench", "tsbad_m", "adbench", "dami"]
CL = {"ovrbench": "#2a7fb8", "oddbench": "#5aa469", "tsbad_m": "#d1495b", "adbench": "#e0a12b", "dami": "#8c6bb1"}
print(f"=== EDA: streamlined benchmark, {len(f)} datasets ===")
print(f"  {'corpus':10s} {'n':>4s} {'med_d':>6s} {'med_norm':>9s} {'med_hard':>9s} {'med_distinct':>13s} {'med_effrac':>11s} {'med_baserate':>13s}")
for c in CORP:
    g = f[f.corpus == c]
    if len(g): print(f"  {c:10s} {len(g):4d} {int(g.d.median()):6d} {int(g.n_norm.median()):9d} {int(g.n_hard.median()):9d} {int(g.n_eff.median()):13d} {g.eff_frac.median():11.2f} {g.base_rate.median():13.2f}")
print(f"  {'ALL':10s} {len(f):4d} {int(f.d.median()):6d} {int(f.n_norm.median()):9d} {int(f.n_hard.median()):9d} {int(f.n_eff.median()):13d} {f.eff_frac.median():11.2f} {f.base_rate.median():13.2f}")

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
# A composition
vc = f.corpus.value_counts().reindex(CORP).fillna(0)
ax[0, 0].bar(range(len(CORP)), vc.values, color=[CL[c] for c in CORP]); ax[0, 0].set_xticks(range(len(CORP))); ax[0, 0].set_xticklabels(CORP, rotation=30, ha="right")
ax[0, 0].set_title(f"Composition ({len(f)} datasets)"); ax[0, 0].set_ylabel("datasets")
for i, v in enumerate(vc.values): ax[0, 0].text(i, v + 0.5, int(v), ha="center", fontsize=9)
# helper for log-hist by corpus
def panel(axx, col, title, xlabel, logx=True):
    for c in CORP:
        g = f[f.corpus == c]
        if len(g): axx.hist(np.log10(g[col].clip(lower=1)) if logx else g[col], bins=20, alpha=0.6, color=CL[c], label=c)
    axx.set_title(title); axx.set_xlabel(xlabel); axx.set_ylabel("datasets")
panel(ax[0, 1], "d", "Dimensionality", "log10(features)")
panel(ax[0, 2], "n_norm", "Normals (modeling pool)", "log10(normals)")
panel(ax[1, 0], "n_hard", "Hard anomalies", "log10(hard anomalies)")
panel(ax[1, 1], "n_eff", "Distinct hard anomalies (n_eff)", "log10(distinct anomalies)")
# eff_frac + base_rate (linear)
for c in CORP:
    g = f[f.corpus == c]
    if len(g): ax[1, 2].scatter(g.eff_frac, g.base_rate, s=18, c=CL[c], alpha=0.6, label=c)
ax[1, 2].set_title("Diversity vs base rate"); ax[1, 2].set_xlabel("eff_frac (anomaly distinctness)"); ax[1, 2].set_ylabel("base rate")
ax[1, 2].legend(fontsize=8)
fig.suptitle("Streamlined HADB benchmark - EDA (184 datasets; solvability + family pending scoring)", fontsize=13)
fig.tight_layout()
out = os.path.join(S, "FIG_stream_eda.png"); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
