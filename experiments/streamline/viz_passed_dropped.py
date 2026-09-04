# -*- coding: utf-8 -*-
"""Passed vs Dropped datasets, embedded with UMAP AND t-SNE side by side. Normals grey, anomalies red."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(200): MTS[str(name)[:40]] = (Xc, lab)
def get_na(corpus, name):
    if corpus in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
        y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
        return X[y == 0], X[y == 1]
    if corpus in ("adbench", "dami"):
        X, y = OBJ[(corpus, name)]; return X[y == 0], X[y == 1]
    Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw); return Xw[yw == 0], Xw[yw == 1]
def emb_umap(Z):
    import umap; return umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(Z)
def emb_tsne(Z):
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Z)

allc = pd.read_csv(os.path.join(D, "STREAM_FINAL2_ALL.csv"))
kept = set(map(tuple, pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv"))[["corpus", "dataset"]].values.tolist()))
def reason(r):
    if r.n_norm < 800: return "few_normals"
    if r.frac_triv >= 0.90: return "OR_solvable"
    if r.n_eff < 100: return "few_distinct"
    return "kept"
allc["reason"] = allc.apply(reason, axis=1); allc["kept"] = allc.apply(lambda r: (r.corpus, r.dataset) in kept, axis=1)
keptdf = allc[allc.kept].sort_values("n_eff")
p_pick = keptdf.iloc[[len(keptdf) // 3, 2 * len(keptdf) // 3]]                       # 2 passed
d_or = allc[(~allc.kept) & (allc.reason == "OR_solvable")].head(1)                    # 1 dropped: easy-separation
d_fd = allc[(~allc.kept) & (allc.reason == "few_distinct") & (allc.frac_triv < 0.3)].sort_values("n_eff").head(1)  # 1 dropped: few-distinct/embedded
PANELS = [("PASSED", p_pick.iloc[0]), ("PASSED", p_pick.iloc[1]), ("DROPPED", d_or.iloc[0]), ("DROPPED", d_fd.iloc[0])]
fig, axes = plt.subplots(4, 2, figsize=(9, 18))
for ri, (tag, r) in enumerate(PANELS):
    Xn, Xa = get_na(r.corpus, r.dataset); rng = np.random.default_rng(0)
    if len(Xn) > 800: Xn = Xn[rng.choice(len(Xn), 800, replace=False)]
    if len(Xa) > 250: Xa = Xa[rng.choice(len(Xa), 250, replace=False)]
    Z = StandardScaler().fit_transform(np.vstack([Xn, Xa]))
    for ci, (mname, fn) in enumerate([("UMAP", emb_umap), ("t-SNE", emb_tsne)]):
        ax = axes[ri, ci]; ax.set_xticks([]); ax.set_yticks([])
        try:
            E = fn(Z); en, ea = E[:len(Xn)], E[len(Xn):]
            ax.scatter(en[:, 0], en[:, 1], s=7, c="#9aa4b2", alpha=0.5, linewidths=0)
            ax.scatter(ea[:, 0], ea[:, 1], s=16, c="#d1495b", alpha=0.85, edgecolors="white", linewidths=0.2)
        except Exception as e:
            ax.text(0.5, 0.5, "skip", ha="center")
        if ci == 0: ax.set_ylabel(f"{tag}\n{r.corpus}/{r.dataset[:16]}\n{'' if tag=='PASSED' else r.reason}", fontsize=9)
        if ri == 0: ax.set_title(mname, fontsize=13, fontweight="bold")
fig.suptitle("Passed vs Dropped - UMAP vs t-SNE (normals grey, anomalies red)", fontsize=13, y=1.001)
fig.tight_layout()
out = os.path.join(S, "FIG_passed_dropped_umap_tsne.png"); fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
