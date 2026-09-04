# -*- coding: utf-8 -*-
"""Grid of UMAP embeddings for DROPPED datasets (triviality reasons), row per source.
Validates streamlining: dropped-as-trivial datasets should look visibly easy (anomalies separable
or in tight clumps). Grey=normals, red=anomalies."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as P
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in P.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
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
def embed(Xn, Xa):
    Z = StandardScaler().fit_transform(np.vstack([Xn, Xa]))
    try:
        import umap; E = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(Z)
    except Exception:
        from sklearn.decomposition import PCA; E = PCA(2, random_state=0).fit_transform(Z)
    return E[:len(Xn)], E[len(Xn):]

a = pd.read_csv(os.path.join(D, "STREAM_FINAL_ALL.csv"))
def reason(r):
    if r.n_norm < 800: return "few_normals"
    if r.frac_triv >= 0.90: return "OR_solvable"
    if r.n_hard < 100 or r.n_eff < 100: return "few_distinct"
    return "kept"
a["reason"] = a.apply(reason, axis=1)
triv = a[a.reason.isin(["OR_solvable", "few_distinct"])]
CORP = ["ovrbench", "oddbench", "tsbad_m", "adbench"]
fig, axes = plt.subplots(len(CORP), 4, figsize=(16, 4 * len(CORP)))
for ri, corp in enumerate(CORP):
    g = triv[triv.corpus == corp].sort_values("reason")  # OR_solvable first
    pick = g.iloc[np.linspace(0, len(g) - 1, min(4, len(g))).astype(int)] if len(g) else g
    for ci in range(4):
        ax = axes[ri, ci]; ax.set_xticks([]); ax.set_yticks([])
        if ci >= len(pick): ax.axis("off"); continue
        r = pick.iloc[ci]
        try:
            Xn, Xa = get_na(corp, r.dataset); rng = np.random.default_rng(0)
            if len(Xn) > 1200: Xn = Xn[rng.choice(len(Xn), 1200, replace=False)]
            if len(Xa) > 300: Xa = Xa[rng.choice(len(Xa), 300, replace=False)]
            en, ea = embed(Xn, Xa)
            ax.scatter(en[:, 0], en[:, 1], s=6, c="#9aa4b2", alpha=0.5, linewidths=0)
            ax.scatter(ea[:, 0], ea[:, 1], s=16, c="#d1495b", alpha=0.85, edgecolors="white", linewidths=0.2)
            ax.set_title(f"{r.dataset[:20]}\nDROPPED: {r.reason} (triv={r.frac_triv:.2f}, n_eff={int(r.n_eff)})", fontsize=8.5)
        except Exception:
            ax.set_title(f"{r.dataset[:18]} (skip)", fontsize=8); ax.axis("off")
    axes[ri, 0].set_ylabel(corp, fontsize=13, fontweight="bold")
fig.suptitle("DROPPED datasets (triviality reasons): normals (grey) + anomalies (red) - validates streamlining", fontsize=13, y=1.003)
fig.tight_layout()
out = os.path.join(S, "FIG_stream_dropped_grid.png"); fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
