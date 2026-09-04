# -*- coding: utf-8 -*-
"""Grid of UMAP embeddings: row per source, 4 datasets each, normals (grey) + hard anomalies (red)."""
import os, sys, io, zipfile, re, warnings
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
Q = 0.05
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
    Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
    return Xw[yw == 0], Xw[yw == 1]


def severity(Xn, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xn[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xn[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b] + 1e-9)))
    return sev


def embed(Xn, Xa):
    Z = StandardScaler().fit_transform(np.vstack([Xn, Xa]))
    try:
        import umap; E = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(Z); algo = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA; E = PCA(2, random_state=0).fit_transform(Z); algo = "PCA"
    return E[:len(Xn)], E[len(Xn):], algo


f = pd.read_csv(os.path.join(D, "STREAM_FINAL_SET.csv")); f = f[f.n_eff >= 100]
CORP = ["ovrbench", "oddbench", "tsbad_m", "adbench", "dami"]
fig, axes = plt.subplots(len(CORP), 4, figsize=(16, 5 * len(CORP)))
algo_used = "?"
for ri, corp in enumerate(CORP):
    g = f[f.corpus == corp].sort_values("eff_frac")
    pick = g.iloc[np.linspace(0, len(g) - 1, min(4, len(g))).astype(int)] if len(g) else g
    for ci in range(4):
        ax = axes[ri, ci]; ax.set_xticks([]); ax.set_yticks([])
        if ci >= len(pick):
            ax.axis("off"); continue
        r = pick.iloc[ci]
        try:
            Xn, Xa = get_na(corp, r.dataset)
            rng = np.random.default_rng(0)
            if len(Xn) > 1200: Xn = Xn[rng.choice(len(Xn), 1200, replace=False)]
            thr = np.quantile(severity(Xn, Xn), 1 - Q); Xh = Xa[severity(Xn, Xa) <= thr]
            if len(Xh) > 300: Xh = Xh[rng.choice(len(Xh), 300, replace=False)]
            en, ea, algo_used = embed(Xn, Xh)
            ax.scatter(en[:, 0], en[:, 1], s=6, c="#9aa4b2", alpha=0.5, linewidths=0)
            ax.scatter(ea[:, 0], ea[:, 1], s=14, c="#d1495b", alpha=0.8, edgecolors="white", linewidths=0.2)
            ax.set_title(f"{r.dataset[:22]}\neff_frac={r.eff_frac:.2f} d={int(r.n_eff)} hard", fontsize=9)
        except Exception as e:
            ax.set_title(f"{r.dataset[:20]} (skip)", fontsize=8); ax.axis("off")
    axes[ri, 0].set_ylabel(corp, fontsize=13, fontweight="bold")
fig.suptitle(f"Streamlined benchmark: normals (grey) + hard anomalies (red), {algo_used} 2D - 4 datasets per source", fontsize=14, y=1.002)
fig.tight_layout()
out = os.path.join(S, "FIG_stream_umap_grid.png"); fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out, "algo", algo_used)
