# -*- coding: utf-8 -*-
"""Why do some KEPT datasets show separable anomaly clusters in UMAP? The Stage-2 hardening is
MARGINAL (per-feature). Anomalies that are marginally-normal but JOINTLY separated survive. Measure
JOINT separability of the hard anomalies with a simple kNN density rule (not UMAP): high => the
cluster is easy for a multivariate detector (multivariate-trivial residual); moderate/low => genuinely
hard (interspersed with normals). Analysis only - changes nothing."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score
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
    Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw); return Xw[yw == 0], Xw[yw == 1]
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b] + 1e-9)))
    return sev
def apn(y, s):
    b = y.mean(); return (average_precision_score(y, s) - b) / (1 - b + 1e-12)

f = pd.read_csv(os.path.join(D, "STREAM_FINAL_SET.csv")); f = f[f.n_eff >= 100]
rows = []
for _, r in f.iterrows():
    try:
        Xn_all, Xa = get_na(r.corpus, r.dataset); rng = np.random.default_rng(0)
        if len(Xn_all) > 6000: Xn_all = Xn_all[rng.choice(len(Xn_all), 6000, replace=False)]
        idx = np.arange(len(Xn_all)); rng.shuffle(idx); ntr = int(0.7 * len(idx))
        Xtr, Xtn = Xn_all[idx[:ntr]], Xn_all[idx[ntr:]]
    except Exception: continue
    if len(Xtr) < 50 or len(Xtn) < 30 or len(Xa) < 20: continue
    thr = np.quantile(severity(Xtr, Xtn), 1 - Q); Xh = Xa[severity(Xtr, Xa) <= thr]
    if len(Xh) < 20: continue
    # standardize by train normals; kNN density detector (mean dist to 10 nearest train normals)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd
    if len(Zt) > 3000: Zt = Zt[rng.choice(len(Zt), 3000, replace=False)]
    nn = NearestNeighbors(n_neighbors=min(10, len(Zt) - 1)).fit(Zt)
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    sc = nn.kneighbors((Xte - mu) / sd)[0].mean(1)
    knn_apn = apn(yte, sc)
    # cluster separation: anomaly-to-normal NN dist vs normal-to-normal baseline
    base_nn = np.median(nn.kneighbors(Zt[:min(len(Zt), 1000)])[0][:, -1]) + 1e-9
    a2n = np.median(nn.kneighbors((Xh - mu) / sd)[0][:, -1]) / base_nn
    rows.append({"corpus": r.corpus, "dataset": r.dataset, "eff_frac": r.eff_frac, "n_hard": len(Xh),
                 "knn_apnorm": round(knn_apn, 3), "anom_dist_ratio": round(float(a2n), 2)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_SUSPECTS.csv"), index=False)
print(f"=== joint separability of KEPT hard anomalies ({len(df)} datasets) - marginal hardening leaves these ===")
print(f"  kNN-detector ap_norm on hard anomalies (high = multivariate-EASY cluster; marginal filter can't see it):")
print(f"    median {df.knn_apnorm.median():.2f}   >0.7 (multivariate-trivial): {int((df.knn_apnorm>0.7).sum())}/{len(df)}   "
      f">0.5: {int((df.knn_apnorm>0.5).sum())}   <0.3 (genuinely hard): {int((df.knn_apnorm<0.3).sum())}")
print(f"  by corpus median kNN ap_norm: " + str({c: round(g.knn_apnorm.median(), 2) for c, g in df.groupby('corpus')}))
print(f"\n  TOP 'suspect' datasets (kept but a simple kNN aces them -> multivariate-trivial anomaly clusters):")
for _, r in df.sort_values("knn_apnorm", ascending=False).head(12).iterrows():
    print(f"    {r.corpus:9s} {r.dataset[:26]:28s} kNN_apnorm={r.knn_apnorm:.2f}  anom_dist_ratio={r.anom_dist_ratio:.1f}  eff_frac={r.eff_frac:.2f}")
print("saved streamline/STREAM_SUSPECTS.csv")
