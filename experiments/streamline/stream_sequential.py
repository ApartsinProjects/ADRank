# -*- coding: utf-8 -*-
"""TEST (do not apply): the UNIFIED triviality/hardening rule combining all three ideas:
  severity = max over {original features UNION PCA-whitened components} of
             max( ECDF-two-sided-tail rarity , histogram -log-density )   [edge + interior-gap + oblique]
calibrated to 5% normal FP. Measure combined catch vs components, suspects fixed, and how many of the
184 datasets still pass n_eff>=100 under this stronger hardening (=> resulting benchmark size)."""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
def n_eff(Xtr, Xh):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd; Zh = (Xh - mu) / sd
    rng = np.random.default_rng(0); sub = Zt[rng.choice(len(Zt), min(len(Zt), 1500), replace=False)]
    r = np.median(NearestNeighbors(n_neighbors=2).fit(sub).kneighbors(sub)[0][:, 1]) + 1e-9
    if len(Zh) > 600: Zh = Zh[rng.choice(len(Zh), 600, replace=False)]
    nn = NearestNeighbors(radius=r).fit(Zh); cov = np.zeros(len(Zh), bool); c = 0
    for i in range(len(Zh)):
        if cov[i]: continue
        c += 1; cov[nn.radius_neighbors(Zh[i:i + 1], return_distance=False)[0]] = True
    return c
def knn_sep(Xtr, Xtn, Xh):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd; rng = np.random.default_rng(0)
    if len(Zt) > 3000: Zt = Zt[rng.choice(len(Zt), 3000, replace=False)]
    nn = NearestNeighbors(n_neighbors=min(10, len(Zt) - 1)).fit(Zt)
    Xe = np.vstack([Xtn, Xh]); ye = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]
    return apn(ye, nn.kneighbors((Xe - mu) / sd)[0].mean(1))
sus = pd.read_csv(os.path.join(D, "STREAM_SUSPECTS.csv")).set_index(["corpus", "dataset"]).knn_apnorm.to_dict()
f = pd.read_csv(os.path.join(D, "STREAM_FINAL_SET.csv")); f = f[f.n_eff >= 100]
rows = []
for _, r in f.iterrows():
    try:
        Xn_all, Xa = get_na(r.corpus, r.dataset); rng = np.random.default_rng(0)
        if len(Xn_all) > 6000: Xn_all = Xn_all[rng.choice(len(Xn_all), 6000, replace=False)]
        idx = np.arange(len(Xn_all)); rng.shuffle(idx); k = int(0.7 * len(idx)); Xtr, Xtn = Xn_all[idx[:k]], Xn_all[idx[k:]]
    except Exception: continue
    if len(Xtr) < 60 or len(Xtn) < 30 or len(Xa) < 20: continue
    so_n, so_a = severity(Xtr, Xtn), severity(Xtr, Xa)
    sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
    Ptr, Ptn, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xtn)), pca.transform(sc.transform(Xa))
    sp_n, sp_a = severity(Ptr, Ptn), severity(Ptr, Pa)
    tho=np.quantile(so_n,1-Q); thp=np.quantile(sp_n,1-Q); triv_a=(so_a>tho)|(sp_a>thp); triv_n=(so_n>tho)|(sp_n>thp)
    hard = Xa[~triv_a]
    rec = {"corpus": r.corpus, "dataset": r.dataset, "knn_orig": sus.get((r.corpus, r.dataset), np.nan),
           "catch_orig": round(float((so_a > np.quantile(so_n, 1 - Q)).mean()), 2),
           "catch_comb": round(float(triv_a.mean()), 2), "fp_comb": round(float(triv_n.mean()), 2),
           "nhard_comb": len(hard)}
    if len(hard) >= 20:
        rec["neff_comb"] = n_eff(Xtr, hard); rec["knn_after"] = round(knn_sep(Xtr, Xtn, hard), 3)
    else:
        rec["neff_comb"] = len(hard); rec["knn_after"] = np.nan
    rows.append(rec)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_COMBINED.csv"), index=False)
print(f"=== SEQUENTIAL two-stage (orig 5% OR PC 5%) TEST ({len(df)} datasets, @5% FP) ===")
print(f"  anomaly catch: original {df.catch_orig.mean():.2f} -> combined {df.catch_comb.mean():.2f}   (combined FP {df.fp_comb.mean():.2f})")
s = df[df.knn_orig > 0.7]
print(f"\n  multivariate-trivial suspects (n={len(s)}):")
print(f"    survivor kNN separability: before {s.knn_orig.mean():.2f} -> after combined-hardening {s.knn_after.mean():.2f}")
print(f"    suspects fixed (survivor kNN<0.5): {int((s.knn_after<0.5).sum())}/{len(s)}   (PC-only fixed 10/21)")
print(f"\n  IMPACT on benchmark size (n_eff>=100 after combined hardening):")
keep = df[df.neff_comb >= 100]
print(f"    would keep {len(keep)}/{len(df)} datasets  {keep.corpus.value_counts().to_dict()}")
print(f"    (vs current 184 under marginal-only hardening) -> combined drops {len(df)-len(keep)} more")
print("saved streamline/STREAM_COMBINED.csv")
