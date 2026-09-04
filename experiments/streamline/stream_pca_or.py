# -*- coding: utf-8 -*-
"""TEST (do not apply): an OR triviality filter in PCA-TRANSFORMED space. The original-feature OR
is blind to anomalies separable along OBLIQUE/principal directions (the multivariate-trivial gap).
A PC-space OR (per-PC calibrated rarity on whitened PCs) should catch those. Measure whether it flags
the multivariate-trivial suspects and how much extra it would drop/harden vs the original OR."""
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
def knn_sep(Xtr, Xtn, Xh):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd
    rng = np.random.default_rng(0)
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
        idx = np.arange(len(Xn_all)); rng.shuffle(idx); k = int(0.7 * len(idx))
        Xtr, Xtn = Xn_all[idx[:k]], Xn_all[idx[k:]]
    except Exception: continue
    if len(Xtr) < 60 or len(Xtn) < 30 or len(Xa) < 20: continue
    # ORIGINAL-space OR
    thr_o = np.quantile(severity(Xtr, Xtn), 1 - Q); hard_o = Xa[severity(Xtr, Xa) <= thr_o]
    catch_o = float((severity(Xtr, Xa) > thr_o).mean())
    # PCA (whitened, top comps for 95% var) then same OR in PC space
    sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
    Ptr, Ptn, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xtn)), pca.transform(sc.transform(Xa))
    thr_p = np.quantile(severity(Ptr, Ptn), 1 - Q); sev_pa = severity(Ptr, Pa)
    catch_p = float((sev_pa > thr_p).mean()); fp_p = float((severity(Ptr, Ptn) > thr_p).mean())
    hard_p = Xa[sev_pa <= thr_p]
    rec = {"corpus": r.corpus, "dataset": r.dataset, "n_pc": pca.n_components_, "knn_orig": sus.get((r.corpus, r.dataset), np.nan),
           "catch_orig": round(catch_o, 2), "catch_pc": round(catch_p, 2), "fp_pc": round(fp_p, 2),
           "nhard_orig": len(hard_o), "nhard_pc": len(hard_p)}
    if len(hard_p) >= 15 and len(hard_o) >= 15:
        rec["knn_after_pc"] = round(knn_sep(Xtr, Xtn, hard_p), 3)
    rows.append(rec)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_PCA_OR.csv"), index=False)
print(f"=== PC-space OR filter TEST ({len(df)} datasets) ===")
print(f"  anomaly catch-rate:  original-feature OR {df.catch_orig.mean():.2f}   PC-space OR {df.catch_pc.mean():.2f}   (PC normal-FP {df.fp_pc.mean():.2f})")
print(f"  PC-OR catches MORE anomalies than original OR on {int((df.catch_pc>df.catch_orig+0.05).sum())}/{len(df)} datasets")
sus_df = df[df.knn_orig > 0.7]  # the multivariate-trivial suspects
print(f"\n  MULTIVARIATE-TRIVIAL suspects (knn_orig>0.7), n={len(sus_df)}:")
print(f"    original-OR catch {sus_df.catch_orig.mean():.2f}  ->  PC-OR catch {sus_df.catch_pc.mean():.2f}   "
      f"(PC-OR catches the joint cluster the original missed)")
if "knn_after_pc" in sus_df:
    print(f"    kNN separability of survivors: before {sus_df.knn_orig.mean():.2f}  ->  after PC-hardening {sus_df.knn_after_pc.mean():.2f}")
print(f"\n  did PC-OR fix suspects? (survivor kNN dropped <0.5): {int((sus_df.get('knn_after_pc', pd.Series())<0.5).sum())}/{len(sus_df)}")
print(f"  datasets PC-OR would newly make sub-threshold (nhard_pc<0.5*nhard_orig): {int((df.nhard_pc<0.5*df.nhard_orig).sum())}/{len(df)}")
print("saved streamline/STREAM_PCA_OR.csv")
