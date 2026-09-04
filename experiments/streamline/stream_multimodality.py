# -*- coding: utf-8 -*-
"""Standard multimodality statistics on the NORMALS of each streamlined dataset:
  dip_frac  fraction of features rejecting unimodality (Hartigan dip test, p<0.05)  [gold standard]
  mean_dip  mean Hartigan dip statistic across features
  gmm_k     BIC-optimal GMM component count (1..6) on standardized PCA(<=8) normals
  silh      best k-means silhouette (k=2..5)  (multivariate cluster clarity)
  bc_frac   fraction of features with Sarle bimodality coefficient > 0.555
"""
import os, sys, warnings
import numpy as np, pandas as pd
import diptest
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
Dd = os.path.join(S, "scratchpad", "streamline")
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


def normals(corpus, name):
    if corpus in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
        y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
        return X[y == 0]
    if corpus in ("adbench", "dami"):
        X, y = OBJ[(corpus, name)]; return X[y == 0]
    Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); return np.nan_to_num(Xw)[yw == 0]


def multimodality(Xn):
    rng = np.random.default_rng(0)
    if len(Xn) > 2000: Xn = Xn[rng.choice(len(Xn), 2000, replace=False)]
    Z = StandardScaler().fit_transform(Xn); n, d = Z.shape
    # per-feature dip test + bimodality coefficient
    dips, ps, bcs = [], [], []
    corr = 3 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
    for j in range(d):
        c = Z[:, j]
        if np.std(c) < 1e-9: continue
        try:
            dp, pv = diptest.diptest(c); dips.append(dp); ps.append(pv)
        except Exception: pass
        g = skew(c); ku = kurtosis(c, fisher=True); bcs.append((g * g + 1) / (ku + corr))
    dip_frac = float(np.mean(np.array(ps) < 0.05)) if ps else np.nan   # MARGINAL (per-feature)
    mean_dip = float(np.mean(dips)) if dips else np.nan
    bc_frac = float(np.mean(np.array(bcs) > 0.555)) if bcs else np.nan
    # JOINT: dip on RANDOM PROJECTIONS (direction-agnostic; catches oblique joint multimodality)
    R = 60; Vv = rng.standard_normal((d, R)); Vv /= (np.linalg.norm(Vv, axis=0, keepdims=True) + 1e-12)
    pj = []
    for r in range(R):
        try: pj.append(diptest.diptest(Z @ Vv[:, r])[1])
        except Exception: pass
    dip_proj_frac = float(np.mean(np.array(pj) < 0.05)) if pj else np.nan     # frac of directions multimodal
    dip_proj_minp = float(np.min(pj)) if pj else np.nan                       # strongest multimodal direction
    # multivariate: GMM BIC-k (uncapped to 15) and silhouette on PCA
    Zp = PCA(min(8, d), random_state=0).fit_transform(Z) if d > 1 else Z
    bic = {}
    for k in range(1, 16):
        try: bic[k] = GaussianMixture(k, reg_covar=1e-3, random_state=0).fit(Zp).bic(Zp)
        except Exception: pass
    gmm_k = int(min(bic, key=bic.get)) if bic else 1
    # JOINT (projection-pursuit): dip along the discriminant axis between the two dominant GMM modes
    try:
        g2 = GaussianMixture(2, reg_covar=1e-3, random_state=0).fit(Zp); ax = g2.means_[1] - g2.means_[0]
        ax = ax / (np.linalg.norm(ax) + 1e-12); bestdir_p = diptest.diptest(Zp @ ax)[1]
    except Exception:
        bestdir_p = np.nan
    silh = 0.0
    for k in range(2, 6):
        try:
            lab = KMeans(k, n_init=5, random_state=0).fit_predict(Zp)
            if len(set(lab)) > 1: silh = max(silh, silhouette_score(Zp, lab))
        except Exception: pass
    return dip_frac, mean_dip, bc_frac, gmm_k, float(silh), dip_proj_frac, dip_proj_minp, bestdir_p


f = pd.read_csv(os.path.join(Dd, "STREAM_FINAL_SET.csv")); f = f[f.n_eff >= 100]
rows = []
for i, (_, r) in enumerate(f.iterrows()):
    try:
        Xn = normals(r.corpus, r.dataset)
        if len(Xn) < 30: continue
        dip_frac, mean_dip, bc_frac, gmm_k, silh, dip_proj_frac, dip_proj_minp, bestdir_p = multimodality(Xn)
        rows.append({"corpus": r.corpus, "dataset": r.dataset, "dip_frac": dip_frac, "mean_dip": mean_dip,
                     "bc_frac": bc_frac, "gmm_k": gmm_k, "silh": silh, "dip_proj_frac": dip_proj_frac,
                     "dip_proj_minp": dip_proj_minp, "bestdir_p": bestdir_p})
    except Exception: continue
    if i % 40 == 0: print(f"  ..{i}/{len(f)}", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(Dd, "STREAM_MULTIMODALITY.csv"), index=False)
print(f"\n=== multimodality of streamlined datasets' normals ({len(df)}) ===")
print(f"  {'corpus':10s} {'n':>4s} {'MARGINAL':>9s} {'JOINT':>7s} {'silh':>6s} {'gmm_k':>6s} {'bc_frac':>8s}")
print(f"  {'':10s} {'':4s} {'dip_frac':>9s} {'projdip':>7s}")
for c in ["ovrbench", "oddbench", "tsbad_m", "adbench", "dami"]:
    g = df[df.corpus == c]
    if len(g): print(f"  {c:10s} {len(g):4d} {g.dip_frac.median():9.2f} {g.dip_proj_frac.median():7.2f} {g.silh.median():6.2f} {g.gmm_k.median():6.1f} {g.bc_frac.median():8.2f}")
print(f"  {'ALL':10s} {len(df):4d} {df.dip_frac.median():9.2f} {df.dip_proj_frac.median():7.2f} {df.silh.median():6.2f} {df.gmm_k.median():6.1f} {df.bc_frac.median():8.2f}")
print(f"\n  MARGINAL dip_frac = frac of FEATURES multimodal;  JOINT projdip = frac of random DIRECTIONS multimodal (oblique)")
print(f"\n  proper JOINT measures:  best-direction dip (discriminant axis) multimodal (p<0.05): "
      f"{int((df.bestdir_p<0.05).sum())}/{len(df)}   median gmm_k (uncapped 1-15) = {df.gmm_k.median():.0f}")
print(f"  by corpus best-direction-multimodal frac: " + str({c: round((g.bestdir_p < 0.05).mean(), 2) for c, g in df.groupby('corpus')}))
print(f"  marginally multimodal (dip_frac>0.5): {int((df.dip_frac>0.5).sum())}/{len(df)}")
print(f"  gmm_k distribution (uncapped): {df.gmm_k.value_counts().sort_index().to_dict()}")
print("saved streamline/STREAM_MULTIMODALITY.csv")
