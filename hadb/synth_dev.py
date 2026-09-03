# -*- coding: utf-8 -*-
"""Cheap SYNTHETIC between-cluster anomaly test - the make-or-break mechanism check before
investing in VAE/VaDE.

Generate synthetic anomalies BETWEEN clusters in a PCA latent, density-filtered to the
inside/between band the geometry EDA found (not the extreme outer edge), map back to feature
space, and ask each detector to separate them from held-out normals. The DECISIVE metric is
the WITHIN-DATASET Spearman correlation between a detector's synthetic-detection AUC and its
TRUE test ap_norm - the exact check that killed the pseudo-anomaly approach (corr ~= 0).

BAR: mean within-dataset Spearman > ~0.15 => synthetic anomalies carry real signal, build
VAE/VaDE for better generation. <= 0 => the direction is dead like pseudo-anomalies, and no
generator quality will save it.
"""
import os, sys, io, zipfile, re, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, POOL as TS_POOL  # noqa: E402
from adrank.ts import _window_features, _window_labels  # noqa: E402
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA


def build_tab_pool():
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
         ("LOF", lambda: LOF(n_neighbors=20)), ("KNN", lambda: KNN(n_neighbors=5)),
         ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()), ("HBOS", lambda: HBOS(n_bins=10)),
         ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10))]
    for k in [3, 10, 35, 50, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50, 100]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9, 0.99]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ncl in [4, 16, 32]:
        p.append((f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=0)))
    for ne in [50, 300]:
        p.append((f"IF_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=0)))
    for nb in [20, 50]:
        p.append((f"LODA_b{nb}", lambda nb=nb: LODA(n_bins=nb)))
    return p


TAB_POOL = build_tab_pool()


def val_tabular(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(),
                        np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    nm = np.where(y == 0)[0]
    g = np.random.default_rng(0); idx = np.arange(len(nm)); g.shuffle(idx)
    return X[nm[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]]


def val_ucr(name):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "ucr", "UCR_TimeSeriesAnomalyDatasets2021.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name]
    if not cand:
        cand = [n for n in z.namelist() if n.lower().endswith(".txt") and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]; m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn, re.I)
    x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()], float)
    a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2
        lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    starts = np.arange(0, len(x) - W + 1, STRIDE)
    Xw, _ = _window_features(x, w=W, stride=STRIDE); yw = _window_labels(lab, starts, w=W, min_count=1)
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    nm = np.where(yw == 0)[0]
    g = np.random.default_rng(0); idx = np.arange(len(nm)); g.shuffle(idx)
    return Xw[nm[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]]


def synth_between(Xval, n_synth=100, band=(60, 95), seed=0):
    """Synthetic anomalies sampled BETWEEN cluster centres in a PCA latent, density-filtered to
    the inside/between band, mapped back to feature space via inverse PCA."""
    sc = StandardScaler().fit(Xval); Zs = sc.transform(Xval)
    pca = PCA(n_components=min(16, Zs.shape[1]), random_state=0).fit(Zs); Z = pca.transform(Zs)
    K = min(20, max(3, len(Z) // 40))
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=5).fit(Z); cen = km.cluster_centers_
    nn = NearestNeighbors(n_neighbors=min(10, len(Z) - 1)).fit(Z)
    dref = nn.kneighbors(Z)[0][:, -1]
    lo, hi = np.percentile(dref, band)
    rng = np.random.default_rng(seed); cand = []
    for _ in range(n_synth * 12):
        i, j = rng.choice(K, 2, replace=False)
        t = rng.uniform(0.35, 0.65)
        p = t * cen[i] + (1 - t) * cen[j] + rng.normal(0, 0.15 * Z.std(0))
        d = nn.kneighbors(p[None])[0][0, -1]
        if lo <= d <= hi:
            cand.append(p)
        if len(cand) >= n_synth:
            break
    if len(cand) < 10:
        return None
    return sc.inverse_transform(pca.inverse_transform(np.array(cand)))


def synth_auc_per_detector(Xval, synth, pool, seed=0):
    g = np.random.default_rng(seed); idx = np.arange(len(Xval)); g.shuffle(idx)
    cut = int(0.7 * len(idx)); tr, ho = idx[:cut], idx[cut:]
    if len(tr) < 30 or len(ho) < 10:
        return {}
    Xtr = Xval[tr]; Xev = np.vstack([Xval[ho], synth]); yev = np.r_[np.zeros(len(ho)), np.ones(len(synth))]
    out = {}
    for vname, ctor in pool:
        try:
            m = ctor()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr)
            s = np.asarray(m.decision_function(Xev), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                out[vname] = float(roc_auc_score(yev, s))
        except Exception:
            continue
    return out


def sample_dev(corpus, n=10):
    inc = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
    inc = inc[inc.include & (inc.corpus == corpus)].sort_values("spread_ap_norm")
    return list(inc.dataset.iloc[np.linspace(0, len(inc) - 1, n).astype(int)])


DEV = [("oddbench", "hadb_oddbench.csv", val_tabular, TAB_POOL, sample_dev("oddbench", 10)),
       ("ucr", "hadb_ts_ucr.csv", None, TS_POOL(), sample_dev("ucr", 10))]

rho_synth, nrank_synth, rho_em = [], [], []
for corpus, csv, loader, pool, names in DEV:
    D = pd.read_csv(os.path.join(S, csv)); D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
    apn = D.groupby(["dataset", "variant"]).ap_norm.mean()
    emv = D.groupby(["dataset", "variant"]).em.mean()
    for name in names:
        try:
            Xval = val_ucr(name) if corpus == "ucr" else loader(corpus, name)
        except Exception:
            continue
        if len(Xval) < 60:
            continue
        synth = synth_between(Xval)
        if synth is None:
            print(f"  [no synth] {name[:34]}"); continue
        sa = synth_auc_per_detector(Xval, synth, pool)
        if len(sa) < 5:
            continue
        ap = apn.loc[name]
        common = [v for v in sa if v in ap.index]
        s_vec = np.array([sa[v] for v in common]); t_vec = np.array([ap[v] for v in common])
        em_vec = np.array([emv.loc[name].get(v, np.nan) for v in common])
        if len(common) >= 5 and np.std(s_vec) > 0 and np.std(t_vec) > 0:
            r = spearmanr(s_vec, t_vec).statistic
            rho_synth.append(r)
            tr = pd.Series(t_vec).rank(ascending=False)
            nrank_synth.append((tr.iloc[int(np.argmax(s_vec))] - 1) / (len(common) - 1))
            if np.std(em_vec[~np.isnan(em_vec)]) > 0:
                rho_em.append(spearmanr(em_vec, t_vec, nan_policy="omit").statistic)
            print(f"  {corpus:9s} {name[:32]:34s} rho_synth={r:+.3f}  synth_pts={len(synth)}")

print(f"\n=== SYNTHETIC between-cluster anomaly signal ({len(rho_synth)} datasets) ===")
print(f"  within-dataset Spearman(synth_AUC, true): mean {np.nanmean(rho_synth):+.3f}  "
      f"median {np.nanmedian(rho_synth):+.3f}  >0 on {int((np.array(rho_synth)>0).sum())}/{len(rho_synth)}")
print(f"  synth argmax pick true-rank percentile  : {np.nanmean(nrank_synth):.3f} (0=best,0.5=random)")
print(f"  (reference) within-dataset Spearman(EM, true): mean {np.nanmean(rho_em):+.3f}")
bar = np.nanmean(rho_synth)
print(f"\n  VERDICT: {'PROMISING - build VAE/VaDE' if bar > 0.15 else 'no real signal - synthetic direction is dead like pseudo-anomalies' if bar < 0.05 else 'WEAK - marginal, needs better generation to judge'}")
