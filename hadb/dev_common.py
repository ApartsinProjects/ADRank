# -*- coding: utf-8 -*-
"""Shared dev helpers (importable, no execution): validation-feature reconstruction for the
dev datasets, the detector pools, and the dev-set sampler. Used by the selection-idea dev
harnesses so the reconstruction logic lives in exactly one place.
"""
import os, sys, zipfile, re
import numpy as np
import pandas as pd

ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, POOL as _TS_POOL  # noqa: E402
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
TS_POOL = _TS_POOL()


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


def val_ucr_raw(name):
    """Return the raw 1-D series and labels (for time-shuffling synthetic generation)."""
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "ucr", "UCR_TimeSeriesAnomalyDatasets2021.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name]
    if not cand:
        cand = [n for n in z.namelist() if n.lower().endswith(".txt") and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]; m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn, re.I)
    x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()], float)
    a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    return x, lab


def val_tsbad_u(name):
    """Validation-normal window features for a TSB-AD-U univariate series (CSV: value,Label)."""
    import io as _io
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "tsbad", "TSB-AD-U.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name]
    if not cand:
        cand = [n for n in z.namelist() if n.lower().endswith(".csv") and name.split("_")[0] in n]
    df = pd.read_csv(_io.BytesIO(z.read(cand[0])))
    x = np.nan_to_num(df.iloc[:, 0].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    lab = df.iloc[:, 1].to_numpy(int)
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2
        lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    starts = np.arange(0, len(x) - W + 1, STRIDE)
    Xw, _ = _window_features(x, w=W, stride=STRIDE); yw = _window_labels(lab, starts, w=W, min_count=1)
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    nm = np.where(yw == 0)[0]
    g = np.random.default_rng(0); idx = np.arange(len(nm)); g.shuffle(idx)
    return Xw[nm[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]]


def sample_holdout(corpus, n, skip=None):
    """Datasets NOT used in the dev sampler (fresh confirmation set)."""
    inc = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
    inc = inc[inc.include & (inc.corpus == corpus)].sort_values("spread_ap_norm")
    names = list(inc.dataset)
    dev = set(sample_dev(corpus, 10)) if skip == "dev" else set()
    fresh = [x for x in names if x not in dev]
    return list(np.array(fresh)[np.linspace(0, len(fresh) - 1, min(n, len(fresh))).astype(int)])


def sample_dev(corpus, n=10):
    inc = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
    inc = inc[inc.include & (inc.corpus == corpus)].sort_values("spread_ap_norm")
    return list(inc.dataset.iloc[np.linspace(0, len(inc) - 1, n).astype(int)])


def true_apnorm(csv, name):
    D = pd.read_csv(os.path.join(S, csv)); D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
    return D.groupby(["dataset", "variant"]).ap_norm.mean().loc[name], \
        D.groupby(["dataset", "variant"]).em.mean().loc[name]
