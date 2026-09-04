# -*- coding: utf-8 -*-
"""RETRY pseudo-anomaly selection via clustering, on the streamlined benchmark. Cluster validation
normals; designate the smallest cluster (and separately, edge points) as PSEUDO-anomalies; select the
detector that best separates them. Compare to beta=1 synthetic (ours), random, oracle. Decisive metric:
within-dataset Spearman of criterion vs true ap_norm, and regret. (Failed on old benchmark, rho~-0.11.)"""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(200): MTS[str(name)[:40]] = (Xc, lab)
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        sev = np.maximum(sev, np.maximum(-np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))), -np.log(dens[b] + 1e-9)))
    return sev
def harden(Xtr, Xhold, Xa):
    tho = np.quantile(severity(Xtr, Xhold), 1 - Q); so_a = severity(Xtr, Xa)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        thp = np.quantile(severity(Ptr, Ph), 1 - Q); sp_a = severity(Ptr, Pa)
    except Exception:
        sp_a = np.zeros(len(Xa)); thp = np.inf
    return Xa[~((so_a > tho) | (sp_a > thp))]
def _H(Xn, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap); ctr = (edges[:-1] + edges[1:]) / 2
        H.append((edges, dens, (cnt > 0) & (ctr >= lo) & (ctr <= hi)))
    return H
def gen_beta(Xn, ns, beta, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; H = _H(Xn); out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1, int(frac * d)), replace=False):
            edges, dens, allowed = H[j]; w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
            base[j] = Xn[rng.integers(n), j] if w.sum() == 0 else rng.uniform(*edges[[(b := rng.choice(len(w), p=w / w.sum())), b + 1]])
        out[r] = base
    return out
def get3(corp, name):
    if corp in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); Xn, Xa = X[y == 0], X[y == 1]
    elif corp in ("adbench", "dami"):
        X, y = OBJ[(corp, name)]; Xn, Xa = X[y == 0], X[y == 1]
    else:
        Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
        return Xw[tr][yw[tr] == 0], Xw[va][yw[va] == 0], Xw[te][yw[te] == 0], Xw[yw == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
    return Xn[idx[:a]], Xn[idx[a:b]], Xn[idx[b:]], Xa
def pseudo_labels(Xval):
    """small-cluster pseudo-anomalies + edge pseudo-anomalies (validation normals)."""
    Z = StandardScaler().fit_transform(Xval); K = int(np.clip(len(Z) // 40, 5, 20))
    lab = KMeans(K, n_init=5, random_state=0).fit_predict(Z)
    sizes = pd.Series(lab).value_counts(); small = sizes.idxmin(); yc = (lab == small).astype(int)          # smallest cluster
    dc = np.linalg.norm(Z - Z.mean(0), axis=1); ye = (dc >= np.percentile(dc, 95)).astype(int)              # edge (farthest 5%)
    return yc, ye
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv")); rows = []
for i, (_, r) in enumerate(f.iterrows()):
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 60 or len(Xtn) < 10 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    yc, ye = pseudo_labels(Xval); syn1 = gen_beta(Xval, 200, 1.0); ye1 = np.r_[np.zeros(len(Xval)), np.ones(len(syn1))]
    ap, cSmall, cEdge, a1 = {}, {}, {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            sv = np.asarray(m.decision_function(Xval), float); st = np.asarray(m.decision_function(Xte), float); s1 = np.asarray(m.decision_function(syn1), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12 and np.nanstd(sv) > 1e-12): continue
            ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            cSmall[vn] = roc_auc_score(yc, sv) if yc.sum() > 0 else 0.5; cEdge[vn] = roc_auc_score(ye, sv) if ye.sum() > 0 else 0.5
            a1[vn] = roc_auc_score(ye1, np.r_[sv, s1])
        except Exception: pass
    common = [v for v in ap if v in cSmall and v in a1]
    if len(common) < 6: continue
    av = pd.Series({v: ap[v] for v in common}); best = av.max()
    def rho(cri):
        cc = [cri[v] for v in common]; return spearmanr(cc, av[common]).statistic if np.std(cc) > 0 else np.nan
    rows.append({"corpus": r.corpus, "dataset": r.dataset, "truefam": "local" if any(fam(v) == "local" and av[v] == best for v in common) else "global",
                 "reg_small": best - av[max(common, key=lambda v: cSmall[v])], "reg_edge": best - av[max(common, key=lambda v: cEdge[v])],
                 "reg_beta1": best - av[max(common, key=lambda v: a1[v])], "reg_random": best - av.mean(),
                 "rho_small": rho(cSmall), "rho_edge": rho(cEdge), "rho_beta1": rho(a1)})
    if i % 30 == 0: print(f"  ..{i}/{len(f)}", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_PSEUDO.csv"), index=False)
print(f"\n=== pseudo-anomaly clustering selectors on streamlined benchmark ({len(df)} datasets) ===")
print(f"  within-dataset Spearman(criterion, true ap_norm)  [>0 = useful signal]:")
print(f"    small-cluster {df.rho_small.mean():+.3f}   edge {df.rho_edge.mean():+.3f}   beta=1 (ours) {df.rho_beta1.mean():+.3f}")
print(f"  regret (micro):")
print(f"    small-cluster {df.reg_small.mean():.3f}   edge {df.reg_edge.mean():.3f}   beta=1 {df.reg_beta1.mean():.3f}   random {df.reg_random.mean():.3f}")
print("saved streamline/STREAM_PSEUDO.csv")
