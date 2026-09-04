# -*- coding: utf-8 -*-
"""Does synthetic close on EM once the max|z|-trivial datasets are removed?

Corrected protocol (fit detectors on TRAIN, score VALIDATION normals vs shuffle synth) on a
sample of the CLEANED benchmark (HADB_MANIFEST_REFILTERED include_v2). Compares shuffle regret
and within-dataset rho to EM, split by whether the true best is local or global.
"""
import os, sys, io, zipfile, re, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "adbench_dami": "hadb_v3.csv",
          "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv", "tsbad_m": "hadb_ts_mts.csv"}


def find_npz(corpus, name):
    for sub in {"oddbench": ["oddbench"], "ovrbench": ["ovrbench"], "adbench_dami": ["adbench", "dami"]}.get(corpus, [corpus]):
        p = os.path.join(ROOT, "data", sub, name + ".npz")
        if os.path.exists(p):
            return p
    return None


def tab_tv(corpus, name):
    d = np.load(find_npz(corpus, name), allow_pickle=True)
    if "X" in d:
        X = np.asarray(d["X"], float); y = np.asarray(d["y"]).ravel().astype(int)
    else:
        X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    ni = np.where(y == 0)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    return X[ni[idx[:int(0.6 * len(idx))]]], X[ni[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]]


def ts_tv(name, zipname):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", zipname))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]
    if fn.endswith(".csv"):
        import io as _io; df = pd.read_csv(_io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()]); a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    st = np.arange(0, len(x) - W + 1, STRIDE); Xw, _ = _window_features(x, w=W, stride=STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); return Xw[tr], Xw[va]


def gen(Xn, ns, seed=0):
    rng = np.random.default_rng(seed); n, dd = Xn.shape; out = np.empty((ns, dd))
    for r in range(ns):
        b = Xn[rng.integers(n)].copy(); k = int(rng.integers(1, max(2, int(0.6 * dd) + 1))); c = rng.choice(dd, k, replace=False); out[r] = b; out[r, c] = Xn[rng.integers(n)][c]
    return out


def sauc(Xtr, Xval, pool):
    syn = gen(Xval, 150); Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; o = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12: o[vn] = float(roc_auc_score(ye, s))
        except Exception: pass
    return o


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST_REFILTERED.csv")); inc = M[M.include_v2].copy(); inc["tbf"] = inc.gt_best.map(fam)
# sample ~ up to 12 per corpus
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(12, len(g))).astype(int)] for _, g in inc.groupby("corpus")])
reg = {"synth": [], "em": []}; rho = []; perfam = {"local": [], "global": [], "other": []}
for _, r in samp.iterrows():
    corpus = r.corpus
    try:
        if corpus in ("oddbench", "ovrbench", "adbench_dami"): Xtr, Xval = tab_tv(corpus, r.dataset); pool = TAB_POOL
        elif corpus == "tsbad_u": Xtr, Xval = ts_tv(r.dataset, "tsbad/TSB-AD-U.zip"); pool = TS_POOL
        elif corpus == "ucr": Xtr, Xval = ts_tv(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"); pool = TS_POOL
        else: continue
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 30: continue
    try: ap, emv = true_apnorm(CSVMAP[corpus], r.dataset)
    except Exception: continue
    sa = sauc(Xtr, Xval, pool); common = [v for v in sa if v in ap.index and v in emv.index and emv[v] == emv[v]]
    if len(common) < 6: continue
    av = pd.Series({v: ap[v] for v in common}); sv = pd.Series({v: sa[v] for v in common}); ev = pd.Series({v: emv[v] for v in common})
    best = av.max(); sr = best - av[sv.idxmax()]; er = best - av[ev.idxmax()]
    reg["synth"].append(sr); reg["em"].append(er); rho.append(spearmanr(sv, av).statistic); perfam[r.tbf].append((sr, er))
print(f"=== synthetic vs EM on CLEANED benchmark ({len(reg['em'])} datasets) ===")
print(f"  within-dataset rho(synth,true): mean {np.nanmean(rho):+.3f}")
print(f"  mean regret:  synth {np.nanmean(reg['synth']):.4f}   EM {np.nanmean(reg['em']):.4f}")
s, e = np.array(reg["synth"]), np.array(reg["em"])
if (s != e).any(): print(f"  paired Wilcoxon p={wilcoxon(s, e).pvalue:.4f}  synth {'<' if s.mean() < e.mean() else '>='} EM")
print("  by true-best family (synth regret / EM regret):")
for f in ["local", "global", "other"]:
    if perfam[f]:
        a = np.array(perfam[f]); print(f"    {f:7s} n={len(a):2d}  synth {a[:,0].mean():.3f}  EM {a[:,1].mean():.3f}  synth-wins {int((a[:,0]<a[:,1]).sum())}/{len(a)}")
