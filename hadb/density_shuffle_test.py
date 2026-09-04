# -*- coding: utf-8 -*-
"""Density-reweighted marginal resampling (shuffle + extreme-value injection) vs EM.

For each modified feature, replace its value with a sample from the feature's empirical marginal
reweighted by density^beta:
  beta = +1  -> sample proportional to density = a TYPICAL value = shuffle (multivariate break)
  beta < 0   -> sample proportional to 1/density^|beta| = a LOW-PROBABILITY value (tails AND
               bimodal valleys, distribution-agnostic)
Non-triviality: only bins inside the [ (100-cap), cap ] value-percentile band are eligible, so
injected values are rare-but-not-trivial (never beyond the triviality zone).

Configs compared (corrected protocol: fit detectors on TRAIN, score VALIDATION normals vs synth):
  shuffle      existing single-donor shuffle (control)
  extreme      beta<0 density resample on a feature subset (pure low-prob injection)
  mixed        half the modified features shuffled (beta=+1), half extreme (beta<0)
Bar: mixed/extreme must LIFT global-best datasets (where shuffle alone loses) WITHOUT giving back
the local-best win.
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


def _hists(Xn, B=30, cap=98):
    H = []
    for j in range(Xn.shape[1]):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=B); dens = cnt / max(cnt.sum(), 1)
        lo, hi = np.percentile(col, 100 - cap), np.percentile(col, cap)
        centers = (edges[:-1] + edges[1:]) / 2
        allowed = (cnt > 0) & (centers >= lo) & (centers <= hi)
        H.append((edges, dens, allowed))
    return H


def gen(Xn, ns, mode="shuffle", beta=-2.0, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape
    H = _hists(Xn) if mode != "shuffle" else None
    out = np.empty((ns, d))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy(); k = max(1, int(frac * d)); cols = rng.choice(d, k, replace=False)
        for i, j in enumerate(cols):
            use_extreme = (mode == "extreme") or (mode == "mixed" and i % 2 == 0)
            if mode == "shuffle" or not use_extreme:
                base[j] = Xn[rng.integers(n), j]                      # shuffle: donor value
            else:
                edges, dens, allowed = H[j]
                w = np.where(allowed, np.maximum(dens, 1e-6) ** beta, 0.0)
                if w.sum() == 0:
                    w = allowed.astype(float)
                if w.sum() == 0:
                    base[j] = Xn[rng.integers(n), j]; continue
                b = rng.choice(len(w), p=w / w.sum()); base[j] = rng.uniform(edges[b], edges[b + 1])
        out[r] = base
    return out


def sauc(Xtr, Xval, syn, pool):
    Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; o = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12: o[vn] = float(roc_auc_score(ye, s))
        except Exception: pass
    return o


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include].copy(); inc["tbf"] = inc.gt_best.map(fam)
samp = pd.concat([g.sort_values("spread_ap_norm").iloc[np.linspace(0, len(g) - 1, min(12, len(g))).astype(int)] for _, g in inc.groupby("corpus")])
MODES = ["shuffle", "extreme", "mixed"]
reg = {m: [] for m in MODES}; reg["em"] = []; perfam = {m: {"local": [], "global": [], "other": []} for m in MODES}
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
    common0 = [v for v, _ in pool if v in ap.index and v in emv.index and emv[v] == emv[v]]
    if len(common0) < 6: continue
    av = pd.Series({v: ap[v] for v in common0}); best = av.max(); ev = pd.Series({v: emv[v] for v in common0})
    reg["em"].append(best - av[ev.idxmax()])
    for mode in MODES:
        sa = sauc(Xtr, Xval, gen(Xval, 150, mode=mode), pool); common = [v for v in sa if v in av.index]
        if len(common) < 6: reg[mode].append(np.nan); continue
        sv = pd.Series({v: sa[v] for v in common}); rr = best - av[sv.idxmax()]
        reg[mode].append(rr); perfam[mode][r.tbf].append((rr, best - av[ev.idxmax()]))
print(f"=== density-shuffle vs EM on finalized benchmark ({len(reg['em'])} datasets) ===")
print(f"  mean regret:  EM {np.nanmean(reg['em']):.4f}")
for m in MODES:
    a = np.array(reg[m]); e = np.array(reg["em"]); ok = ~np.isnan(a)
    p = wilcoxon(a[ok], e[ok]).pvalue if ok.sum() > 5 and (a[ok] != e[ok]).any() else np.nan
    print(f"    {m:8s} {np.nanmean(a):.4f}  (vs EM p={p:.3f})")
print("  by true-best family (mode regret / EM regret / mode-wins):")
for m in MODES:
    for f in ["local", "global", "other"]:
        pf = np.array(perfam[m][f])
        if len(pf): print(f"    {m:8s} {f:7s} n={len(pf):2d}  synth {pf[:,0].mean():.3f}  EM {pf[:,1].mean():.3f}  wins {int((pf[:,0]<pf[:,1]).sum())}/{len(pf)}")
