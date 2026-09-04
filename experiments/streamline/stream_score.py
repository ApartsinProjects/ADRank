# -*- coding: utf-8 -*-
"""SIDE PROJECT payoff (part 1): score detectors on the 184-dataset streamlined benchmark's HARDENED
sets. Verify SOLVABILITY (oracle ap_norm > random), FAMILY mix, and our methods (beta=1, matched)
vs random/oracle. Detector-free UOMS leaderboard is a later step. Writes to streamline/ only."""
import os, sys, io, zipfile, re, contextlib, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as P
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05; MAX_MTS = 200
_ZIP = {}
def zf(p):
    if p not in _ZIP: _ZIP[p] = zipfile.ZipFile(os.path.join(ROOT, "data", p))
    return _ZIP[p]
# --- build name->(X,y) maps for adbench/dami and MTS ---
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in P.load_npz_dir(dd): OBJ[("%s" % sub, str(ds.name)[:40])] = (np.asarray(ds.X, float), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(MAX_MTS): MTS[str(name)[:40]] = (Xc, lab)


def split3_norm(Xn_all, rng):
    idx = np.arange(len(Xn_all)); rng.shuffle(idx)
    a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
    return Xn_all[idx[:a]], Xn_all[idx[a:b]], Xn_all[idx[b:]]


def load_any(corpus, name):
    if corpus in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
        y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    elif corpus in ("adbench", "dami"):
        X, y = OBJ[(corpus, name)]; X = np.nan_to_num(X)
    elif corpus == "tsbad_m":
        Xc, lab = MTS[name]; Xw, starts = mts_window_features(Xc); yw = mts_wlabels(lab, starts)
        return ("ts", np.nan_to_num(Xw), yw)
    else:
        raise ValueError(corpus)
    return ("tab", X, y.astype(int))


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


final = pd.read_csv(os.path.join(OUT, "STREAM_FINAL_SET.csv")); final = final[final.n_eff >= 100]
rows = []
for _, r in final.iterrows():
    try:
        kind, X, y = load_any(r.corpus, r.dataset)
        if kind == "tab":
            Xn_all = X[y == 0]; Xa = X[y == 1]
            if len(Xn_all) > 6000: Xn_all = Xn_all[np.random.default_rng(0).choice(len(Xn_all), 6000, replace=False)]
            Xtr, Xval, Xtn = split3_norm(Xn_all, np.random.default_rng(0))
        else:
            Xw, yw = X, y; pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
            Xtr = Xw[tr][yw[tr] == 0]; Xval = Xw[va][yw[va] == 0]; Xtn = Xw[te][yw[te] == 0]; Xa = Xw[yw == 1]
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 20 or len(Xtn) < 10 or len(Xa) < 5: continue
    thr = np.quantile(severity(Xtr, np.vstack([Xval, Xtn])), 1 - Q); Xh = Xa[severity(Xtr, Xa) <= thr]
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    s1 = gen_beta(Xval, 200, 1.0); s4 = gen_beta(Xval, 200, -4.0)
    ap, a1, a4 = {}, {}, {}
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            st = np.asarray(m.decision_function(Xte), float); sv = np.asarray(m.decision_function(Xval), float)
            ss1 = np.asarray(m.decision_function(s1), float); ss4 = np.asarray(m.decision_function(s4), float)
            if not (np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            a1[vn] = roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(len(s1))], np.r_[sv, ss1])
            a4[vn] = roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(len(s4))], np.r_[sv, ss4])
        except Exception: pass
    common = [v for v in ap if v in a1 and v in a4]
    if len(common) < 6: continue
    av = pd.Series({v: ap[v] for v in common}); best = av.max()
    loc = [av[v] for v in common if fam(v) == "local"]; glo = [av[v] for v in common if fam(v) == "global"]
    p1 = max(common, key=lambda v: a1[v]); pe = max(common, key=lambda v: a4[v])
    lev = max(a1[v] for v in common) - max(a4[v] for v in common); pm = p1 if lev > -0.14 else pe
    rows.append({"corpus": r.corpus, "dataset": r.dataset, "n_hard": int(len(Xh)), "base_rate": round(base, 3),
                 "oracle_apnorm": round(best, 3), "reg_beta1": best - av[p1], "reg_matched": best - av[pm],
                 "reg_random": best - av.mean(),
                 "truefam": "local" if (loc and (not glo or max(loc) > max(glo))) else "global"})
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "STREAM_SCORE.csv"), index=False)
print(f"=== streamlined benchmark scoring ({len(df)} datasets) ===")
print(f"  SOLVABILITY (oracle ap_norm on hardened test):")
print(f"    median {df.oracle_apnorm.median():.3f}  mean {df.oracle_apnorm.mean():.3f}  "
      f">0.1: {int((df.oracle_apnorm>0.1).mean()*100)}%  <=0.02 (unsolvable): {int((df.oracle_apnorm<=0.02).sum())}")
print(f"  FAMILY mix: local {int((df.truefam=='local').sum())} / global {int((df.truefam=='global').sum())}   "
      f"median base_rate {df.base_rate.median():.2f}")
def macro(c): return np.mean([df[df.truefam == f][c].mean() for f in ["local", "global"] if (df.truefam == f).any()])
print(f"  our methods (regret, micro | macro):")
for nm, c in [("beta1", "reg_beta1"), ("matched", "reg_matched"), ("random", "reg_random")]:
    print(f"    {nm:8s} {df[c].mean():.3f} | {macro(c):.3f}")
print("saved streamline/STREAM_SCORE.csv")
