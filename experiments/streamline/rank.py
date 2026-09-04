# -*- coding: utf-8 -*-
"""Ranking comparison on the 176-dataset TWO-STAGE streamlined benchmark. Single fit/dataset computes
UOMS criteria (em/mv/consensus/model_centrality/hits) + our synthetic (beta=1, matched) + ground-truth
ap_norm on the two-stage-hardened test. Leaderboard: regret micro|macro, vs EM/MV/random."""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from dev_common import TAB_POOL
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
from hadb_round2_common import consensus_scores, model_centrality_scores, hits_authority_scores, _em_auc, _mv_auc
LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")
Q = 0.05; NGEN = 5000; TAU = -0.14
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
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b] + 1e-9)))
    return sev
def harden(Xtr, Xhold, Xa):
    so_h, so_a = severity(Xtr, Xhold), severity(Xtr, Xa); tho = np.quantile(so_h, 1 - Q)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ptr, Ph, Pa = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xa))
        sp_h, sp_a = severity(Ptr, Ph), severity(Ptr, Pa); thp = np.quantile(sp_h, 1 - Q)
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
f = pd.read_csv(os.path.join(D, "STREAM_FINAL2_SET.csv")); rows = []
t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
for i, (_, r) in enumerate(f.iterrows()):
    try: Xtr, Xval, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xval) < 20 or len(Xtn) < 10 or len(Xa) < 5: continue
    Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    rng = np.random.default_rng(0); lo, hi = Xval.min(0), Xval.max(0); U = lo + rng.random((NGEN, Xval.shape[1])) * np.where(hi > lo, hi - lo, 1.0)
    syn1, syn4 = gen_beta(Xval, 200, 1.0), gen_beta(Xval, 200, -4.0)
    names, V, T, ap, a1, a4, emv, mvv = [], [], [], {}, {}, {}, {}, {}
    ye1 = np.r_[np.zeros(len(Xval)), np.ones(len(syn1))]
    for vn, ct in TAB_POOL:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
            sv = np.asarray(m.decision_function(Xval), float); st = np.asarray(m.decision_function(Xte), float); su = np.asarray(m.decision_function(U), float)
            s1 = np.asarray(m.decision_function(syn1), float); s4 = np.asarray(m.decision_function(syn4), float)
            if not (np.all(np.isfinite(sv)) and np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12): continue
            names.append(vn); V.append(sv); T.append(st)
            ap[vn] = (average_precision_score(yte, st) - base) / (1 - base + 1e-12)
            a1[vn] = roc_auc_score(ye1, np.r_[sv, s1]); a4[vn] = roc_auc_score(np.r_[np.zeros(len(Xval)), np.ones(len(syn4))], np.r_[sv, s4])
            su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
            emv[vn] = _em_auc(t, 1.0, -su, -sv, NGEN); mvv[vn] = _mv_auc(alpha, 1.0, -su, -sv, NGEN)
        except Exception: pass
    if len(names) < 6: continue
    Vm = np.column_stack(V); cons = dict(zip(names, consensus_scores(Vm))); mc = dict(zip(names, model_centrality_scores(Vm))); ht = dict(zip(names, hits_authority_scores(Vm)))
    av = pd.Series(ap); best = av.max(); loc = [av[v] for v in names if fam(v) == "local"]; glo = [av[v] for v in names if fam(v) == "global"]
    p1 = max(names, key=lambda v: a1[v]); pe = max(names, key=lambda v: a4[v]); lev = max(a1[v] for v in names) - max(a4[v] for v in names)
    pk = {"beta1": p1, "matched": p1 if lev > TAU else pe, "em": max(names, key=lambda v: emv[v]), "mv": min(names, key=lambda v: mvv[v]),
          "consensus": max(names, key=lambda v: cons[v]), "model_centrality": max(names, key=lambda v: mc[v]), "hits": max(names, key=lambda v: ht[v])}
    rec = {"corpus": r.corpus, "dataset": r.dataset, "truefam": "local" if (loc and (not glo or max(loc) > max(glo))) else "global", "reg_random": best - av.mean()}
    for k, v in pk.items(): rec["reg_" + k] = best - av[v]
    rows.append(rec)
    if i % 30 == 0: print(f"  ..{i}/{len(f)} {len(rows)} kept", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_RANK.csv"), index=False)
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
e, mv = df.reg_em.values, df.reg_mv.values
print(f"\n=== RANKING leaderboard on streamlined benchmark ({len(df)}: {int((df.truefam=='local').sum())} local / {int((df.truefam=='global').sum())} global) ===")
print(f"  {'method':22s} {'micro':>6s} {'macro':>6s} {'vsEM':>7s} {'vsMV':>7s}")
for nm, c in [("NoMaS matched (ours)", "reg_matched"), ("NoMaS beta=1 (ours)", "reg_beta1"), ("EM", "reg_em"), ("MV", "reg_mv"),
              ("consensus/UDR", "reg_consensus"), ("ModelCentrality", "reg_model_centrality"), ("HITS", "reg_hits"), ("random", "reg_random")]:
    a = df[c].values
    pe = wilcoxon(a, e).pvalue if c != "reg_em" and (a != e).any() else np.nan
    pm = wilcoxon(a, mv).pvalue if c != "reg_mv" and (a != mv).any() else np.nan
    print(f"  {nm:22s} {a.mean():6.3f} {macro(c):6.3f} {pe:7.3f} {pm:7.3f}")
print("saved streamline/STREAM_RANK.csv")
