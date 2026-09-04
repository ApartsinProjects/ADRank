# -*- coding: utf-8 -*-
"""Faithful UDR (Unsupervised Disentanglement Ranking, Ma et al) on the STOCHASTIC sub-pool where it
is defined: iForest configs x seeds; rank each config by cross-seed ranking STABILITY (median pairwise
Spearman of test scores), pick the most stable, its ap_norm = mean over its seeds. Regret vs the full-
pool oracle. (Full-pool UDR is ill-defined: deterministic detectors have perfect stability.)"""
import os, sys, io, contextlib, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from pyod.models.iforest import IForest
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import W, STRIDE, block_split3
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
Q = 0.05; NEST = [50, 100, 200]; MAXF = [0.5, 0.75, 1.0]; SEEDS = [0, 1, 2]
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
def get3(corp, name):
    if corp in ("oddbench", "ovrbench"):
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); Xn, Xa = X[y == 0], X[y == 1]
    elif corp in ("adbench", "dami"):
        X, y = OBJ[(corp, name)]; Xn, Xa = X[y == 0], X[y == 1]
    else:
        Xc, lab = MTS[name]; Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
        return Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[yw == 1]
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
    return Xn[idx[:a]], Xn[idx[b:]], Xa
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv")); sc = pd.read_csv(os.path.join(D, "STREAM_SCORE2.csv"))[["corpus", "dataset", "oracle_apnorm"]]
f = rk.merge(sc, on=["corpus", "dataset"], how="left"); rows = []
for i, (_, r) in enumerate(f.iterrows()):
    try: Xtr, Xtn, Xa = get3(r.corpus, r.dataset)
    except Exception: continue
    if len(Xtr) < 40 or len(Xtn) < 10 or len(Xa) < 5 or r.oracle_apnorm != r.oracle_apnorm: continue
    Xh = harden(Xtr, Xtn, Xa)
    if len(Xh) < 20: continue
    Xte = np.vstack([Xtn, Xh]); yte = np.r_[np.zeros(len(Xtn)), np.ones(len(Xh))]; base = yte.mean()
    stab, apc = {}, {}
    for ne in NEST:
        for mf in MAXF:
            svs, aps = [], []
            for sd in SEEDS:
                try:
                    m = IForest(n_estimators=ne, max_features=mf, random_state=sd)
                    with contextlib.redirect_stdout(io.StringIO()): m.fit(Xtr)
                    st = np.asarray(m.decision_function(Xte), float)
                    if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12: svs.append(st); aps.append((average_precision_score(yte, st) - base) / (1 - base + 1e-12))
                except Exception: pass
            if len(svs) >= 2:
                ps = [spearmanr(svs[a], svs[b]).statistic for a in range(len(svs)) for b in range(a + 1, len(svs))]
                stab[(ne, mf)] = np.median(ps); apc[(ne, mf)] = np.mean(aps)
    if len(stab) < 4: continue
    pick = max(stab, key=stab.get)  # UDR: most seed-stable config
    rows.append({"corpus": r.corpus, "dataset": r.dataset, "reg_udr": r.oracle_apnorm - apc[pick],
                 "reg_matched": r.reg_matched, "reg_em": r.reg_em, "reg_random": r.reg_random, "truefam": r.truefam})
    if i % 30 == 0: print(f"  ..{i}/{len(f)}", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_UDR.csv"), index=False)
def macro(c): return np.mean([df[df.truefam == fa][c].mean() for fa in ["local", "global"] if (df.truefam == fa).any()])
print(f"\n=== UDR (iForest sub-pool, seed-stability) on streamlined benchmark ({len(df)} datasets) ===")
u = df.reg_udr.values
for nm, c in [("UDR (seed-stability)", "reg_udr"), ("NoMaS matched (ours)", "reg_matched"), ("EM", "reg_em"), ("random", "reg_random")]:
    a = df[c].values; p = wilcoxon(a, u).pvalue if c != "reg_udr" and (a != u).any() else np.nan
    print(f"  {nm:22s} micro {a.mean():.3f}  macro {macro(c):.3f}  vs UDR p={p if p==p else float('nan'):.3f}")
print(f"  (UDR on FULL mixed pool is ill-defined: deterministic detectors have perfect seed-stability)")
print("saved streamline/STREAM_UDR.csv")
