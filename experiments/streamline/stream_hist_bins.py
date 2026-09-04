# -*- coding: utf-8 -*-
"""TEST (do not apply): histogram-BIN triviality rule vs edge-only rule. Define low-density normal
histogram bins as 'anomalous value' regions (INTERIOR gaps + edges), flag an anomaly if any feature
lands in one. Compare to an edge-only (ECDF two-sided tail) rule, both calibrated to 5% normal FP.
Measure the EXTRA anomalies the histogram (interior-gap) rule catches that the edge rule misses."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as P
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
Q = 0.05; BINS = 40
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

def edge_score(Xtr, X):    # max over features of ECDF two-sided tail rarity (edges/tails only)
    n, d = X.shape; s = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j]); m = len(tr)
        if tr[-1] - tr[0] < 1e-12: continue
        fb = np.searchsorted(tr, X[:, j], side="right") / m
        s = np.maximum(s, -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m))))
    return s
def hist_score(Xtr, X):    # max over features of -log(bin density): low-density bins incl INTERIOR gaps
    n, d = X.shape; s = np.zeros(n)
    for j in range(d):
        cnt, edges = np.histogram(Xtr[:, j], bins=BINS)
        if edges[-1] - edges[0] < 1e-12: continue
        dens = cnt / max(cnt.sum(), 1); b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, BINS - 1)
        s = np.maximum(s, -np.log(dens[b] + 1e-9))
    return s

f = pd.read_csv(os.path.join(D, "STREAM_FINAL_SET.csv")); f = f[f.n_eff >= 100]
rows = []
for _, r in f.iterrows():
    try:
        Xn_all, Xa = get_na(r.corpus, r.dataset); rng = np.random.default_rng(0)
        if len(Xn_all) > 6000: Xn_all = Xn_all[rng.choice(len(Xn_all), 6000, replace=False)]
        idx = np.arange(len(Xn_all)); rng.shuffle(idx); k = int(0.7 * len(idx)); Xtr, Xtn = Xn_all[idx[:k]], Xn_all[idx[k:]]
    except Exception: continue
    if len(Xtr) < 60 or len(Xtn) < 30 or len(Xa) < 20: continue
    e_n, e_a = edge_score(Xtr, Xtn), edge_score(Xtr, Xa); h_n, h_a = hist_score(Xtr, Xtn), hist_score(Xtr, Xa)
    te = np.quantile(e_n, 1 - Q); th = np.quantile(h_n, 1 - Q)          # both calibrated to 5% normal FP
    ce, ch = e_a > te, h_a > th
    rows.append({"corpus": r.corpus, "dataset": r.dataset, "edge_recall": round(float(ce.mean()), 3),
                 "hist_recall": round(float(ch.mean()), 3), "hist_extra": round(float((ch & ~ce).mean()), 3),
                 "fp_edge": round(float((e_n > te).mean()), 3), "fp_hist": round(float((h_n > th).mean()), 3)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "STREAM_HIST_BINS.csv"), index=False)
print(f"=== histogram-bin (interior-gap) vs edge-only triviality rule ({len(df)} datasets, both @5% FP) ===")
print(f"  {'corpus':10s} {'edge_recall':>11s} {'hist_recall':>11s} {'hist_EXTRA':>11s}")
for c in ["ovrbench", "oddbench", "tsbad_m", "adbench", "dami"]:
    g = df[df.corpus == c]
    if len(g): print(f"  {c:10s} {g.edge_recall.mean():11.2f} {g.hist_recall.mean():11.2f} {g.hist_extra.mean():11.2f}")
print(f"  {'ALL':10s} {df.edge_recall.mean():11.2f} {df.hist_recall.mean():11.2f} {df.hist_extra.mean():11.2f}")
print(f"\n  hist_extra = anomalies caught by histogram-gap rule but MISSED by edge rule (interior-gap trivial anomalies)")
print(f"  realized normal FP: edge {df.fp_edge.mean():.3f}  hist {df.fp_hist.mean():.3f}  (target {Q})")
print(f"  datasets where histogram rule catches notably more (hist_recall > edge_recall+0.1): {int((df.hist_recall>df.edge_recall+0.1).sum())}/{len(df)}")
print("saved streamline/STREAM_HIST_BINS.csv")
