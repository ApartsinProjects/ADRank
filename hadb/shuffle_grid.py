# -*- coding: utf-8 -*-
"""Parameterised feature-shuffle synthetic-anomaly generator + cluster filter, evaluated by the
within-dataset Spearman(synth_AUC, true) on the dev set. A small principled grid only (not a
big sweep - that would overfit the 20 dev datasets); the winner is confirmed on a fresh holdout.

Generator knobs (user spec):
  n_replace   how many features to overwrite: 'few', 'many', or 'sample' (random per synthetic)
  source      'single' = all replaced features from ONE donor point; 'perfeat' = each replaced
              feature from a DIFFERENT donor
  alpha       blend weight new = (1-a)*old + a*donor; 1.0 = hard replace, 'sample' = random in
              [0.3,1.0] per synthetic (continuous anomaly strength)
  filt        'none' or 'cluster' (keep synthetics away from cluster cores: nearest-centre
              distance above the 60th percentile of the normals -> between/outside regions)
"""
import os, sys, io, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import val_tabular, val_ucr, TAB_POOL, TS_POOL, sample_dev, true_apnorm


def gen_shuffle(Xn, n_synth, n_replace="sample", source="single", alpha=1.0, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape
    out = np.empty((n_synth, d))
    for r in range(n_synth):
        base = Xn[rng.integers(n)].copy()
        if n_replace == "few":
            k = max(1, int(0.15 * d))
        elif n_replace == "many":
            k = max(1, int(0.5 * d))
        else:
            k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        cols = rng.choice(d, k, replace=False)
        if source == "single":
            vals = Xn[rng.integers(n)][cols]
        else:
            vals = np.array([Xn[rng.integers(n), c] for c in cols])
        a = alpha if isinstance(alpha, (int, float)) else float(rng.uniform(0.3, 1.0))
        base[cols] = (1 - a) * base[cols] + a * vals
        out[r] = base
    return out


def cluster_filter(Xn, synth, band=60):
    sc = StandardScaler().fit(Xn); Zn = sc.transform(Xn); Zs = sc.transform(synth)
    if Zn.shape[1] > 16:
        p = PCA(n_components=16, random_state=0).fit(Zn); Zn, Zs = p.transform(Zn), p.transform(Zs)
    cen = MiniBatchKMeans(n_clusters=min(20, max(2, len(Zn) // 30)), random_state=0, n_init=5).fit(Zn).cluster_centers_
    dn = np.linalg.norm(Zn[:, None] - cen[None], axis=2).min(1)
    ds = np.linalg.norm(Zs[:, None] - cen[None], axis=2).min(1)
    thr = np.percentile(dn, band)
    keep = synth[ds > thr]
    return keep if len(keep) >= 10 else synth


def synth_auc(Xval, synth, pool, seed=0):
    g = np.random.default_rng(seed); idx = np.arange(len(Xval)); g.shuffle(idx)
    cut = int(0.7 * len(idx)); tr, ho = idx[:cut], idx[cut:]
    if len(tr) < 30 or len(ho) < 10 or len(synth) < 10:
        return {}
    Xtr, Xev = Xval[tr], np.vstack([Xval[ho], synth]); yev = np.r_[np.zeros(len(ho)), np.ones(len(synth))]
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


CONFIGS = [
    ("single_hard_nofilt", dict(n_replace="sample", source="single", alpha=1.0, filt="none")),
    ("perfeat_hard_nofilt", dict(n_replace="sample", source="perfeat", alpha=1.0, filt="none")),
    ("single_blend_nofilt", dict(n_replace="sample", source="single", alpha="sample", filt="none")),
    ("single_hard_cluster", dict(n_replace="sample", source="single", alpha=1.0, filt="cluster")),
    ("few_single_hard", dict(n_replace="few", source="single", alpha=1.0, filt="none")),
    ("many_perfeat_hard", dict(n_replace="many", source="perfeat", alpha=1.0, filt="none")),
]

DEV = [("oddbench", "hadb_oddbench.csv", val_tabular, TAB_POOL, sample_dev("oddbench", 10)),
       ("ucr", "hadb_ts_ucr.csv", None, TS_POOL, sample_dev("ucr", 10))]

rho = {c[0]: [] for c in CONFIGS}
for corpus, csv, loader, pool, names in DEV:
    for name in names:
        try:
            Xval = val_ucr(name) if corpus == "ucr" else loader(corpus, name)
        except Exception:
            continue
        if len(Xval) < 80:
            continue
        ap, _ = true_apnorm(csv, name)
        for cname, cfg in CONFIGS:
            synth = gen_shuffle(Xval, 300, cfg["n_replace"], cfg["source"], cfg["alpha"])
            if cfg["filt"] == "cluster":
                synth = cluster_filter(Xval, synth)
            synth = synth[:150]
            sa = synth_auc(Xval, synth, pool)
            common = [v for v in sa if v in ap.index]
            if len(common) < 5:
                continue
            sv = np.array([sa[v] for v in common]); tv = np.array([ap[v] for v in common])
            if np.std(sv) > 0 and np.std(tv) > 0:
                rho[cname].append(spearmanr(sv, tv).statistic)

print(f"=== shuffle-generator grid: within-dataset Spearman(synth_AUC, true), dev 20 ===")
print(f"  {'config':22s} {'mean':>7s} {'median':>7s}  >0")
for cname, _ in CONFIGS:
    a = np.array(rho[cname])
    print(f"  {cname:22s} {np.nanmean(a):+.3f}  {np.nanmedian(a):+.3f}   {int((a>0).sum())}/{len(a)}")
print(f"\n  baselines: unfiltered shuffle +0.176 | EM +0.133 | PCA-between +0.102")
