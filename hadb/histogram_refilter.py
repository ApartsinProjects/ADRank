# -*- coding: utf-8 -*-
"""Gaussian vs histogram triviality filtering, resolved empirically.

The max|z| rule assumes Gaussian marginals. A HISTOGRAM rule (per-feature empirical log-density
= HBOS) catches skewed/bimodal-rare values that z misses. Compute the histogram-rule test AUC
for every max|z|-cleaned tabular dataset, and re-filter at AUC > 0.85: this removes datasets
solvable by ANY per-feature marginal rule, leaving only genuinely MULTIVARIATE-hard anomalies.
Report the composition shift (local vs global) - the key question is whether this collapses the
global-win datasets, making it a local-only benchmark.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
CUT = 0.85
GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA"); LOCAL = ("LOF", "KNN", "CBLOF")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")


def find_npz(corpus, name):
    for sub in {"oddbench": ["oddbench"], "ovrbench": ["ovrbench"], "adbench_dami": ["adbench", "dami"]}.get(corpus, [corpus]):
        p = os.path.join(ROOT, "data", sub, name + ".npz")
        if os.path.exists(p):
            return p
    return None


def rule_aucs(corpus, name, nbins=20):
    p = find_npz(corpus, name)
    if p is None:
        return np.nan, np.nan
    d = np.load(p, allow_pickle=True)
    if "X" in d:
        X = np.asarray(d["X"], float); y = np.asarray(d["y"]).ravel().astype(int)
    else:
        X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 6000:
        r = np.random.RandomState(0); k = r.choice(len(X), 6000, replace=False); X, y = X[k], y[k]
    nm = y == 0; an = np.where(y == 1)[0]
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9
    hard = an[np.abs((X[an] - mu) / sd).max(1) <= np.percentile(np.abs((X[nm] - mu) / sd).max(1), 99)]
    if len(hard) < 5:
        return np.nan, np.nan
    ni = np.where(nm)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx); c2 = int(0.8 * len(idx))
    tr = ni[idx[:c2]]; tn = ni[idx[c2:]]
    Xtr = X[tr]; Xe = np.vstack([X[tn], X[hard]]); ye = np.r_[np.zeros(len(tn)), np.ones(len(hard))]
    # max|z| rule
    mt, st = Xtr.mean(0), Xtr.std(0) + 1e-9
    mz = roc_auc_score(ye, np.abs((Xe - mt) / st).max(1))
    # histogram rule = HBOS: sum of per-feature -log(train-normal bin density)
    hb = np.zeros(len(Xe))
    for j in range(Xtr.shape[1]):
        cnt, edges = np.histogram(Xtr[:, j], bins=nbins)
        dens = cnt / cnt.sum() + 1e-6
        b = np.clip(np.digitize(Xe[:, j], edges) - 1, 0, nbins - 1)
        hb += -np.log(dens[b])
    hbauc = roc_auc_score(ye, hb)
    return float(mz), float(hbauc)


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST_REFILTERED.csv"))
tab = M[M.include_v2 & (M.modality == "tabular")].copy()
print(f"computing histogram-rule AUC for {len(tab)} max|z|-cleaned tabular datasets ...")
hb = {}
for _, r in tab.iterrows():
    _, h = rule_aucs(r.corpus, r.dataset)
    hb[r.dataset] = h
M["hbos_rule_auc"] = M.dataset.map(hb)
M["include_v3"] = M.include_v2 & ~((M.modality == "tabular") & (M.hbos_rule_auc > CUT))
M.to_csv(os.path.join(S, "HADB_MANIFEST_HISTFILT.csv"), index=False)

v2 = M[M.include_v2]; v3 = M[M.include_v3]
dropped = v2[~v2.dataset.isin(v3.dataset)]
v2["fam"] = v2.gt_best.map(fam); v3["fam"] = v3.gt_best.map(fam); dropped["fam"] = dropped.gt_best.map(fam)
print(f"\n=== histogram (HBOS-rule) re-filter at AUC > {CUT} ===")
print(f"  max|z|-cleaned (v2): {len(v2)}  ->  histogram-cleaned (v3): {len(v3)}   (dropped {len(dropped)} tabular)")
print(f"  dropped by true-best family: {dict(dropped.fam.value_counts())}")
print(f"\n  true-best family composition:")
print(f"    v2 (max|z|-clean):  {dict(v2[v2.modality=='tabular'].fam.value_counts())}")
print(f"    v3 (hist-clean):    {dict(v3[v3.modality=='tabular'].fam.value_counts())}")
print(f"\n  -> global-best tabular datasets: v2={int((v2[v2.modality=='tabular'].fam=='global').sum())}"
      f"  v3={int((v3[v3.modality=='tabular'].fam=='global').sum())}"
      f"  ({'COLLAPSES global -> local-only' if (v3[v3.modality=='tabular'].fam=='global').sum() < 5 else 'global datasets remain'})")
