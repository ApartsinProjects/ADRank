# -*- coding: utf-8 -*-
"""Per-dataset triviality re-filter: drop tabular datasets where the max|z| RULE (the triviality
criterion) still separates the surviving hard anomalies at test AUC > 0.85. The original filter
was per-DATAPOINT (drop points above the max|z| cut); this adds the per-DATASET check that was
missing, so a dataset whose survivors are still max|z|-rankable is removed.

Only the max|z| rule is used (NOT HBOS): HBOS is a real pool detector, so datasets HBOS solves
but max|z| does not are legitimate selection datasets and are KEPT.

Recomputes the benchmark composition and re-aggregates the existing selectors (HADB_SELECTORS_V2)
on the cleaned set to show the impact on the EM story.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
CUT = 0.85


def find_npz(corpus, name):
    dirs = {"oddbench": ["oddbench"], "ovrbench": ["ovrbench"], "adbench_dami": ["adbench", "dami"]}.get(corpus, [corpus])
    for sub in dirs:
        p = os.path.join(ROOT, "data", sub, name + ".npz")
        if os.path.exists(p):
            return p
    return None


def mz_rule_auc(corpus, name, seed=0):
    p = find_npz(corpus, name)
    if p is None:
        return np.nan
    d = np.load(p, allow_pickle=True)
    if "X" in d:
        X = np.asarray(d["X"], float); y = np.asarray(d["y"]).ravel().astype(int)
    else:
        X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
        y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 6000:
        r = np.random.RandomState(0); k = r.choice(len(X), 6000, replace=False); X, y = X[k], y[k]
    nm = y == 0; an = np.where(y == 1)[0]
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9
    mzn = np.abs((X[nm] - mu) / sd).max(1); thr = np.percentile(mzn, 99)
    hard = an[np.abs((X[an] - mu) / sd).max(1) <= thr]
    if len(hard) < 5:
        return np.nan
    ni = np.where(nm)[0]; g = np.random.default_rng(seed); idx = np.arange(len(ni)); g.shuffle(idx); c2 = int(0.8 * len(idx))
    tr = ni[idx[:c2]]; tn = ni[idx[c2:]]
    mt, st = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xe = np.vstack([X[tn], X[hard]]); ye = np.r_[np.zeros(len(tn)), np.ones(len(hard))]
    return float(roc_auc_score(ye, np.abs((Xe - mt) / st).max(1)))


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
inc = M[M.include].copy()
tab = inc[inc.modality == "tabular"]
print(f"computing max|z|-rule AUC for {len(tab)} included tabular datasets ...")
auc = {}
for _, r in tab.iterrows():
    auc[r.dataset] = mz_rule_auc(r.corpus, r.dataset)
M["mz_rule_auc"] = M.dataset.map(auc)
# new include: keep TS unchanged; for tabular drop max|z|-rule-trivial (>CUT)
M["include_v2"] = M.include & ~((M.modality == "tabular") & (M.mz_rule_auc > CUT))
M.to_csv(os.path.join(S, "HADB_MANIFEST_REFILTERED.csv"), index=False)

old = M[M.include]; new = M[M.include_v2]
dropped = old[~old.dataset.isin(new.dataset)]
print(f"\n=== re-filter at max|z|-rule AUC > {CUT} ===")
print(f"  included before: {len(old)}  ->  after: {len(new)}   (dropped {len(dropped)} tabular)")
print(f"  by modality after: {dict(new.modality.value_counts())}")
print(f"  dropped were global-best: {int(dropped.gt_best.str.startswith(('HBOS','COPOD','ECOD','PCA')).sum())}/{len(dropped)}")

# impact on selectors (from V2), old vs cleaned
V = pd.read_csv(os.path.join(S, "HADB_SELECTORS_V2.csv"))
keep = set(new.dataset)
for tag, sub in [("BEFORE (all included)", V), ("AFTER (cleaned)", V[V.dataset.isin(keep)])]:
    print(f"\n  {tag}: {sub.dataset.nunique()} datasets")
    for sel in ["global_fixed", "em", "mv", "consensus", "iforest_random", "random"]:
        s = sub[sub.selector == sel]
        if len(s):
            print(f"    {sel:16s} regret {s.regret.mean():.4f}")
