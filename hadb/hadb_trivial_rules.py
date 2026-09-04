# -*- coding: utf-8 -*-
"""Per-dataset triviality-RULE audit for the HADB construction.

The max|z| anomaly filter is per-DATAPOINT (drop points above the 99th-pct max|z| cut); it does
not check whether a simple per-feature RULE still separates a dataset's SURVIVORS. This computes,
for every scored TABULAR dataset, the test AUC of two simple marginal rules against the surviving
hard anomalies:
  mz_rule_auc    max|z| rule (Gaussian marginal extremity)
  hbos_rule_auc  HBOS-lite rule (sum of per-feature empirical -log histogram density) - catches
                 skewed / bimodal / categorical rarity that the Gaussian max|z| misses.
The consolidator drops tabular datasets where max(mz, hbos) > TRIV_RULE_CUT: solvable by ANY
simple per-feature rule -> not genuinely hard. Time series use the Wu-Keogh one-liner criterion
(already a per-series check) and are left unchanged.

Writes HADB_TRIVIAL_RULES.csv (dataset, corpus, mz_rule_auc, hbos_rule_auc).
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
NBINS = 20
TAB_CORPORA = ("oddbench", "ovrbench", "adbench_dami")


def find_npz(corpus, name):
    for sub in {"oddbench": ["oddbench"], "ovrbench": ["ovrbench"], "adbench_dami": ["adbench", "dami"]}.get(corpus, [corpus]):
        p = os.path.join(ROOT, "data", sub, name + ".npz")
        if os.path.exists(p):
            return p
    return None


def rule_aucs(corpus, name):
    p = find_npz(corpus, name)
    if p is None:
        return np.nan, np.nan
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
    hard = an[np.abs((X[an] - mu) / sd).max(1) <= np.percentile(np.abs((X[nm] - mu) / sd).max(1), 99)]
    if len(hard) < 5:
        return np.nan, np.nan
    ni = np.where(nm)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx); c2 = int(0.8 * len(idx))
    tr = ni[idx[:c2]]; tn = ni[idx[c2:]]
    Xtr = X[tr]; Xe = np.vstack([X[tn], X[hard]]); ye = np.r_[np.zeros(len(tn)), np.ones(len(hard))]
    mt, st = Xtr.mean(0), Xtr.std(0) + 1e-9
    mz = roc_auc_score(ye, np.abs((Xe - mt) / st).max(1))
    hb = np.zeros(len(Xe))
    for j in range(Xtr.shape[1]):
        cnt, edges = np.histogram(Xtr[:, j], bins=NBINS)
        dens = cnt / cnt.sum() + 1e-6
        b = np.clip(np.digitize(Xe[:, j], edges) - 1, 0, NBINS - 1)
        hb += -np.log(dens[b])
    return float(mz), float(roc_auc_score(ye, hb))


if __name__ == "__main__":
    M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
    tab = M[M.corpus.isin(TAB_CORPORA)]
    print(f"auditing triviality rules on {len(tab)} scored tabular datasets ...", flush=True)
    rows = []
    for i, (_, r) in enumerate(tab.iterrows(), 1):
        mz, hb = rule_aucs(r.corpus, r.dataset)
        rows.append(dict(dataset=r.dataset, corpus=r.corpus, mz_rule_auc=mz, hbos_rule_auc=hb))
        if i % 40 == 0:
            print(f"  [{i}/{len(tab)}]", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(S, "HADB_TRIVIAL_RULES.csv"), index=False)
    R["worst"] = R[["mz_rule_auc", "hbos_rule_auc"]].max(1)
    print(f"\n  wrote HADB_TRIVIAL_RULES.csv ({len(R)} datasets)")
    print(f"  max|z|-rule AUC>0.85: {int((R.mz_rule_auc>0.85).sum())}   HBOS-rule AUC>0.85: {int((R.hbos_rule_auc>0.85).sum())}")
    print(f"  EITHER rule >0.85 (will be dropped): {int((R.worst>0.85).sum())}/{len(R)}")
