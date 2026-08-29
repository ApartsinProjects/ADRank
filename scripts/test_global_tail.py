"""Test the audit's #1 fix: add a GLOBAL-TAIL pseudo-anomaly regime to the bank.

Root cause (from audit): cluster-holdout pseudo-anomalies are LOCAL; on datasets
whose true-best detector is GLOBAL (HBOS/COPOD/ECOD/PCA/IForest/LODA), no
cluster-based regime can make the global detector win. A global-tail regime
selects marginal-outlier points (not clusters) as pseudo-anomalies, which global
detectors should separate. If truth-aligned, adding it recovers regret on the
global-best datasets that dominate the DAMI and tabular weak spots.

Two global-regime variants are generated fresh on DAMI:
  gtail_marg : pseudo-anomalies = points with highest mean|z| (marginal tail)
  gtail_dist : pseudo-anomalies = points farthest from the global centroid (std space)
We then augment the existing DAMI regime bank and recompute discriminative regret.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from adrank.pipeline import load_npz_dir, fit_and_score, detector_names, true_rank_from_labels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from analyze_regimes import _aggregate_rank

SEEDS = [0, 1, 2, 3, 4]
M = 10
TARGET = 0.05
HOLDOUT = 0.2


def global_tail_pseudo(ds, mode, seed):
    """Return per-(detector) mean pseudo-AUC for a global-tail regime."""
    rng = np.random.default_rng(seed)
    Xs = StandardScaler().fit_transform(ds.X)
    n = len(Xs)
    if mode == "marg":
        score = np.mean(np.abs(Xs), axis=1)            # marginal-tail outlyingness
    else:  # dist
        score = np.linalg.norm(Xs - Xs.mean(0), axis=1)  # far from global centroid
    order = np.argsort(-score)                          # most-anomalous first
    pool_size = max(int(0.25 * n), int(2 * TARGET * n) + 10)
    anom_pool = order[:pool_size]                       # candidate global anomalies
    normal_pool = order[pool_size:]                     # the rest = normal
    dets = detector_names(ds.X.shape[1])
    per = {d: [] for d in dets}
    tsize = max(20, int(TARGET * n))
    for j in range(M):
        panom = rng.choice(anom_pool, size=min(tsize, len(anom_pool)), replace=False)
        rest = np.setdiff1d(normal_pool, panom, assume_unique=False)
        rng.shuffle(rest)
        nh = min(max(50, int(HOLDOUT * len(rest))), len(rest) // 3)
        pnorm = rest[:nh]; train = rest[nh:]
        sidx = np.concatenate([pnorm, panom]); y = np.concatenate([np.zeros(len(pnorm)), np.ones(len(panom))])
        for d in dets:
            s = fit_and_score(d, ds.X[train], ds.X[sidx])
            per[d].append(np.nan if s is None else roc_auc_score(y, s))
    return {d: np.nanmean(v) if len(v) else np.nan for d, v in per.items()}


def regret_of_bank(pseudo_bank, tavg, scheme="discriminative"):
    """pseudo_bank: df with regime,detector,pseudo_auc for ONE dataset+seed."""
    score = _aggregate_rank(pseudo_bank, scheme)
    pick = score.sort_values(ascending=False).index[0]
    return tavg.true_auc.max() - tavg[tavg.detector == pick].true_auc.values[0]


def main():
    p = pd.read_parquet(os.path.join(ROOT, "results", "raw", "modal_pseudo_dami.parquet"))
    t = pd.read_parquet(os.path.join(ROOT, "results", "raw", "modal_true_dami.parquet"))
    dss = {d.name: d for d in load_npz_dir(os.path.join(ROOT, "data", "dami")) if d.y.mean() <= 0.35 and d.name != "InternetAds"}

    rows = []
    for add in ["none", "marg", "dist", "both"]:
        seed_reg = []
        for seed in SEEDS:
            ps = p[p.seed == seed]; ts_ = t[t.seed == seed]
            regrets = []
            for name, cell in ps.groupby("dataset"):
                if name not in dss:
                    continue
                tavg = ts_[ts_.dataset == name].groupby("detector").true_auc.mean().reset_index()
                bank = cell[["regime", "detector", "pseudo_auc"]].copy()
                extra = []
                if add in ("marg", "both"):
                    g = global_tail_pseudo(dss[name], "marg", seed)
                    extra.append(pd.DataFrame({"regime": "gtail_marg", "detector": list(g), "pseudo_auc": list(g.values())}))
                if add in ("dist", "both"):
                    g = global_tail_pseudo(dss[name], "dist", seed)
                    extra.append(pd.DataFrame({"regime": "gtail_dist", "detector": list(g), "pseudo_auc": list(g.values())}))
                if extra:
                    bank = pd.concat([bank] + extra, ignore_index=True)
                regrets.append(regret_of_bank(bank, tavg))
            seed_reg.append(np.mean(regrets))
        rows.append({"bank": add, "regret1": np.mean(seed_reg), "std": np.std(seed_reg)})
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "audit_global_tail_dami.csv"), index=False)
        print(f"  DAMI + gtail[{add}]: regret@1 = {np.mean(seed_reg):.4f} +- {np.std(seed_reg):.4f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "audit_global_tail_dami.csv"), index=False)
    print("\nbaseline (cluster-only) is bank=none; lower is better.")


if __name__ == "__main__":
    main()
