"""Auto-calibrated ranking from a bank of pseudo-anomaly regimes.

Given per-regime pseudo-AUC (from modal_adrank_regimes.py), aggregate the bank
into one detector ranking under several LABEL-FREE weighting schemes, and measure
regret per modality. The question: can auto-calibration fix the text weak spot
without hurting the other modalities?

Weighting schemes (all label-free unless marked ORACLE):
  single_smallest   : one regime (smallest_K30) -- the old default component
  ens_smallest_rand : mean of smallest+random at K30 -- the current paper default
  uniform_bank      : mean rank across ALL regimes (incl. hard)
  discriminative    : weight each regime by cross-detector pseudo-AUC variance
                      (a regime where detectors tie is uninformative)
  stability         : weight each regime by 1/(rank instability across its subsets)
  ORACLE_best_regime: pick, per dataset, the regime whose ranking best matches
                      truth (uses labels; upper bound only)

Metric: regret@1 (true-AUC gap between best detector and the aggregated pick),
mean over datasets, mean +/- std over seeds.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")


def _detector_scores_per_regime(pp):
    """mean pseudo-AUC per (regime, detector) for one (modality,dataset,seed)."""
    return pp.dropna(subset=["pseudo_auc"]).groupby(["regime", "detector"]).pseudo_auc.mean()


def _regime_weight(pp, scheme):
    """Return dict regime -> weight (label-free)."""
    regimes = pp.regime.unique()
    if scheme in ("uniform_bank",):
        return {r: 1.0 for r in regimes}
    if scheme == "discriminative":
        # variance of mean-pseudo-AUC across detectors, per regime
        w = {}
        for r in regimes:
            v = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
            w[r] = float(v.var())
        s = sum(w.values()) or 1.0
        return {r: w[r] / s for r in regimes}
    if scheme == "stability":
        # inverse rank-instability: within a regime, variance of per-subset detector ranks
        w = {}
        for r in regimes:
            sub = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).copy()
            sub["rk"] = sub.groupby("subset").pseudo_auc.rank(ascending=False)
            inst = sub.groupby("detector").rk.std().mean()  # avg rank wobble across subsets
            w[r] = 1.0 / (inst + 1e-6)
        s = sum(w.values()) or 1.0
        return {r: w[r] / s for r in regimes}
    raise ValueError(scheme)


def _aggregate_rank(pp, scheme):
    """Return a Series detector -> aggregated score (higher=better) for one cell."""
    ms = _detector_scores_per_regime(pp)  # index (regime, detector)
    if scheme == "single_smallest":
        sub = ms.loc[[r for r in ms.index.get_level_values(0).unique() if r == "smallest_K30"]]
        return sub.groupby("detector").mean()
    if scheme == "ens_smallest_rand":
        keep = [r for r in ms.index.get_level_values(0).unique() if r in ("smallest_K30", "random_K30")]
        return ms.loc[keep].groupby("detector").mean()
    # weighted schemes over the full bank: convert each regime to per-detector ranks, weight, average
    w = _regime_weight(pp, scheme)
    dets = ms.index.get_level_values(1).unique()
    agg = pd.Series(0.0, index=dets); wsum = 0.0
    for r in ms.index.get_level_values(0).unique():
        rr = ms.loc[r]
        rank = rr.rank(ascending=True)  # higher rank number = better (so it sums like a score)
        agg = agg.add(rank * w.get(r, 0.0), fill_value=0.0); wsum += w.get(r, 0.0)
    return agg / (wsum or 1.0)


def _regret(pred_score, tavg_ds):
    best = tavg_ds.true_auc.max()
    pick = pred_score.sort_values(ascending=False).index[0]
    pv = tavg_ds[tavg_ds.detector == pick].true_auc
    return best - (pv.values[0] if len(pv) else best)


def main():
    p = pd.read_parquet(os.path.join(RAW, "modal_pseudo_regimes.parquet"))
    t = pd.read_parquet(os.path.join(RAW, "modal_true_regimes.parquet"))

    schemes = ["single_smallest", "ens_smallest_rand", "uniform_bank",
               "discriminative", "stability", "ORACLE_best_regime"]
    out = []
    for m in ["tabular", "cv", "nlp", "ts"]:
        pm = p[p.modality == m]; tm = t[t.modality == m]
        row = {"modality": m}
        for scheme in schemes:
            seed_reg = []
            for seed in sorted(pm.seed.unique()):
                ps = pm[pm.seed == seed]; ts_ = tm[tm.seed == seed]
                tavg = ts_.groupby(["dataset", "detector"]).true_auc.mean().reset_index()
                regrets = []
                for ds, cell in ps.groupby("dataset"):
                    tds = tavg[tavg.dataset == ds]
                    if scheme == "ORACLE_best_regime":
                        # pick regime whose top-1 has best true AUC (uses labels)
                        best_r = None; best_pick_auc = -1
                        ms = _detector_scores_per_regime(cell)
                        for r in ms.index.get_level_values(0).unique():
                            pick = ms.loc[r].sort_values(ascending=False).index[0]
                            av = tds[tds.detector == pick].true_auc
                            if len(av) and av.values[0] > best_pick_auc:
                                best_pick_auc = av.values[0]; best_r = r
                        regrets.append(tds.true_auc.max() - best_pick_auc)
                    else:
                        score = _aggregate_rank(cell, scheme)
                        regrets.append(_regret(score, tds))
                seed_reg.append(np.mean(regrets))
            row[scheme] = f"{np.mean(seed_reg):.3f}±{np.std(seed_reg):.3f}"
        out.append(row)
    df = pd.DataFrame(out)
    print("regret@1 by weighting scheme (lower=better); ORACLE uses labels (upper bound)\n")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(ROOT, "results", "regime_calibration.csv"), index=False)


if __name__ == "__main__":
    main()
