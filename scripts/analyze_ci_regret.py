"""Final cross-modality analysis from the Modal 5-seed run.

Computes, per modality, mean +/- std over 5 seeds of:
  - regret@1, regret@3  (true-AUC gap between best detector and ADRank's pick;
    threshold-free, decision-relevant, self-neutralizing on tied datasets)
  - Spearman rho on the spread>=0.10 subset (secondary, for continuity)
  - top-1 / top-3 hit rate
Plus a random-pick regret baseline per modality.

Config per modality matches each one's seed-0 config (ensemble smallest+random
for tabular/cv/ts; smallest for nlp).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import aggregate_pseudo, correlate_ranks

RAW = os.path.join(ROOT, "results", "raw")


def main():
    p = pd.read_parquet(os.path.join(RAW, "modal_pseudo_allseeds.parquet"))
    t = pd.read_parquet(os.path.join(RAW, "modal_true_allseeds.parquet"))

    rows = []
    per_seed_all = []
    for m in ["tabular", "cv", "nlp", "ts"]:
        ps = p[p.modality == m]
        ts_ = t[t.modality == m]
        seed_rows = []
        for seed in sorted(ps.seed.unique()):
            pp = ps[ps.seed == seed].dropna(subset=["pseudo_auc"])
            tt = ts_[ts_.seed == seed]
            pred = pp.groupby(["dataset", "detector"]).pseudo_auc.mean().reset_index()
            tavg = tt.groupby(["dataset", "detector"]).true_auc.mean().reset_index()

            # regret
            r1, r3 = [], []
            for ds, sub in pred.groupby("dataset"):
                tsub = tavg[tavg.dataset == ds]
                best = tsub.true_auc.max()
                order = sub.sort_values("pseudo_auc", ascending=False)
                pick1 = order.iloc[0].detector
                pick3 = set(order.head(3).detector)
                a1 = tsub[tsub.detector == pick1].true_auc.values
                a3 = tsub[tsub.detector.isin(pick3)].true_auc.max()
                if len(a1):
                    r1.append(best - a1[0])
                r3.append(best - a3)

            # spearman on spread>=0.10 (secondary)
            agg = aggregate_pseudo(ps[ps.seed == seed])
            agg = agg[agg.aggregation == "mean"]
            corr = correlate_ranks(tavg, agg)
            spread = tavg.groupby("dataset").true_auc.agg(lambda s: s.max() - s.min()).rename("true_spread")
            corr = corr.merge(spread, on="dataset")
            good = corr[corr.true_spread >= 0.10]

            seed_rows.append({
                "modality": m, "seed": seed,
                "regret1": np.mean(r1), "regret3": np.mean(r3),
                "rho_s10": good.spearman_rho.mean() if len(good) else np.nan,
                "top1_s10": good.top1_hit.mean() if len(good) else np.nan,
                "n_s10": len(good),
            })
        sd = pd.DataFrame(seed_rows)
        per_seed_all.append(sd)

        # random-pick regret baseline
        rb = []
        for ds, sub in ts_.groupby("dataset"):
            av = sub.groupby("detector").true_auc.mean()
            rb.append(av.max() - av.mean())

        rows.append({
            "modality": m, "n_seeds": sd.seed.nunique(), "n_datasets": ts_.dataset.nunique(),
            "regret1": f"{sd.regret1.mean():.3f}±{sd.regret1.std():.3f}",
            "regret3": f"{sd.regret3.mean():.3f}±{sd.regret3.std():.3f}",
            "regret1_random": f"{np.mean(rb):.3f}",
            "rho_s10": f"{sd.rho_s10.mean():.2f}±{sd.rho_s10.std():.2f}",
            "top1_s10": f"{sd.top1_s10.mean():.2f}±{sd.top1_s10.std():.2f}",
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(ROOT, "results", "ci_regret_summary.csv"), index=False)
    pd.concat(per_seed_all, ignore_index=True).to_csv(
        os.path.join(ROOT, "results", "ci_regret_per_seed.csv"), index=False)
    print("\nwrote results/ci_regret_summary.csv and ci_regret_per_seed.csv")


if __name__ == "__main__":
    main()
