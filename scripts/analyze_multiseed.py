"""Compute confidence intervals for the winning config from the multi-seed run.

For each dataset, aggregate pseudo-AUC across (subset, seed) then rank. Report
per-dataset ρ mean and std across the 5 seeds, and the aggregate ρ CI.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import aggregate_pseudo, correlate_ranks


def main():
    raw = os.path.join(ROOT, "results", "raw")
    pseudo = pd.read_parquet(os.path.join(raw, "pseudo_multiseed.parquet"))
    true = pd.read_parquet(os.path.join(raw, "true_multiseed.parquet"))

    per_seed_rows = []
    for seed, sub in pseudo.groupby("seed"):
        true_seed = true[true.seed == seed]
        true_avg = true_seed.groupby(["dataset", "detector"])[["true_auc"]].mean().reset_index()
        agg = aggregate_pseudo(sub)
        agg = agg[agg.aggregation == "mean"]
        corr = correlate_ranks(true_avg, agg)
        spread = true_avg.groupby("dataset")["true_auc"].agg(lambda s: s.max() - s.min()).rename("true_spread")
        corr = corr.merge(spread, on="dataset")
        corr["seed"] = seed
        per_seed_rows.append(corr)

    per_seed = pd.concat(per_seed_rows, ignore_index=True)
    per_seed.to_csv(os.path.join(ROOT, "results", "multiseed_per_seed.csv"), index=False)

    # per-dataset mean +- std across seeds
    per_ds = per_seed.groupby(["dataset", "true_spread"]).agg(
        rho_mean=("spearman_rho", "mean"),
        rho_std=("spearman_rho", "std"),
        top1_hit=("top1_hit", "mean"),
    ).reset_index().round(3).sort_values("true_spread", ascending=False)
    per_ds.to_csv(os.path.join(ROOT, "results", "multiseed_per_dataset.csv"), index=False)
    print("=== per-dataset (5 seeds) ===")
    print(per_ds.to_string(index=False))

    # aggregate across datasets: for each seed, get overall stats; then mean +- std across seeds
    seed_agg_rows = []
    for seed, sub in per_seed.groupby("seed"):
        good = sub[sub["true_spread"] >= 0.10]
        seed_agg_rows.append({
            "seed": seed,
            "rho_all_mean": sub["spearman_rho"].mean(),
            "rho_all_median": sub["spearman_rho"].median(),
            "rho_spread10_mean": good["spearman_rho"].mean(),
            "rho_spread10_median": good["spearman_rho"].median(),
            "top1_all": sub["top1_hit"].mean(),
            "top1_spread10": good["top1_hit"].mean(),
            "top3_spread10": good["top3_hit_ratio"].mean(),
        })
    seed_agg = pd.DataFrame(seed_agg_rows).round(3)
    seed_agg.to_csv(os.path.join(ROOT, "results", "multiseed_per_seed_agg.csv"), index=False)
    print("\n=== per-seed aggregate ===")
    print(seed_agg.to_string(index=False))

    print("\n=== 5-seed mean +- std ===")
    metrics = ["rho_all_mean", "rho_spread10_mean", "top1_all", "top1_spread10", "top3_spread10"]
    for m in metrics:
        v = seed_agg[m]
        print(f"  {m}: {v.mean():.3f} +- {v.std():.3f}")


if __name__ == "__main__":
    main()
