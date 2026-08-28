"""Smoke test on 5 synthetic datasets: check that the pipeline runs end-to-end
and produces a positive correlation. Runtime target: <60s on CPU.

Success signal: mean Spearman rho > 0 across aggregations. Real validation happens
on ADBench (see run_v1.py).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import (
    load_synthetic,
    pseudo_auc_for_dataset,
    true_rank_from_labels,
    aggregate_pseudo,
    correlate_ranks,
    detector_names,
)


def main():
    t0 = time.time()
    datasets = load_synthetic(seed=0)
    print(f"[smoke] {len(datasets)} synthetic datasets, {len(detector_names())} detectors")

    true_rows = []
    pseudo_rows = []
    for ds in datasets:
        print(f"  - {ds.name}: X={ds.X.shape}, anomalies={int(ds.y.sum())}/{len(ds.y)}")
        true_rows.append(true_rank_from_labels(ds))
        pseudo_rows.append(pseudo_auc_for_dataset(ds, K=30, M=15, selection="composite_top_quartile"))

    true_df = pd.concat(true_rows, ignore_index=True)
    pseudo_df = pd.concat(pseudo_rows, ignore_index=True)
    agg_df = aggregate_pseudo(pseudo_df)
    corr_df = correlate_ranks(true_df, agg_df)

    print("\n=== per-dataset x aggregation correlations ===")
    print(corr_df.pivot(index="dataset", columns="aggregation", values="spearman_rho").round(3))

    print("\n=== mean across datasets ===")
    summary = corr_df.groupby("aggregation").agg(
        spearman_rho=("spearman_rho", "mean"),
        kendall_tau=("kendall_tau", "mean"),
        top1_hit=("top1_hit", "mean"),
        top3_hit_ratio=("top3_hit_ratio", "mean"),
    ).round(3)
    print(summary)

    # write results
    os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)
    pseudo_df.to_csv(os.path.join(ROOT, "results", "raw", "smoke_pseudo.csv"), index=False)
    true_df.to_csv(os.path.join(ROOT, "results", "raw", "smoke_true.csv"), index=False)
    agg_df.to_csv(os.path.join(ROOT, "results", "raw", "smoke_agg.csv"), index=False)
    corr_df.to_csv(os.path.join(ROOT, "results", "smoke_correlations.csv"), index=False)
    summary.to_csv(os.path.join(ROOT, "results", "smoke_summary.csv"))

    dt = time.time() - t0
    print(f"\n[smoke] finished in {dt:.1f}s")

    # Sanity check: scrambled pseudo scores should give ~0 correlation
    print("\n=== sanity: scrambled pseudo-scores ===")
    scrambled = pseudo_df.copy()
    rng = np.random.default_rng(0)
    scrambled["pseudo_auc"] = rng.permutation(scrambled["pseudo_auc"].values)
    agg_scr = aggregate_pseudo(scrambled)
    corr_scr = correlate_ranks(true_df, agg_scr)
    print(corr_scr.groupby("aggregation")["spearman_rho"].mean().round(3))


if __name__ == "__main__":
    main()
