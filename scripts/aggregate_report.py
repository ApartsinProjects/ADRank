"""Aggregate and correlate results from run_v1.py.

Produces:
  results/correlations.csv       - per (dataset, K, selection, aggregation) rho/tau/hit
  results/summary.csv            - mean/median across datasets, grouped
  results/pivot_rho.csv          - readable pivot: dataset x (K, selection, aggregation)
  results/sanity_scrambled.csv   - control: scrambled pseudo scores -> should be ~0
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import aggregate_pseudo, correlate_ranks


def _agg_over_facets(pseudo_all, true_all, facets):
    """For each unique combination of facet columns in pseudo_all, aggregate and correlate."""
    rows = []
    for keys, sub in pseudo_all.groupby(facets):
        keys = keys if isinstance(keys, tuple) else (keys,)
        agg_df = aggregate_pseudo(sub)
        corr_df = correlate_ranks(true_all, agg_df)
        for f, k in zip(facets, keys):
            corr_df[f] = k
        rows.append(corr_df)
    return pd.concat(rows, ignore_index=True)


def main():
    raw = os.path.join(ROOT, "results", "raw")
    pseudo_all = pd.read_parquet(os.path.join(raw, "pseudo_all.parquet"))
    true_all = pd.read_parquet(os.path.join(raw, "true_all.parquet"))
    print(f"[report] pseudo rows={len(pseudo_all)}, true rows={len(true_all)}")

    # true_all may have multiple seeds; average
    true_avg = true_all.groupby(["dataset", "detector"])[["true_auc", "true_ap"]].mean().reset_index()

    facets = ["K", "selection", "seed"]
    facets = [f for f in facets if f in pseudo_all.columns]

    corr = _agg_over_facets(pseudo_all, true_avg, facets)

    # annotate each dataset with its true-AUC spread (max - min across detectors)
    spread = (true_avg.groupby("dataset")["true_auc"].agg(lambda s: s.max() - s.min())
              .rename("true_spread").reset_index())
    corr = corr.merge(spread, on="dataset", how="left")

    corr.to_csv(os.path.join(ROOT, "results", "correlations.csv"), index=False)
    print(f"[report] correlations rows={len(corr)} -> results/correlations.csv")

    # summary: mean spearman by (K, selection, aggregation), across datasets
    group_cols = ["aggregation"] + facets
    def _summarize(df, tag):
        s = df.groupby(group_cols).agg(
            n_datasets=("dataset", "nunique"),
            n_valid=("spearman_rho", lambda s: int(s.notna().sum())),
            rho_mean=("spearman_rho", "mean"),
            rho_median=("spearman_rho", "median"),
            tau_mean=("kendall_tau", "mean"),
            top1_hit=("top1_hit", "mean"),
            top3_hit_ratio=("top3_hit_ratio", "mean"),
        ).reset_index().round(3)
        s["subset"] = tag
        return s

    summary_all = _summarize(corr, "all")
    summary_meaningful = _summarize(corr[corr["true_spread"] >= 0.10], "spread>=0.10")
    summary = pd.concat([summary_all, summary_meaningful], ignore_index=True)
    summary = summary.sort_values(["subset", "rho_mean"], ascending=[True, False])
    summary.to_csv(os.path.join(ROOT, "results", "summary.csv"), index=False)
    print("\n[report] === SUMMARY: ALL datasets ===")
    print(summary_all.sort_values("rho_mean", ascending=False).to_string(index=False))
    print("\n[report] === SUMMARY: datasets with true-AUC spread >= 0.10 ===")
    print(summary_meaningful.sort_values("rho_mean", ascending=False).to_string(index=False))

    # pivot: dataset x (K, selection, aggregation) for spearman
    corr["facet"] = corr[facets + ["aggregation"]].astype(str).agg("|".join, axis=1)
    piv = corr.pivot_table(index="dataset", columns="facet", values="spearman_rho")
    piv.to_csv(os.path.join(ROOT, "results", "pivot_rho.csv"))
    print(f"\n[report] pivot -> results/pivot_rho.csv ({piv.shape[0]} datasets x {piv.shape[1]} facets)")

    # sanity: scrambled control
    print("\n[report] running scrambled-control sanity check ...")
    scrambled = pseudo_all.copy()
    rng = np.random.default_rng(0)
    scrambled["pseudo_auc"] = rng.permutation(scrambled["pseudo_auc"].values)
    sc = _agg_over_facets(scrambled, true_avg, facets)
    sc_summary = sc.groupby(group_cols).agg(
        rho_mean=("spearman_rho", "mean"),
        top1_hit=("top1_hit", "mean"),
    ).reset_index().round(3)
    sc_summary.to_csv(os.path.join(ROOT, "results", "sanity_scrambled.csv"), index=False)
    print("[report] scrambled rho_mean (expect near 0):")
    print(sc_summary.to_string(index=False))


if __name__ == "__main__":
    main()
