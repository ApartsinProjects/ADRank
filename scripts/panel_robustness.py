"""Drop-one-detector panel robustness.

For each detector d, recompute the ADRank ranking correlation using only the
remaining 9 detectors. If the reported ρ collapses when a specific detector is
dropped, the whole result depends on that detector.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import aggregate_pseudo, correlate_ranks, detector_names


def main():
    raw = os.path.join(ROOT, "results", "raw")
    pseudo = pd.read_parquet(os.path.join(raw, "pseudo_all.parquet"))
    true = pd.read_parquet(os.path.join(raw, "true_all.parquet"))
    true_avg = true.groupby(["dataset", "detector"])[["true_auc", "true_ap"]].mean().reset_index()

    # winning config
    pseudo = pseudo[(pseudo.K == 30) & (pseudo.selection == "smallest")]

    all_rows = []
    for drop in [None] + detector_names():
        p_sub = pseudo if drop is None else pseudo[pseudo.detector != drop]
        t_sub = true_avg if drop is None else true_avg[true_avg.detector != drop]
        agg = aggregate_pseudo(p_sub)
        agg = agg[agg.aggregation == "mean"]
        corr = correlate_ranks(t_sub, agg)
        spread = t_sub.groupby("dataset")["true_auc"].agg(lambda s: s.max() - s.min()).rename("true_spread")
        corr = corr.merge(spread, on="dataset")
        good = corr[corr["true_spread"] >= 0.10]
        all_rows.append({
            "dropped": drop if drop else "none",
            "rho_all": corr["spearman_rho"].mean(),
            "rho_spread10": good["spearman_rho"].mean(),
            "top1_all": corr["top1_hit"].mean(),
            "top1_spread10": good["top1_hit"].mean(),
            "n_detectors": 10 if drop is None else 9,
        })

    df = pd.DataFrame(all_rows).round(3)
    df = df.sort_values("rho_spread10", ascending=False)
    df.to_csv(os.path.join(ROOT, "results", "panel_robustness.csv"), index=False)
    print(df.to_string(index=False))
    print()
    baseline_rho = df[df.dropped == "none"]["rho_spread10"].iloc[0]
    print(f"Baseline (all 10 detectors) rho_spread10 = {baseline_rho}")
    delta = df.set_index("dropped")["rho_spread10"] - baseline_rho
    print("\nDelta rho_spread10 when each detector is dropped (negative = dropping hurts):")
    print(delta.drop("none").sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
