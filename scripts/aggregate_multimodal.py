"""Aggregate per-modality: tabular / cv / nlp / ts. Same aggregation and
correlation code as aggregate_report.py, applied per parquet pair.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import aggregate_pseudo, correlate_ranks


def _report(pseudo_path, true_path, tag):
    if not (os.path.exists(pseudo_path) and os.path.exists(true_path)):
        return None
    pseudo = pd.read_parquet(pseudo_path)
    true = pd.read_parquet(true_path)
    true_avg = true.groupby(["dataset", "detector"])[["true_auc"]].mean().reset_index()

    agg = aggregate_pseudo(pseudo)
    agg = agg[agg.aggregation == "mean"]
    corr = correlate_ranks(true_avg, agg)
    spread = true_avg.groupby("dataset")["true_auc"].agg(lambda s: s.max() - s.min()).rename("true_spread")
    corr = corr.merge(spread, on="dataset")
    corr["modality"] = tag

    all_stats = {
        "modality": tag,
        "n_datasets": len(corr),
        "rho_all_mean": corr["spearman_rho"].mean(),
        "rho_all_median": corr["spearman_rho"].median(),
        "top1_all": corr["top1_hit"].mean(),
        "top3_all": corr["top3_hit_ratio"].mean(),
    }
    good = corr[corr["true_spread"] >= 0.10]
    all_stats.update({
        "n_spread10": len(good),
        "rho_spread10_mean": good["spearman_rho"].mean() if len(good) else float("nan"),
        "rho_spread10_median": good["spearman_rho"].median() if len(good) else float("nan"),
        "top1_spread10": good["top1_hit"].mean() if len(good) else float("nan"),
        "top3_spread10": good["top3_hit_ratio"].mean() if len(good) else float("nan"),
    })
    return corr, all_stats


def main():
    raw = os.path.join(ROOT, "results", "raw")
    reports = []
    per_dataset = []

    # Apples-to-apples: tabular baseline uses the 9-detector panel (no OCSVM) at seed=0,
    # same config as CV/NLP/TS. Read from the multiseed parquet, seed=0 only.
    pmul = os.path.join(raw, "pseudo_multiseed.parquet")
    tmul = os.path.join(raw, "true_multiseed.parquet")
    if os.path.exists(pmul):
        p = pd.read_parquet(pmul); t = pd.read_parquet(tmul)
        p0 = p[p.seed == 0]; t0 = t[t.seed == 0]
        p0.to_parquet(os.path.join(raw, "_tmp_pseudo_tabular9.parquet"))
        t0.to_parquet(os.path.join(raw, "_tmp_true_tabular9.parquet"))
        out = _report(os.path.join(raw, "_tmp_pseudo_tabular9.parquet"),
                      os.path.join(raw, "_tmp_true_tabular9.parquet"),
                      "tabular")
        if out is not None:
            corr, stats = out
            per_dataset.append(corr)
            reports.append(stats)

    for tag in ["cv", "nlp", "ts"]:
        out = _report(
            os.path.join(raw, f"pseudo_{tag}.parquet"),
            os.path.join(raw, f"true_{tag}.parquet"),
            tag,
        )
        if out is None:
            print(f"[skip] {tag}: parquet files missing")
            continue
        corr, stats = out
        per_dataset.append(corr)
        reports.append(stats)

    summary = pd.DataFrame(reports).round(3)
    print("\n=== per-modality summary (best config: mean/smallest/K=30) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(ROOT, "results", "multimodal_summary.csv"), index=False)

    if per_dataset:
        allc = pd.concat(per_dataset, ignore_index=True)
        allc.to_csv(os.path.join(ROOT, "results", "multimodal_per_dataset.csv"), index=False)
        print(f"\n[report] wrote multimodal_summary.csv and multimodal_per_dataset.csv")


if __name__ == "__main__":
    main()
