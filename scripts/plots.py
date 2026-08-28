"""Visualize v1 results.

Produces:
  results/plots/rho_by_dataset.png    - horizontal bar of per-dataset rho, colored by true-spread
  results/plots/scatter_pred_vs_true.png - one panel per dataset: predicted rank vs true rank
  results/plots/rho_vs_spread.png     - scatter of rho vs true-AUC spread
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import aggregate_pseudo, correlate_ranks


def main():
    raw = os.path.join(ROOT, "results", "raw")
    pseudo = pd.read_parquet(os.path.join(raw, "pseudo_all.parquet"))
    true = pd.read_parquet(os.path.join(raw, "true_all.parquet"))
    true_avg = true.groupby(["dataset", "detector"])[["true_auc", "true_ap"]].mean().reset_index()

    plots_dir = os.path.join(ROOT, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # pick the default config: K=50, composite selection, mean aggregation
    pseudo_default = pseudo[(pseudo["K"] == 50) & (pseudo["selection"] == "composite_top_quartile")]
    if len(pseudo_default) == 0:
        pseudo_default = pseudo[pseudo["selection"] == "composite_top_quartile"]

    agg = aggregate_pseudo(pseudo_default)
    corr = correlate_ranks(true_avg, agg)

    # 1) rho per dataset
    spread = true_avg.groupby("dataset")["true_auc"].agg(lambda s: s.max() - s.min())
    for aggname in ["mean", "borda", "varweight"]:
        sub = corr[corr["aggregation"] == aggname].copy()
        sub = sub.merge(spread.rename("true_spread"), on="dataset")
        sub = sub.sort_values("spearman_rho")

        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(sub))))
        colors = plt.cm.viridis((sub["true_spread"] - sub["true_spread"].min()) /
                                 (sub["true_spread"].max() - sub["true_spread"].min() + 1e-9))
        ax.barh(sub["dataset"], sub["spearman_rho"], color=colors)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Spearman rho (pseudo-rank vs true-rank)")
        ax.set_title(f"ADRank v1 — per-dataset correlation (agg={aggname})\ncolor = true-AUC spread across detectors")
        ax.set_xlim(-1, 1)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"rho_by_dataset_{aggname}.png"), dpi=120)
        plt.close(fig)

    # 2) scatter of rho vs spread
    fig, ax = plt.subplots(figsize=(6, 5))
    for aggname, color in zip(["mean", "borda", "varweight"], ["C0", "C1", "C2"]):
        sub = corr[corr["aggregation"] == aggname].copy()
        sub = sub.merge(spread.rename("true_spread"), on="dataset")
        ax.scatter(sub["true_spread"], sub["spearman_rho"], label=aggname, alpha=0.7, color=color)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0.10, color="red", lw=0.5, ls="--", label="spread=0.10")
    ax.set_xlabel("True-AUC spread (max - min across detectors)")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Signal quality vs detector-disagreement")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "rho_vs_spread.png"), dpi=120)
    plt.close(fig)

    # 3) predicted rank vs true rank scatter per dataset
    true["true_rank"] = true.groupby("dataset")["true_auc"].rank(ascending=False, method="average")
    mean_agg = agg[agg["aggregation"] == "mean"]
    m = mean_agg.merge(
        true.groupby(["dataset", "detector"])["true_rank"].mean().reset_index(),
        on=["dataset", "detector"],
    )
    datasets = sorted(m["dataset"].unique())
    ncol = 4
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    for ax, ds in zip(axes.ravel(), datasets):
        sub = m[m["dataset"] == ds]
        ax.scatter(sub["true_rank"], sub["rank"], s=25)
        ax.plot([0, len(sub) + 1], [0, len(sub) + 1], color="grey", ls="--", lw=0.5)
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("true rank")
        ax.set_ylabel("pred rank")
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "scatter_pred_vs_true.png"), dpi=100)
    plt.close(fig)

    print(f"[plots] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
