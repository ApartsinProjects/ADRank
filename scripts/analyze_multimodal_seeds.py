"""Combine seed-0 (existing) + seeds 1-4 (from run_multimodal_seeds) into
per-modality confidence intervals on well-posed Spearman rho.

Seed-0 sources:
  ts  : pseudo_ts.parquet   (ensemble)      true_ts.parquet
  cv  : pseudo_cv.parquet   (ensemble)      true_cv.parquet
  nlp : pseudo_nlp.parquet  (smallest)      true_nlp.parquet
Seeds 1-4 sources:
  {ts,cv,nlp}_seeds.parquet  (same config per modality)
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


def _load(tag):
    p0 = pd.read_parquet(os.path.join(RAW, f"pseudo_{tag}.parquet"))
    t0 = pd.read_parquet(os.path.join(RAW, f"true_{tag}.parquet"))
    if "seed" not in p0.columns:
        p0 = p0.assign(seed=0)
    if "seed" not in t0.columns:
        t0 = t0.assign(seed=0)
    p0 = p0[p0.seed == 0]; t0 = t0[t0.seed == 0]
    ps = os.path.join(RAW, f"pseudo_{tag}_seeds.parquet")
    ts = os.path.join(RAW, f"true_{tag}_seeds.parquet")
    if os.path.exists(ps):
        p0 = pd.concat([p0, pd.read_parquet(ps)], ignore_index=True)
        t0 = pd.concat([t0, pd.read_parquet(ts)], ignore_index=True)
    return p0, t0


def _per_seed_stats(pseudo, true):
    rows = []
    for seed, psub in pseudo.groupby("seed"):
        tsub = true[true.seed == seed]
        tavg = tsub.groupby(["dataset", "detector"])[["true_auc"]].mean().reset_index()
        agg = aggregate_pseudo(psub)
        agg = agg[agg.aggregation == "mean"]
        corr = correlate_ranks(tavg, agg)
        spread = tavg.groupby("dataset")["true_auc"].agg(lambda s: s.max() - s.min()).rename("true_spread")
        corr = corr.merge(spread, on="dataset")
        good = corr[corr["true_spread"] >= 0.10]
        rows.append({
            "seed": seed,
            "rho_all": corr["spearman_rho"].mean(),
            "rho_s10": good["spearman_rho"].mean() if len(good) else np.nan,
            "top1_s10": good["top1_hit"].mean() if len(good) else np.nan,
            "top3_s10": good["top3_hit_ratio"].mean() if len(good) else np.nan,
            "n_s10": len(good),
        })
    return pd.DataFrame(rows)


def main():
    out = []
    for tag in ["tabular", "cv", "nlp", "ts"]:
        # tabular uses the multiseed parquet directly (already 5 seeds, smallest)
        if tag == "tabular":
            p = pd.read_parquet(os.path.join(RAW, "pseudo_multiseed.parquet"))
            t = pd.read_parquet(os.path.join(RAW, "true_multiseed.parquet"))
        else:
            if not os.path.exists(os.path.join(RAW, f"pseudo_{tag}_seeds.parquet")):
                print(f"[skip] {tag}: seeds parquet not present yet")
                continue
            p, t = _load(tag)
        st = _per_seed_stats(p, t)
        n_seeds = st["seed"].nunique()
        out.append({
            "modality": tag,
            "n_seeds": n_seeds,
            "rho_all_mean": st["rho_all"].mean(),
            "rho_all_std": st["rho_all"].std(),
            "rho_s10_mean": st["rho_s10"].mean(),
            "rho_s10_std": st["rho_s10"].std(),
            "top1_s10_mean": st["top1_s10"].mean(),
            "top1_s10_std": st["top1_s10"].std(),
            "top3_s10_mean": st["top3_s10"].mean(),
        })
        print(f"\n=== {tag} ({n_seeds} seeds) ===")
        print(st.round(3).to_string(index=False))

    summary = pd.DataFrame(out).round(3)
    print("\n=== CROSS-MODALITY CI SUMMARY ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(ROOT, "results", "multimodal_ci_summary.csv"), index=False)


if __name__ == "__main__":
    main()
