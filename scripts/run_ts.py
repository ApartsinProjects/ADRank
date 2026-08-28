"""Run ADRank pipeline on synthetic time-series datasets (10 series).

Uses the same detectors and pseudo-eval as the tabular runner. Time-series are
windowed into feature vectors by `adrank.ts.load_synthetic_ts`.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import pseudo_auc_for_dataset, true_rank_from_labels, detector_names
from adrank.ts import load_synthetic_ts

warnings.filterwarnings("ignore")


def _one_pseudo(ds, seed):
    t0 = time.time()
    # ensemble: smallest + random cluster-selection, K=30, mean-aggregated downstream
    parts = []
    for sel in ("smallest", "random"):
        d = pseudo_auc_for_dataset(ds, K=30, M=20, selection=sel, seed=seed)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["seed"] = seed
    print(f"  pseudo done: {ds.name} in {time.time()-t0:.1f}s", flush=True)
    return df


def _one_true(ds, seed):
    t0 = time.time()
    df = true_rank_from_labels(ds, seed=seed)
    df["seed"] = seed
    print(f"  true done: {ds.name} in {time.time()-t0:.1f}s", flush=True)
    return df


def main():
    datasets = load_synthetic_ts(seed=0)
    print(f"[ts] {len(datasets)} time-series datasets, {len(detector_names())} detectors")
    for ds in datasets:
        print(f"  {ds.name}: X={ds.X.shape}, anom_windows={int(ds.y.sum())}/{len(ds.y)} ({100*ds.y.mean():.1f}%)")

    out = os.path.join(ROOT, "results", "raw")
    os.makedirs(out, exist_ok=True)

    true_dfs = Parallel(n_jobs=4)(delayed(_one_true)(ds, 0) for ds in datasets)
    pd.concat(true_dfs, ignore_index=True).to_parquet(os.path.join(out, "true_ts.parquet"))

    pseudo_dfs = Parallel(n_jobs=4)(delayed(_one_pseudo)(ds, 0) for ds in datasets)
    pd.concat(pseudo_dfs, ignore_index=True).to_parquet(os.path.join(out, "pseudo_ts.parquet"))

    print("[ts] done.")


if __name__ == "__main__":
    main()
