"""Compute the consensus and cluster-geometry baselines on the same 26 ADBench
datasets, for direct comparison against ADRank in aggregate_report.py.
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

from adrank.pipeline import (
    load_npz_dir,
    consensus_baseline_rank,
    cluster_geometry_baseline_rank,
    true_rank_from_labels,
)

warnings.filterwarnings("ignore")


def _one_consensus(ds, seed):
    t0 = time.time()
    df = consensus_baseline_rank(ds, seed=seed)
    df["seed"] = seed
    print(f"  consensus done: {ds.name} seed={seed} in {time.time()-t0:.1f}s", flush=True)
    return df


def _one_geom(ds, seed):
    t0 = time.time()
    df = cluster_geometry_baseline_rank(ds, K=50, M=20, seed=seed)
    df["seed"] = seed
    print(f"  cluster_geom done: {ds.name} seed={seed} in {time.time()-t0:.1f}s", flush=True)
    return df


def main():
    data_dir = os.path.join(ROOT, "data", "adbench")
    out_dir = os.path.join(ROOT, "results", "raw")
    os.makedirs(out_dir, exist_ok=True)
    datasets = load_npz_dir(data_dir)
    datasets.sort(key=lambda d: d.X.shape[0])
    print(f"[baselines] {len(datasets)} datasets")

    print("[baselines] consensus baseline ...")
    dfs = Parallel(n_jobs=-2, verbose=0)(
        delayed(_one_consensus)(ds, 0) for ds in datasets
    )
    pd.concat(dfs, ignore_index=True).to_parquet(os.path.join(out_dir, "baseline_consensus.parquet"))

    print("[baselines] cluster-geometry baseline ...")
    dfs = Parallel(n_jobs=-2, verbose=0)(
        delayed(_one_geom)(ds, 0) for ds in datasets
    )
    pd.concat(dfs, ignore_index=True).to_parquet(os.path.join(out_dir, "baseline_cluster_geom.parquet"))

    print("[baselines] done.")


if __name__ == "__main__":
    main()
