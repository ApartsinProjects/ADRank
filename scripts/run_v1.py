"""ADRank v1 main runner: real ADBench validation.

Parallelizes across (dataset, K, seed) via joblib. Within each cell it runs all
detectors x subsets serially so they share the clustering.

Outputs:
  results/raw/pseudo_all.parquet  - per (dataset, detector, subset) pseudo-AUC
  results/raw/true_all.parquet    - per (dataset, detector) true AUC (uses y)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adrank.pipeline import (
    load_npz_dir,
    pseudo_auc_for_dataset,
    true_rank_from_labels,
    detector_names,
)

warnings.filterwarnings("ignore")


def _one_pseudo(ds, K, M, selection, seed):
    t0 = time.time()
    df = pseudo_auc_for_dataset(ds, K=K, M=M, selection=selection, seed=seed)
    df["seed"] = seed
    dt = time.time() - t0
    print(f"  pseudo done: {ds.name} K={K} sel={selection} seed={seed} in {dt:.1f}s", flush=True)
    return df


def _one_true(ds, seed):
    t0 = time.time()
    df = true_rank_from_labels(ds, seed=seed)
    df["seed"] = seed
    dt = time.time() - t0
    print(f"  true done: {ds.name} seed={seed} in {dt:.1f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "adbench"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--K", nargs="+", type=int, default=[30, 50])
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--selection", nargs="+", default=["composite_top_quartile", "random", "smallest"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n_jobs", type=int, default=-2)
    ap.add_argument("--limit", type=int, default=0, help="if >0, only run first N datasets by size")
    ap.add_argument("--tag", default="", help="suffix appended to output filenames (e.g. 'cv', 'nlp')")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    datasets = load_npz_dir(args.data)
    datasets.sort(key=lambda d: d.X.shape[0])
    if args.limit > 0:
        datasets = datasets[: args.limit]
    print(f"[run_v1] {len(datasets)} datasets, {len(detector_names())} detectors")
    for ds in datasets:
        print(f"  {ds.name}: X={ds.X.shape}, anom={int(ds.y.sum())} ({100*ds.y.mean():.2f}%)")

    print(f"\n[run_v1] K={args.K}, M={args.M}, selection={args.selection}, seeds={args.seeds}, n_jobs={args.n_jobs}")

    # True ranks
    print("\n[run_v1] computing true ranks (labels used)...")
    t0 = time.time()
    true_jobs = [delayed(_one_true)(ds, seed) for ds in datasets for seed in args.seeds]
    true_dfs = Parallel(n_jobs=args.n_jobs, verbose=0)(true_jobs)
    true_all = pd.concat(true_dfs, ignore_index=True)
    suffix = f"_{args.tag}" if args.tag else "_all"
    true_all.to_parquet(os.path.join(args.out, f"true{suffix}.parquet"))
    print(f"[run_v1] true ranks: {time.time()-t0:.1f}s, saved -> true_all.parquet")

    # Pseudo evaluation
    print("\n[run_v1] computing pseudo-AUC over cluster subsets...")
    t0 = time.time()
    pseudo_jobs = []
    for ds in datasets:
        for K in args.K:
            for sel in args.selection:
                for seed in args.seeds:
                    pseudo_jobs.append(delayed(_one_pseudo)(ds, K, args.M, sel, seed))
    pseudo_dfs = Parallel(n_jobs=args.n_jobs, verbose=0)(pseudo_jobs)
    pseudo_all = pd.concat(pseudo_dfs, ignore_index=True)
    pseudo_all.to_parquet(os.path.join(args.out, f"pseudo{suffix}.parquet"))
    print(f"[run_v1] pseudo-AUC: {time.time()-t0:.1f}s, saved -> pseudo_all.parquet")

    print("\n[run_v1] done.")


if __name__ == "__main__":
    main()
