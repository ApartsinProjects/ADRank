"""Multi-seed run for confidence intervals on the winning config.

Runs only the winning config (mean/smallest/K=30/M=20) across seeds 0..4.
Seed drives both clustering + subset sampling in `pseudo_auc_for_dataset` and
the normal-fold split in `true_rank_from_labels`.

Estimated wall-clock: 5 seeds x (~7 min per pseudo pass) = ~35 min, plus true
ranks. Runs in parallel across (dataset, seed).
"""
from __future__ import annotations

import argparse
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
    pseudo_auc_for_dataset,
    true_rank_from_labels,
)

warnings.filterwarnings("ignore")


def _one_pseudo(ds, seed):
    t0 = time.time()
    df = pseudo_auc_for_dataset(ds, K=30, M=20, selection="smallest", seed=seed)
    df["seed"] = seed
    print(f"  pseudo done: {ds.name} seed={seed} in {time.time()-t0:.1f}s", flush=True)
    return df


def _one_true(ds, seed):
    t0 = time.time()
    df = true_rank_from_labels(ds, seed=seed)
    df["seed"] = seed
    print(f"  true done: {ds.name} seed={seed} in {time.time()-t0:.1f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_jobs", type=int, default=-2)
    args = ap.parse_args()

    data_dir = os.path.join(ROOT, "data", "adbench")
    out_dir = os.path.join(ROOT, "results", "raw")
    datasets = load_npz_dir(data_dir)
    datasets.sort(key=lambda d: d.X.shape[0])
    print(f"[multiseed] {len(datasets)} datasets x {len(args.seeds)} seeds")

    # true
    print("[multiseed] true ranks ...")
    t0 = time.time()
    true_dfs = Parallel(n_jobs=args.n_jobs)(
        delayed(_one_true)(ds, seed) for ds in datasets for seed in args.seeds
    )
    true_all = pd.concat(true_dfs, ignore_index=True)
    true_all.to_parquet(os.path.join(out_dir, "true_multiseed.parquet"))
    print(f"[multiseed] true ranks: {time.time()-t0:.1f}s")

    # pseudo
    print("[multiseed] pseudo AUC ...")
    t0 = time.time()
    pseudo_dfs = Parallel(n_jobs=args.n_jobs)(
        delayed(_one_pseudo)(ds, seed) for ds in datasets for seed in args.seeds
    )
    pseudo_all = pd.concat(pseudo_dfs, ignore_index=True)
    pseudo_all.to_parquet(os.path.join(out_dir, "pseudo_multiseed.parquet"))
    print(f"[multiseed] pseudo: {time.time()-t0:.1f}s")

    print("[multiseed] done.")


if __name__ == "__main__":
    main()
