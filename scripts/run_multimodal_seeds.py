"""Multi-seed cross-modality run for confidence intervals (resilient version).

Runs seeds 1..4 (seed 0 already exists) for TS, CV, NLP using each modality's
seed-0 config (TS/CV: smallest+random ensemble; NLP: smallest only).

Resilience (learned from a worker OOM/crash on the first attempt):
  - process ONE dataset at a time, writing a per-dataset parquet
  - n_jobs=2 (not 4) to halve peak memory on 512/768-dim embeddings
  - idempotent: skip a dataset whose per-dataset parquet already exists
  - catch a worker crash (TerminatedWorkerError) per dataset and skip it,
    so one bad cell cannot kill the whole run; re-running picks up the rest

Per-dataset outputs land in results/raw/seeds_parts/{tag}/{dataset}.parquet ;
a final pass concatenates them to results/raw/pseudo_{tag}_seeds.parquet and
true_{tag}_seeds.parquet.
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

os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")

from adrank.pipeline import pseudo_auc_for_dataset, true_rank_from_labels, load_npz_dir
from adrank.ts import load_synthetic_ts

warnings.filterwarnings("ignore")

SEEDS = [1, 2, 3, 4]
RAW = os.path.join(ROOT, "results", "raw")
PARTS = os.path.join(RAW, "seeds_parts")


def _pseudo(ds, seed, selections):
    parts = []
    for sel in selections:
        d = pseudo_auc_for_dataset(ds, K=30, M=20, selection=sel, seed=seed)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["seed"] = seed
    return df


def _true(ds, seed):
    df = true_rank_from_labels(ds, seed=seed)
    df["seed"] = seed
    return df


def run_modality(tag, datasets, selections):
    outdir = os.path.join(PARTS, tag)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n[{tag}] {len(datasets)} datasets x {len(SEEDS)} new seeds, sel={selections}", flush=True)

    for ds in datasets:
        p_part = os.path.join(outdir, f"pseudo_{ds.name}.parquet")
        t_part = os.path.join(outdir, f"true_{ds.name}.parquet")
        if os.path.exists(p_part) and os.path.exists(t_part):
            print(f"  [{tag}] skip {ds.name} (cached)", flush=True)
            continue
        t0 = time.time()
        try:
            true_dfs = Parallel(n_jobs=2)(delayed(_true)(ds, s) for s in SEEDS)
            pseudo_dfs = Parallel(n_jobs=2)(delayed(_pseudo)(ds, s, selections) for s in SEEDS)
            pd.concat(true_dfs, ignore_index=True).to_parquet(t_part)
            pd.concat(pseudo_dfs, ignore_index=True).to_parquet(p_part)
            print(f"  [{tag}] {ds.name} done in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  [{tag}] {ds.name} FAILED ({type(e).__name__}: {e}); skipping", flush=True)

    # concatenate all present parts
    p_all, t_all = [], []
    for fn in os.listdir(outdir):
        full = os.path.join(outdir, fn)
        if fn.startswith("pseudo_"):
            p_all.append(pd.read_parquet(full))
        elif fn.startswith("true_"):
            t_all.append(pd.read_parquet(full))
    if p_all:
        pd.concat(p_all, ignore_index=True).to_parquet(os.path.join(RAW, f"pseudo_{tag}_seeds.parquet"))
        pd.concat(t_all, ignore_index=True).to_parquet(os.path.join(RAW, f"true_{tag}_seeds.parquet"))
        print(f"[{tag}] concatenated {len(p_all)} datasets -> pseudo_{tag}_seeds.parquet", flush=True)


def main():
    run_modality("ts", load_synthetic_ts(seed=0), ["smallest", "random"])
    run_modality("cv", load_npz_dir(os.path.join(ROOT, "data", "cv")), ["smallest", "random"])
    run_modality("nlp", load_npz_dir(os.path.join(ROOT, "data", "nlp")), ["smallest"])
    print("\n[multimodal-seeds] all done.")


if __name__ == "__main__":
    main()
