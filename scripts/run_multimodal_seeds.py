"""Multi-seed cross-modality run for confidence intervals.

Runs seeds 1..4 (seed 0 already exists) for TS, CV, NLP using the ensemble
default (smallest+random cluster selection, K=30, M=20, 9 detectors no OCSVM).
Sequential across modalities to avoid CPU oversubscription; parallel across
(dataset, seed) within each modality.

Outputs, one per modality, appended to the existing seed-0 parquet:
  results/raw/pseudo_{cv,nlp,ts}_seeds.parquet
  results/raw/true_{cv,nlp,ts}_seeds.parquet
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
    print(f"\n[{tag}] {len(datasets)} datasets x {len(SEEDS)} new seeds, sel={selections}", flush=True)
    t0 = time.time()
    true_dfs = Parallel(n_jobs=4)(
        delayed(_true)(ds, s) for ds in datasets for s in SEEDS
    )
    pd.concat(true_dfs, ignore_index=True).to_parquet(os.path.join(RAW, f"true_{tag}_seeds.parquet"))
    print(f"[{tag}] true done in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    pseudo_dfs = Parallel(n_jobs=4)(
        delayed(_pseudo)(ds, s, selections) for ds in datasets for s in SEEDS
    )
    pd.concat(pseudo_dfs, ignore_index=True).to_parquet(os.path.join(RAW, f"pseudo_{tag}_seeds.parquet"))
    print(f"[{tag}] pseudo done in {time.time()-t0:.0f}s", flush=True)


def main():
    # config per modality matches each one's seed-0 config:
    #   TS, CV -> smallest+random ensemble; NLP -> smallest only (random too costly on 10k-row sets)
    run_modality("ts", load_synthetic_ts(seed=0), ["smallest", "random"])
    run_modality("cv", load_npz_dir(os.path.join(ROOT, "data", "cv")), ["smallest", "random"])
    run_modality("nlp", load_npz_dir(os.path.join(ROOT, "data", "nlp")), ["smallest"])
    print("\n[multimodal-seeds] all done.")


if __name__ == "__main__":
    main()
