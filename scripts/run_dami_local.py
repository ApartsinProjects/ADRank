"""Local fallback: DAMI regime sweep (Modal workspace was disabled).

Produces the same parquet schema as modal_adrank_dami.py so analyze_regimes-style
code works unchanged. Classical 9-detector panel, six-regime bank, 5 seeds.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed

os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import load_npz_dir, pseudo_auc_for_dataset, true_rank_from_labels

SEEDS = [0, 1, 2, 3, 4]
REGIMES = [(sel, K) for sel in ("smallest", "random", "hard") for K in (30, 50)]


def _cell(ds, seed):
    true_df = true_rank_from_labels(ds, seed=seed); true_df["seed"] = seed; true_df["modality"] = "dami"
    parts = []
    for sel, K in REGIMES:
        d = pseudo_auc_for_dataset(ds, K=K, M=20, selection=sel, seed=seed)
        d["regime"] = f"{sel}_K{K}"; parts.append(d)
    p = pd.concat(parts, ignore_index=True); p["seed"] = seed; p["modality"] = "dami"
    print(f"  {ds.name} seed{seed} done", flush=True)
    return true_df, p


def main():
    dss = [d for d in load_npz_dir(os.path.join(ROOT, "data", "dami")) if d.y.mean() <= 0.35]
    print(f"[dami-local] {len(dss)} datasets (rate<=0.35, size-filtered)")
    for d in dss:
        print(f"  {d.name}: {d.X.shape} anom={d.y.mean()*100:.1f}%")
    t0 = time.time()
    out = Parallel(n_jobs=2)(delayed(_cell)(ds, s) for ds in dss for s in SEEDS)
    true_all = pd.concat([o[0] for o in out], ignore_index=True)
    pseudo_all = pd.concat([o[1] for o in out], ignore_index=True)
    os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)
    true_all.to_parquet(os.path.join(ROOT, "results", "raw", "modal_true_dami.parquet"))
    pseudo_all.to_parquet(os.path.join(ROOT, "results", "raw", "modal_pseudo_dami.parquet"))
    print(f"[dami-local] DONE in {time.time()-t0:.0f}s, pseudo={len(pseudo_all)} rows")


if __name__ == "__main__":
    main()
