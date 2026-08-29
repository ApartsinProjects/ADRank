"""Modal sweep over a BANK of pseudo-anomaly regimes, for auto-calibrated ranking.

For every (modality, dataset, seed) cell, compute pseudo-AUC under a bank of
regimes = selection {smallest, random, hard} x K {30, 50}. The local analysis
(analyze_regimes.py) then aggregates the bank under several LABEL-FREE weighting
schemes and measures regret, testing whether auto-calibration fixes the text
weak spot without hurting the other modalities.

Launch:
  modal run scripts/modal_adrank_regimes.py
"""
import os
import modal

APP_NAME = "adrank-regimes"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
)
data_vol = modal.Volume.from_name("adrank-data", create_if_missing=False)
app = modal.App(APP_NAME)

SEEDS = [0, 1, 2, 3, 4]
REGIMES = [(sel, K) for sel in ("smallest", "random", "hard") for K in (30, 50)]
DATA_SUBDIR = {"cv": "cv", "nlp": "nlp", "tabular": "adbench"}
TS_NAMES = ["ts_point_spikes_0", "ts_point_spikes_1", "ts_point_spikes_2", "ts_point_spikes_3",
            "ts_subseq_0", "ts_subseq_1", "ts_trend", "ts_amplitude", "ts_freq_shift", "ts_mixed"]


@app.function(image=image, volumes={"/data": data_vol}, cpu=2.0, memory=8192, timeout=60 * 60)
def run_cell(spec):
    import sys; sys.path.insert(0, "/root")
    import numpy as np, pandas as pd
    from adrank.pipeline import Dataset, pseudo_auc_for_dataset, true_rank_from_labels

    tag, name, seed = spec
    if tag == "ts":
        from adrank.ts import load_synthetic_ts
        ds = next(d for d in load_synthetic_ts(seed=0) if d.name == name)
    else:
        arr = np.load(f"/data/{DATA_SUBDIR[tag]}/{name}.npz")
        X = np.asarray(arr["X"], dtype=np.float64)
        if not (200 <= X.shape[0] <= 50000):
            return {"skip": True}
        ds = Dataset(name=name, X=X, y=np.asarray(arr["y"], dtype=int).ravel())

    true_df = true_rank_from_labels(ds, seed=seed)
    true_df["seed"] = seed; true_df["modality"] = tag

    parts = []
    for sel, K in REGIMES:
        d = pseudo_auc_for_dataset(ds, K=K, M=20, selection=sel, seed=seed)
        d["regime"] = f"{sel}_K{K}"
        parts.append(d)
    pseudo_df = pd.concat(parts, ignore_index=True)
    pseudo_df["seed"] = seed; pseudo_df["modality"] = tag

    return {"true": true_df.to_dict("records"), "pseudo": pseudo_df.to_dict("records")}


@app.local_entrypoint()
def main():
    import pandas as pd
    specs = [("ts", nm, s) for nm in TS_NAMES for s in SEEDS]
    for tag in ["cv", "nlp", "tabular"]:
        names = sorted({f.path.split("/")[-1][:-4]
                        for f in data_vol.listdir(DATA_SUBDIR[tag]) if f.path.endswith(".npz")})
        specs += [(tag, nm, s) for nm in names for s in SEEDS]

    print(f"[adrank-regimes] launching {len(specs)} cells x {len(REGIMES)} regimes ...")
    true_rows, pseudo_rows, n = [], [], 0
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  FAILED: {res}"); continue
        if res.get("skip"):
            continue
        true_rows.extend(res["true"]); pseudo_rows.extend(res["pseudo"]); n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(specs)} cells done")
    os.makedirs("results/raw", exist_ok=True)
    pd.DataFrame(true_rows).to_parquet("results/raw/modal_true_regimes.parquet")
    pd.DataFrame(pseudo_rows).to_parquet("results/raw/modal_pseudo_regimes.parquet")
    print(f"[adrank-regimes] DONE: {n} cells, pseudo={len(pseudo_rows)} rows")
