"""Regime sweep on the DAMI benchmark (second tabular family) via Modal.

Same six-regime bank and classical 9-detector panel as modal_adrank_regimes.py,
applied to the DAMI datasets (uploaded to the volume under /data/dami). Filters
to AD-typical outlier rate <= 0.35 and size 200..50000. Writes
results/raw/modal_{true,pseudo}_dami.parquet.
"""
import os
import modal

APP_NAME = "adrank-dami"
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


@app.function(image=image, volumes={"/data": data_vol}, cpu=2.0, memory=8192, timeout=60 * 60)
def run_cell(spec):
    import sys; sys.path.insert(0, "/root")
    import numpy as np, pandas as pd
    from adrank.pipeline import Dataset, pseudo_auc_for_dataset, true_rank_from_labels

    name, seed = spec
    arr = np.load(f"/data/dami/{name}.npz")
    X = np.asarray(arr["X"], dtype=np.float64); y = np.asarray(arr["y"], dtype=int).ravel()
    n = X.shape[0]
    if not (200 <= n <= 50000) or y.mean() > 0.35:  # AD-typical rate + size filter
        return {"skip": True}
    ds = Dataset(name=name, X=X, y=y)
    true_df = true_rank_from_labels(ds, seed=seed); true_df["seed"] = seed; true_df["modality"] = "dami"
    parts = []
    for sel, K in REGIMES:
        d = pseudo_auc_for_dataset(ds, K=K, M=20, selection=sel, seed=seed)
        d["regime"] = f"{sel}_K{K}"; parts.append(d)
    pseudo_df = pd.concat(parts, ignore_index=True); pseudo_df["seed"] = seed; pseudo_df["modality"] = "dami"
    return {"true": true_df.to_dict("records"), "pseudo": pseudo_df.to_dict("records")}


@app.local_entrypoint()
def main():
    import pandas as pd
    names = sorted({f.path.split("/")[-1][:-4] for f in data_vol.listdir("dami") if f.path.endswith(".npz")})
    specs = [(nm, s) for nm in names for s in SEEDS]
    print(f"[adrank-dami] launching {len(specs)} cells ({len(names)} datasets) x {len(REGIMES)} regimes ...")
    true_rows, pseudo_rows, n = [], [], 0
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  FAILED: {res}"); continue
        if res.get("skip"):
            continue
        true_rows.extend(res["true"]); pseudo_rows.extend(res["pseudo"]); n += 1
    os.makedirs("results/raw", exist_ok=True)
    pd.DataFrame(true_rows).to_parquet("results/raw/modal_true_dami.parquet")
    pd.DataFrame(pseudo_rows).to_parquet("results/raw/modal_pseudo_dami.parquet")
    print(f"[adrank-dami] DONE: {n} cells kept, pseudo={len(pseudo_rows)} rows")
