"""Modal fan-out for ADRank cross-modality confidence intervals.

Fans out every (modality, dataset, seed) cell across Modal CPU containers in
parallel. TS data is generated on the container (synthetic); CV/NLP npz come
from the `adrank-data` volume. Each cell returns its true-AUC and pseudo-AUC
records; the local entrypoint concatenates them into parquet files.

Config per modality (matches the local seed-0 config):
  ts, cv -> selections [smallest, random];  nlp -> [smallest].  K=30, M=20.
Seeds 0..4 computed fresh on Modal for a fully consistent 5-seed set.
OCSVM excluded (libsvm multi-worker crash).

Launch:
  python C:/Users/apart/.claude/skills/gpu2modal/modal_runner.py run \
      --script scripts/modal_adrank_ci.py --detach
"""
import os
import modal

APP_NAME = "adrank-ci"

image = (
    # Pin to match the local stack exactly; a pyod/sklearn mismatch silently
    # NaNs out the sklearn-based detectors (only COPOD/ECOD survive otherwise).
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
)

data_vol = modal.Volume.from_name("adrank-data", create_if_missing=False)
app = modal.App(APP_NAME)

SEEDS = [0, 1, 2, 3, 4]
CONFIG = {
    "ts": ["smallest", "random"],
    "cv": ["smallest", "random"],
    "nlp": ["smallest"],
    "tabular": ["smallest", "random"],
}
# tabular npz live under /data/adbench ; cv under /data/cv ; nlp under /data/nlp
DATA_SUBDIR = {"cv": "cv", "nlp": "nlp", "tabular": "adbench"}


def _build_specs():
    """Return list of (tag, dataset_name, seed). Dataset names are resolved on
    the container for ts (synthetic) and passed as npz stems for cv/nlp."""
    import numpy as np  # noqa
    specs = []
    # ts dataset names are fixed by ts.load_synthetic_ts
    ts_names = [
        "ts_point_spikes_0", "ts_point_spikes_1", "ts_point_spikes_2", "ts_point_spikes_3",
        "ts_subseq_0", "ts_subseq_1", "ts_trend", "ts_amplitude", "ts_freq_shift", "ts_mixed",
    ]
    for nm in ts_names:
        for s in SEEDS:
            specs.append(("ts", nm, s))
    # cv / nlp resolved from the volume listing (done in entrypoint)
    return specs


@app.function(image=image, volumes={"/data": data_vol}, cpu=2.0,
              memory=8192, timeout=60 * 60)
def run_cell(spec):
    """Run one (tag, dataset, seed) cell. Returns dict with 'true' and 'pseudo' record lists."""
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import pandas as pd
    from adrank.pipeline import (
        Dataset, pseudo_auc_for_dataset, true_rank_from_labels,
    )

    tag, name, seed = spec
    selections = CONFIG[tag]

    # load dataset
    if tag == "ts":
        from adrank.ts import load_synthetic_ts
        ds = next(d for d in load_synthetic_ts(seed=0) if d.name == name)
    else:
        arr = np.load(f"/data/{DATA_SUBDIR[tag]}/{name}.npz")
        X = np.asarray(arr["X"], dtype=np.float64)
        n = X.shape[0]
        if not (200 <= n <= 50000):  # same size filter as the paper's 26-set
            return {"tag": tag, "dataset": name, "seed": seed, "skip": True}
        ds = Dataset(name=name, X=X, y=np.asarray(arr["y"], dtype=int).ravel())

    true_df = true_rank_from_labels(ds, seed=seed)
    true_df["seed"] = seed
    true_df["modality"] = tag

    parts = []
    for sel in selections:
        d = pseudo_auc_for_dataset(ds, K=30, M=20, selection=sel, seed=seed)
        parts.append(d)
    pseudo_df = pd.concat(parts, ignore_index=True)
    pseudo_df["seed"] = seed
    pseudo_df["modality"] = tag

    return {
        "tag": tag, "dataset": name, "seed": seed,
        "true": true_df.to_dict("records"),
        "pseudo": pseudo_df.to_dict("records"),
    }


@app.local_entrypoint()
def main():
    import pandas as pd

    # resolve cv/nlp dataset stems from the volume
    specs = _build_specs()
    for tag in ["cv", "nlp", "tabular"]:
        subdir = DATA_SUBDIR[tag]
        names = sorted({
            f.path.split("/")[-1][:-4]
            for f in data_vol.listdir(subdir)
            if f.path.endswith(".npz")
        })
        for nm in names:
            for s in SEEDS:
                specs.append((tag, nm, s))

    print(f"[adrank-ci] launching {len(specs)} cells across Modal containers ...")

    true_rows, pseudo_rows = [], []
    n_done = 0
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  cell FAILED: {res}")
            continue
        if res.get("skip"):
            continue
        true_rows.extend(res["true"])
        pseudo_rows.extend(res["pseudo"])
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(specs)} cells done")

    true_all = pd.DataFrame(true_rows)
    pseudo_all = pd.DataFrame(pseudo_rows)
    os.makedirs("results/raw", exist_ok=True)
    true_all.to_parquet("results/raw/modal_true_allseeds.parquet")
    pseudo_all.to_parquet("results/raw/modal_pseudo_allseeds.parquet")
    print(f"[adrank-ci] DONE: {n_done} cells, "
          f"true={len(true_all)} rows, pseudo={len(pseudo_all)} rows")
    print("  wrote results/raw/modal_true_allseeds.parquet and modal_pseudo_allseeds.parquet")
