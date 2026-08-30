"""Modal (llmcourse) contamination experiment (reviewer W1/E1).

Does ADRank rank detectors when its SELECTION input contains zero genuine
anomalies? For each ADBench Classical tabular dataset, ADRank's selection set is
(all normals) + (fraction c of the real anomalies), unlabeled; the TRUE ranking is
computed by the fixed protocol (train on 80% normals, evaluate on 20% normals +
all anomalies) and does NOT depend on c. We sweep c and measure regret@1.

c = 0.0 is the key point: ADRank sees pure normals. A flat regret-vs-c curve means
the method does not rely on hidden real anomalies -- converting the title's claim
from a vulnerability into a strength.
"""
import modal

APP_NAME = "adrank-contam"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
)
results_vol = modal.Volume.from_name("adrank-contam-results", create_if_missing=True)
app = modal.App(APP_NAME)

CLASSICAL = ["6_cardio","14_glass","18_Ionosphere","19_landsat","20_letter","21_Lymphography",
    "22_magic.gamma","23_mammography","25_musk","26_optdigits","27_PageBlocks","28_pendigits",
    "29_Pima","30_satellite","31_satimage-2","35_SpamBase","36_speech","37_Stamps","38_thyroid",
    "39_vertebral","40_vowels","41_Waveform","42_WBC","43_WDBC","44_Wilt","47_yeast"]
CONTAM = [0.0, 0.005, 0.01, 0.02, 0.05, -1.0]  # -1.0 = natural benchmark rate
SEEDS = [0, 1, 2]
KS = [30, 50]
SELS = ["smallest", "random"]
M = 20


@app.function(image=image, volumes={"/results": results_vol}, cpu=2.0, memory=8192, timeout=60 * 60)
def run_cell(name):
    import sys, os; sys.path.insert(0, "/root")
    import io, urllib.request
    import numpy as np, pandas as pd
    from adrank.pipeline import Dataset, pseudo_auc_for_dataset, true_rank_from_labels
    # resilient: skip if already computed (idempotent resume)
    outp = f"/results/{name}.parquet"
    if os.path.exists(outp):
        return pd.read_parquet(outp).to_dict("records")

    url = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/" + name + ".npz"
    arr = np.load(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    X = np.asarray(arr["X"], dtype=np.float64); y = np.asarray(arr["y"], dtype=int).ravel()
    if not (200 <= X.shape[0] <= 50000):
        return []
    Xn, Xa = X[y == 0], X[y == 1]
    ds_full = Dataset(name=name, X=X, y=y)

    def agg_disc(pp):
        ms = pp.dropna(subset=["pseudo_auc"]).groupby(["regime", "detector"]).pseudo_auc.mean()
        regs = pp.regime.unique(); w = {}
        for r in regs:
            v = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
            w[r] = float(v.var())
        sden = sum(w.values()) or 1.0; w = {r: w[r] / sden for r in regs}
        dets = ms.index.get_level_values(1).unique()
        out = pd.Series(0.0, index=dets); ws = 0.0
        for r in ms.index.get_level_values(0).unique():
            out = out.add(ms.loc[r].rank(ascending=True) * w.get(r, 0.0), fill_value=0.0); ws += w.get(r, 0.0)
        return out / (ws or 1.0)

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # TRUE ranking is fixed (does not depend on c)
        tr = true_rank_from_labels(ds_full, seed=seed).set_index("detector").true_auc.dropna()
        for c in CONTAM:
            if c < 0:                       # natural: use the full X as-is
                Xsel = X
            else:
                k = int(round(c * len(Xn))) # inject c*|normals| real anomalies
                k = min(k, len(Xa))
                idx = rng.choice(len(Xa), size=k, replace=False) if k > 0 else np.array([], dtype=int)
                Xsel = np.vstack([Xn, Xa[idx]]) if k > 0 else Xn
            ds_sel = Dataset(name=name, X=Xsel, y=np.zeros(len(Xsel), dtype=int))
            parts = []
            for K in KS:
                for sel in SELS:
                    p = pseudo_auc_for_dataset(ds_sel, K=K, M=M, selection=sel, seed=seed)
                    p["regime"] = f"{sel}_K{K}"; parts.append(p)
            bank = pd.concat(parts, ignore_index=True)
            score = agg_disc(bank); score = score[score.index.isin(tr.index)]
            if score.empty:
                continue
            pick = score.sort_values(ascending=False).index[0]
            rows.append({"dataset": name, "seed": seed,
                         "contam": (Xa.shape[0] / X.shape[0]) if c < 0 else c,
                         "contam_label": "natural" if c < 0 else f"{c:.3f}",
                         "regret": float(tr.max() - tr[pick]),
                         "random_regret": float(tr.max() - tr.mean())})
    if rows:
        pd.DataFrame(rows).to_parquet(outp)   # persist to volume; survives client death
        results_vol.commit()
    return rows


@app.local_entrypoint()
def main():
    import pandas as pd, os
    print(f"[contam] launching {len(CLASSICAL)} datasets ...")
    allrows = []
    for res in run_cell.map(CLASSICAL, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("  FAILED:", res); continue
        allrows.extend(res)
    df = pd.DataFrame(allrows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/contamination.csv", index=False)
    # summarize: mean regret per contamination label
    g = df.groupby("contam_label").regret.agg(["mean", "std", "count"]).round(4)
    print("\n[contam] mean regret@1 by contamination of ADRank's selection input:")
    print(g.to_string())
    print(f"\n  random-pick regret (reference): {df.random_regret.mean():.4f}")
    print("[contam] DONE -> results/contamination.csv")
