"""Modal (llmcourse) FROZEN external evaluation on OddBench (reviewer W2/E2/W10).

OddBench (MacrOData-CMU, 690 public tabular AD datasets) was NOT used during any
ADRank development. We run the FROZEN current ADRank (smallest+random+hard regime
bank, K in {30,50}, discriminative variance weighting, M=20, PCA-16, MiniBatch
KMeans) unchanged, and report regret@1 vs random. Deterministic dataset selection
(fixed rule, not performance): 200 <= n <= 50000, feature dim <= 500, anomaly rate
in [0.005, 0.35], sorted by name, first N. Each container fetches its dataset from
HuggingFace and commits its result to a volume (survives client death).
"""
import modal

APP_NAME = "adrank-oddbench"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
)
results_vol = modal.Volume.from_name("adrank-oddbench-hash-results", create_if_missing=True)
app = modal.App(APP_NAME)

# Hash-sampled (name-independent) draw, NOT alphabetical: the first-200-by-name
# draw covered only A-E and over-weighted semantic-label 'Campaign*/Customer*'
# datasets. We sort by sha1(name) and take the first POOL, then the per-cell
# filter drops unsuitable ones, yielding ~180 usable across the full alphabet.
POOL = 260
SEEDS = [0, 1, 2]
KS = [30, 50]
SELS = ["smallest", "random", "hard"]   # FROZEN bank
M = 20
HF = "https://huggingface.co/datasets/MacrOData-CMU/OddBench/resolve/main/public/"


def _load_oddbench(name):
    import io, urllib.request, numpy as np
    d = np.load(io.BytesIO(urllib.request.urlopen(HF + name + "?download=true", timeout=120).read()),
                allow_pickle=True)
    tr, trl = np.asarray(d["train"], dtype=np.float64), np.asarray(d["train_labels"]).ravel()
    te, tel = np.asarray(d["test"], dtype=np.float64), np.asarray(d["test_labels"]).ravel()
    X = np.vstack([tr, te]); y = np.concatenate([trl, tel]).astype(int)
    return X, y


@app.function(image=image, volumes={"/results": results_vol}, cpu=2.0, memory=8192, timeout=60 * 60)
def run_cell(name):
    import sys, os; sys.path.insert(0, "/root")
    import numpy as np, pandas as pd
    from adrank.pipeline import Dataset, pseudo_auc_for_dataset, true_rank_from_labels
    outp = f"/results/{name}.parquet"
    if os.path.exists(outp):
        return pd.read_parquet(outp).to_dict("records")
    try:
        X, y = _load_oddbench(name)
    except Exception as e:
        return [{"dataset": name, "status": f"fetch_fail:{type(e).__name__}"}]
    n, d = X.shape; rate = float((y == 1).mean())
    if not (200 <= n <= 50000 and d <= 500 and 0.005 <= rate <= 0.35):
        return [{"dataset": name, "status": "filtered", "n": n, "d": d, "rate": round(rate, 4)}]
    # replace non-finite, dedup-safe
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ds = Dataset(name=name, X=X, y=y)

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

    import json
    def regime_vars(pp):
        out = {}
        for r in pp.regime.unique():
            v = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
            out[r] = float(v.var())
        return out

    rows = []
    for seed in SEEDS:
        tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc.dropna()
        if len(tr) < 3:
            continue
        parts = []
        for K in KS:
            for sel in SELS:
                p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=seed)
                p["regime"] = f"{sel}_K{K}"; parts.append(p)
        bank = pd.concat(parts, ignore_index=True)
        score = agg_disc(bank); score = score[score.index.isin(tr.index)]
        if score.empty:
            continue
        pick = score.sort_values(ascending=False).index[0]
        # persist per-detector tables + regime variances so the abstention gate
        # (fall back to a robust default when no regime shows cross-detector
        # spread) can be tested offline without re-running.
        rvars = regime_vars(bank)
        rows.append({"dataset": name, "seed": seed, "n": n, "d": d, "rate": round(rate, 4),
                     "status": "ok",
                     "regret": float(tr.max() - tr[pick]),
                     "random_regret": float(tr.max() - tr.mean()),
                     "best_regret_possible": 0.0,
                     "pick": str(pick),
                     "max_regime_var": float(max(rvars.values()) if rvars else 0.0),
                     "score_json": json.dumps({k: float(v) for k, v in score.items()}),
                     "true_auc_json": json.dumps({k: float(v) for k, v in tr.items()}),
                     "regime_var_json": json.dumps(rvars)})
    if not rows:
        rows = [{"dataset": name, "status": "no_valid_seed"}]
    pd.DataFrame(rows).to_parquet(outp); results_vol.commit()
    return rows


@app.local_entrypoint()
def main():
    import io, json, hashlib, urllib.request, pandas as pd, os
    api = "https://huggingface.co/api/datasets/MacrOData-CMU/OddBench/tree/main/public"
    files = [e["path"].split("/")[-1] for e in
             json.loads(urllib.request.urlopen(api, timeout=60).read())
             if e["path"].endswith(".npz")]
    # hash-sample: name-independent, spans the whole alphabet (fixes A-E bias)
    subset = sorted(files, key=lambda n: hashlib.sha1(n.encode()).hexdigest())[:POOL]
    print(f"[oddbench] {len(files)} public datasets; running frozen ADRank on {len(subset)} "
          f"hash-sampled (per-cell filter drops unsuitable) ...")
    allrows = []
    for res in run_cell.map(subset, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("  FAILED:", res); continue
        allrows.extend(res)
    df = pd.DataFrame(allrows)
    os.makedirs("results", exist_ok=True); df.to_csv("results/oddbench.csv", index=False)
    ok = df[df.status == "ok"] if "status" in df else df
    if len(ok):
        per = ok.groupby("dataset").agg(regret=("regret", "mean"), rnd=("random_regret", "mean"))
        print(f"\n[oddbench] {per.shape[0]} datasets evaluated")
        print(f"  mean regret@1 = {per.regret.mean():.4f}  |  random = {per.rnd.mean():.4f}")
        print(f"  reduction vs random = {100*(per.rnd.mean()-per.regret.mean())/per.rnd.mean():.0f}%")
        print(f"  datasets where ADRank beats random: {(per.regret < per.rnd).sum()}/{per.shape[0]}")
    print("[oddbench] DONE -> results/oddbench.csv")
