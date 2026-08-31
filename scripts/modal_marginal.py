"""B4: does a MARGINAL-TAIL pseudo-anomaly regime close the global-best ceiling?

The cluster-holdout bank is local, so it misses datasets whose true-best detector
is a marginal/global method (HBOS/PCA at AUC ~1.0 on many OddBench losers). We add
a marginal-tail regime (pseudo-anomalies = points in the extreme tail of a single
raw feature) and test, on the 53 OddBench losers + a control set of winners:
  (a) truth-alignment: does marginal-tail pseudo-AUC rank the detectors correctly?
  (b) does adding it to the discriminative bank help losers WITHOUT hurting winners?
Frozen everything else. Results committed to a volume.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
)
vol = modal.Volume.from_name("adrank-marginal-results", create_if_missing=True)
app = modal.App("adrank-marginal")
HF = "https://huggingface.co/datasets/MacrOData-CMU/OddBench/resolve/main/public/"
KS, SELS, M = [30, 50], ["smallest", "random", "hard"], 20
MTAIL_FRAC = 0.05


def _load(name):
    import io, urllib.request, numpy as np
    d = np.load(io.BytesIO(urllib.request.urlopen(HF + name + "?download=true", timeout=120).read()),
                allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    return np.nan_to_num(X), y


@app.function(image=image, volumes={"/results": vol}, cpu=2.0, memory=8192, timeout=60 * 60)
def run_cell(spec):
    import sys, os; sys.path.insert(0, "/root")
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    name, group = spec
    outp = f"/results/{name}.parquet"
    if os.path.exists(outp):
        return pd.read_parquet(outp).to_dict("records")
    from adrank.pipeline import (Dataset, pseudo_auc_for_dataset, true_rank_from_labels,
                                 detector_names, fit_and_score)
    try:
        X, y = _load(name)
    except Exception as e:
        return [{"dataset": name, "status": f"fetch_fail:{type(e).__name__}"}]
    n, d = X.shape
    if not (200 <= n <= 50000 and d <= 500 and 0.005 <= (y == 1).mean() <= 0.35):
        return [{"dataset": name, "status": "filtered"}]
    ds = Dataset(name=name, X=X, y=y)
    dets = detector_names(d)

    def marginal_tail_bank(seed):
        """M draws: each picks a random feature+direction, pseudo-anom = extreme
        MTAIL_FRAC tail on that feature, pseudo-norm = holdout of the rest."""
        rng = np.random.default_rng(seed); rows = []
        allidx = np.arange(n)
        for j in range(M):
            f = int(rng.integers(d)); col = X[:, f]
            if rng.random() < 0.5:
                thr = np.quantile(col, 1 - MTAIL_FRAC); anom = allidx[col >= thr]
            else:
                thr = np.quantile(col, MTAIL_FRAC); anom = allidx[col <= thr]
            if len(anom) < 5 or len(anom) > n // 2:
                continue
            comp = allidx[~np.isin(allidx, anom)]; rng.shuffle(comp)
            nh = min(max(50, len(comp) // 5), len(comp) // 3)
            pnorm, train = np.sort(comp[:nh]), np.sort(comp[nh:])
            Xsc = X[np.concatenate([pnorm, anom])]
            yp = np.concatenate([np.zeros(len(pnorm)), np.ones(len(anom))])
            for det in dets:
                s = fit_and_score(det, X[train], Xsc)
                auc = np.nan if (s is None or len(np.unique(yp)) < 2) else roc_auc_score(yp, s)
                rows.append({"detector": det, "subset": j, "regime": "mtail", "pseudo_auc": auc})
        return pd.DataFrame(rows)

    def agg_disc(pp):
        ms = pp.dropna(subset=["pseudo_auc"]).groupby(["regime", "detector"]).pseudo_auc.mean()
        w = {}
        for r in pp.regime.unique():
            v = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
            w[r] = float(v.var())
        sden = sum(w.values()) or 1.0; w = {r: w[r] / sden for r in w}
        out = pd.Series(0.0, index=ms.index.get_level_values(1).unique()); ws = 0.0
        for r in ms.index.get_level_values(0).unique():
            out = out.add(ms.loc[r].rank(ascending=True) * w.get(r, 0.0), fill_value=0.0); ws += w.get(r, 0.0)
        return out / (ws or 1.0)

    rows = []
    for seed in [0, 1, 2]:
        tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc.dropna()
        if len(tr) < 3:
            continue
        parts = []
        for K in KS:
            for sel in SELS:
                p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=seed)
                p["regime"] = f"{sel}_K{K}"; parts.append(p)
        clbank = pd.concat(parts, ignore_index=True)
        mt = marginal_tail_bank(seed)
        augbank = pd.concat([clbank, mt], ignore_index=True)

        def pick_regret(bank):
            sc = agg_disc(bank); sc = sc[sc.index.isin(tr.index)]
            if sc.empty:
                return np.nan
            return float(tr.max() - tr[sc.sort_values(ascending=False).index[0]])
        # marginal-only truth alignment
        mtscore = mt.dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
        mtscore = mtscore[mtscore.index.isin(tr.index)]
        rho = spearmanr(tr.reindex(mtscore.index).rank(), mtscore.rank()).correlation if len(mtscore) > 2 else np.nan
        mt_pick_regret = float(tr.max() - tr[mtscore.sort_values(ascending=False).index[0]]) if not mtscore.empty else np.nan
        rows.append({"dataset": name, "group": group, "seed": seed, "status": "ok",
                     "regret_cluster": pick_regret(clbank),
                     "regret_aug": pick_regret(augbank),
                     "regret_mtonly": mt_pick_regret,
                     "mt_rho": rho,
                     "random_regret": float(tr.max() - tr.mean())})
    if not rows:
        rows = [{"dataset": name, "status": "no_seed"}]
    pd.DataFrame(rows).to_parquet(outp); vol.commit()
    return rows


@app.local_entrypoint()
def main():
    import pandas as pd, os
    losers = [l.strip() for l in open("E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/losers.txt") if l.strip()]
    winners = [l.strip() for l in open("E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/winners.txt") if l.strip()]
    specs = [(n, "loser") for n in losers] + [(n, "winner") for n in winners]
    print(f"[marginal] {len(specs)} datasets ({len(losers)} losers + {len(winners)} winners)")
    allrows = []
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("  FAIL", res); continue
        allrows.extend(res)
    df = pd.DataFrame(allrows); os.makedirs("results", exist_ok=True)
    df.to_csv("results/marginal.csv", index=False)
    print("[marginal] DONE -> results/marginal.csv")
