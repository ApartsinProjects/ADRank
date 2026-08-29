"""A/B: does VaDE (joint deep embedding + clustering) beat PCA-16 + MiniBatchKMeans
inside ADRank, on the datasets where the current clustering hits the local-detector
ceiling?

For each dataset and seed, build the regime bank {smallest,random,hard}x{K30,K50}
two ways -- default PCA+KMeans, and VaDE-precomputed (Z,labels) -- aggregate with
the discriminative weighting, and compare regret@1 against the true ranking.
Same detectors, same regimes, same metric; only the latent/clustering differs.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adrank.pipeline import load_npz_dir, pseudo_auc_for_dataset, true_rank_from_labels
from analyze_regimes import _aggregate_rank
from vade import fit_vade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
KS = [30, 50]
SELS = ["smallest", "random", "hard"]
M = 10

# ceiling-fail (global-best detector) + controls (local-best, ADRank already strong)
TARGETS = {
    "dami":    ["Annthyroid", "WDBC", "Cardiotocography", "PageBlocks"],
    "adbench": ["39_vertebral", "23_mammography", "40_vowels", "6_cardio"],
}


def bank_regret(ds, seed, method):
    parts = []
    for K in KS:
        pre = None
        if method == "vade":
            Z, lab = fit_vade(ds.X, K, seed=seed)
            pre = (Z, lab)
        for sel in SELS:
            p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=seed, precomputed=pre)
            p["regime"] = f"{sel}_K{K}"
            parts.append(p)
    bank = pd.concat(parts, ignore_index=True)
    tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc.dropna()
    score = _aggregate_rank(bank, "discriminative")
    score = score[score.index.isin(tr.index)]
    if score.empty:
        return np.nan
    pick = score.sort_values(ascending=False).index[0]
    return tr.max() - tr[pick]


def main():
    rows = []
    for src, names in TARGETS.items():
        folder = "dami" if src == "dami" else "adbench"
        dss = {d.name: d for d in load_npz_dir(os.path.join(ROOT, "data", folder))}
        for nm in names:
            if nm not in dss:
                print(f"  skip {nm} (not found)"); continue
            ds = dss[nm]
            for method in ["pca", "vade"]:
                reg = []
                for s in SEEDS:
                    t = time.time()
                    try:
                        r = bank_regret(ds, s, method)
                    except Exception as e:
                        r = np.nan
                        print(f"  {nm}/{method}/seed{s}: FAILED {type(e).__name__}", flush=True)
                    if not np.isnan(r):
                        reg.append(r)
                        print(f"  {nm}/{method}/seed{s}: regret={r:.3f} ({time.time()-t:.0f}s)", flush=True)
                if not reg:
                    continue
                rows.append({"dataset": nm, "src": src, "method": method,
                             "regret1": float(np.mean(reg)), "std": float(np.std(reg))})
                pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "audit_vade_ab.csv"), index=False)
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["src", "dataset"], columns="method", values="regret1")
    piv["delta_vade_minus_pca"] = piv["vade"] - piv["pca"]
    print("\n=== regret@1: PCA+KMeans vs VaDE (negative delta = VaDE better) ===")
    print(piv.round(3).to_string())
    print(f"\nmean regret  PCA={df[df.method=='pca'].regret1.mean():.3f}  VaDE={df[df.method=='vade'].regret1.mean():.3f}")
    piv.to_csv(os.path.join(ROOT, "results", "audit_vade_pivot.csv"))


if __name__ == "__main__":
    main()
