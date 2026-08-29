"""Root-cause test for the text weak spot.

Hypothesis: real text anomalies need LOF's LOCAL density normalization, but
held-out COMPACT/FAR clusters are separable by global density (KNN) equally
well, so the pseudo-task flips LOF<->KNN. If we instead hold out HARD clusters
(closest to the normal mass, locally embedded), the pseudo-task should exercise
local structure and restore LOF's advantage.

Tests three pseudo-anomaly selection modes on the NLP datasets and reports, per
mode, the LOF-minus-KNN pseudo-AUC gap (true gap is +0.045) and the mean regret.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
os.environ["ADRANK_EXCLUDE_DETECTORS"] = "OCSVM"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from adrank.pipeline import (load_npz_dir, embed, cluster, cluster_composite_scores,
                             fit_and_score, detector_names, true_rank_from_labels)
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def select_clusters(Z, labels, scores, mode, rng):
    uniq = np.unique(labels)
    centroids = np.array([Z[labels == c].mean(0) for c in uniq])
    gcentroid = Z.mean(0)
    dist_to_center = np.linalg.norm(centroids - gcentroid, axis=1)
    sizes = np.array([(labels == c).sum() for c in uniq])
    if mode == "smallest":
        cand = uniq[np.argsort(sizes)[: max(2, len(uniq)//2)]]
    elif mode == "random":
        cand = uniq
    elif mode == "hard":  # closest clusters to global centroid (locally embedded)
        cand = uniq[np.argsort(dist_to_center)[: max(2, len(uniq)//2)]]
    return list(cand)


def pseudo_gap_and_regret(ds, mode, seed, K=30, M=20, target=0.05):
    Z = embed(ds.X, 16); labels = cluster(Z, K, seed)
    scores = cluster_composite_scores(Z, labels)
    rng = np.random.default_rng(seed)
    cand = select_clusters(Z, labels, scores, mode, rng)
    sizes = {c: (labels == c).sum() for c in np.unique(labels)}
    n = len(ds.X); tsize = max(20, int(target*n)); allidx = np.arange(n)
    det = detector_names()
    per_det = {d: [] for d in det}
    for j in range(M):
        rem = list(cand); rng.shuffle(rem); chosen = []; cur = 0
        for c in rem:
            if cur >= tsize: break
            chosen.append(c); cur += sizes[c]
        if not chosen: continue
        amask = np.isin(labels, chosen); comp = allidx[~amask]
        rng.shuffle(comp); nh = min(max(50, int(0.2*len(comp))), len(comp)//3)
        pnorm = np.sort(comp[:nh]); train = np.sort(comp[nh:]); panom = allidx[amask]
        Xtr = ds.X[train]; sidx = np.concatenate([pnorm, panom])
        y = np.concatenate([np.zeros(len(pnorm)), np.ones(len(panom))])
        for d in det:
            s = fit_and_score(d, Xtr, ds.X[sidx])
            per_det[d].append(np.nan if s is None else roc_auc_score(y, s))
    return {d: np.nanmean(v) if len(v) else np.nan for d, v in per_det.items()}


def main():
    ds_all = load_npz_dir(os.path.join(ROOT, "data", "nlp"))
    seeds = [0, 1, 2]
    rows = []
    for mode in ["smallest", "random", "hard"]:
        gaps, regrets = [], []
        for seed in seeds:
            for ds in ds_all:
                pa = pseudo_gap_and_regret(ds, mode, seed)
                tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc
                pick = max(pa, key=lambda d: (pa[d] if not np.isnan(pa[d]) else -1))
                regrets.append(tr.max() - tr[pick])
                if not (np.isnan(pa["LOF"]) or np.isnan(pa["KNN"])):
                    gaps.append(pa["LOF"] - pa["KNN"])
        rows.append({"mode": mode, "pseudo_LOF_minus_KNN": np.mean(gaps),
                     "mean_regret1": np.mean(regrets), "n": len(regrets)})
    r = pd.DataFrame(rows)
    print("true LOF-minus-KNN gap = +0.045 (target the pseudo-task should reproduce)\n")
    print(r.round(4).to_string(index=False))
    r.to_csv(os.path.join(ROOT, "results", "nlp_rootcause.csv"), index=False)


if __name__ == "__main__":
    main()
