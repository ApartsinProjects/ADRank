# ADRank v1

Unsupervised ranking of anomaly detection models via cluster-holdout pseudo-labels.

## Hypothesis

For each detector `D`, cluster the unlabeled data, hide a subset `S` of clusters,
train `D` on the complement, and score points in `S` as pseudo-anomalies. Then the
ranking of detectors by pseudo-AUC should correlate with their ranking by true AUC
on real anomalies.

## Validation plan

Datasets from ADBench (public benchmarks with real labels). Labels are used **only**
to compute the ground-truth ranking; the pseudo-ranking pipeline never sees them.
For each dataset we produce a rank correlation (Spearman, Kendall) between
pseudo-rank and true-rank, plus top-1 / top-3 hit rates.

- `src/adrank/pipeline.py` — datasets, detectors, clustering, pseudo-eval, aggregation
- `scripts/fetch_adbench.py` — downloads ADBench Classical `.npz` files into `data/adbench/`
- `scripts/run_v1.py` — main experiment runner (parallel across dataset x subset)
- `scripts/aggregate_report.py` — Borda / mean-AUC / var-weighted aggregation + correlations
- `scripts/smoke_test.py` — 5-dataset synthetic smoke test (~30 s)

## Fixed decisions (do not tune per dataset)

- Latent embedding: PCA to 16 dims (or `d` if `d < 16`).
- Clustering: MiniBatchKMeans with `K ∈ {30, 50}` (sweep).
- Cluster-composite anomaly-likeness score: `1/3 * (1 - size/N) + 1/3 * mean-dist-to-k-nearest-centroids + 1/3 * (1 - local-density)`, each min-max normalized within dataset.
- Subset draws: `M = 20` per (dataset, K). Each subset targets ~5% of `N` as pseudo-anomalies, drawn from the top-quartile of the composite score. Held-out normal fold is 20% of the training complement.
- Detectors: `IForest, LOF, KNN, OCSVM, ECOD, COPOD, HBOS, PCA, CBLOF, LODA` with PyOD defaults, fixed seed.
- Aggregations reported: Borda-over-ranks, mean pseudo-AUC, variance-weighted mean.

## Preregistered success criterion

Mean Spearman ρ ≥ 0.6 across ≥ 30 datasets, and top-1 hit-rate ≥ 40%
(random baseline ≈ 1/10 = 10%).

## Sanity checks (must pass before believing anything)

1. Scrambled pseudo-scores → ρ near 0.
2. Trivial "all-same" pseudo-score baseline (mean intra-cluster density, detector-independent) should NOT match the pseudo-ranking's correlation. If it does, the signal is cluster geometry, not detector behavior.
3. Random cluster-subset selection (uniform, not composite-weighted) should give strictly lower correlation than the composite-weighted selection — otherwise selection does not matter and we defend the simpler variant.
