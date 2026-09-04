# Streamlined HADB construction (ADOPTED as the primary benchmark, 2026-09-04)

**Date:** 2026-09-04. A single, principled, detector-free pipeline that rebuilds the benchmark FROM
SOURCE, replacing the accumulated ad-hoc filters (max|z|, Wu-Keogh, per-dataset rule, low-spread) of
the earlier build. **DECISION (2026-09-04): this streamlined 176-dataset benchmark is ADOPTED as the
paper's PRIMARY benchmark.** The earlier 199-dataset build is retained as the prior/robustness version
(the headline - NoMaS beats the UOMS field incl. EM/MV/IFOREST-R - reproduces on both, so the result is
not a filtering artifact). All code + result CSVs live in `experiments/streamline/`.

## Migration checklist (to finish promoting streamlined -> primary)
- [x] Selection pipeline + final set (`STREAM_FINAL2_SET.csv`, 176 datasets)
- [x] Full EDA (`STREAM_EDA2_ALL.csv`): sizes, diversity, marginal+joint multimodality, solvability, family
- [x] Selector leaderboard + all UOMS baselines incl. faithful IFOREST-R and UDR (`STREAM_RANK/IFR/UDR.csv`)
- [x] Pseudo-anomaly control (fails -> the win is OOD synthesis)
- [ ] Save per-detector ground-truth ap_norm on the hardened test as a canonical results CSV (rank.py
      currently keeps regrets only; regenerate a per-variant table for full reproducibility)
- [ ] Regenerate paper tables/figures from the streamlined numbers (the canonical FIG_leaderboard etc.
      become the prior-version appendix); primary leaderboard fig = `experiments/figs/FIG_stream_leaderboard.png`
- [ ] Zenodo/data-availability: package `STREAM_FINAL2_SET` + `STREAM_EDA2_ALL` as the released benchmark

## Pipeline (5 stages, all feature-space, no detector fitting)

Philosophy shift: instead of DROPPING whole datasets that look easy, HARDEN each dataset in place
(remove trivially-caught anomalies) and keep it if enough genuinely-hard, distinct anomalies remain.

1. **Stage 1 - drop OR-solvable datasets** (hardening rule catches >= 90% of anomalies).
2. **Stage 2a - marginal hardening**: per-original-feature OR, severity = max(ECDF two-sided tail,
   histogram -log-density) => edges + INTERIOR multimodal gaps; threshold at 5% normal FP.
3. **Stage 2b - joint hardening**: same OR in PCA-whitened space (95% var) => oblique/principal-direction
   separations the original axes miss; 5% normal FP. Applied as a SEPARATE stage, UNION with 2a
   (an anomaly is trivial if either flags it). NB merging 2a+2b into one rule and recalibrating to 5%
   FP BACKFIRES (multiple-comparison threshold inflation: catch 0.22, fixes 1/21 suspects); the
   sequential union is correct (catch 0.31, fixes 9/21).
4. **Stage 3 - keep >= 100 DISTINCT hard anomalies** (n_eff = greedy radius-cover at the normal NN
   scale). Distinct-count, not raw count: e.g. internet_firewall has 3028 hard anomalies but only 9
   distinct (near-duplicates) -> correctly dropped.
5. **Stage 4 - dedup** by data fingerprint (dim + sorted normal feature-moments); catches same-source
   datasets across corpora (mostly overlapping MTS series). adbench/dami vs oddbench/ovrbench are
   anomaly-MODIFIED variants, not identical, so correctly NOT merged.
6. **Stage 5 - keep >= 800 normals** (held-out) to model the normal distribution.
- **No base-rate cap**: anomalies need not be rare; a 1000-normal/1000-anomaly dataset is valid as long
  as normals model well and anomalies are enough+distinct for a robust ap_norm estimate.

## Result: 176 datasets (`STREAM_FINAL2_SET.csv`)

ovrbench 76, oddbench 46, MTS 43, adbench 7, dami 4 (tabular + multivariate-TS). Univariate TS (UCR,
tsbad_u) excluded: their single contiguous anomaly region yields <100 DISTINCT (mostly overlapping)
anomaly windows, so they cannot support robust evaluation.

EDA profile (`STREAM_EDA2_ALL.csv`, 176 datasets, all stats co-computed):
- **Well-modeled**: median 2400 normals (all >= 800).
- **Robustly evaluable**: median 371 distinct hard anomalies, eff_frac 0.54.
- **Solvable (validated by scoring the pool on the hardened sets)**: oracle ap_norm median 0.22, 71%
  > 0.1, only 6 unsolvable. Two-stage is HARDER than single-stage (was 0.33 / 82%) by design - it
  removes the oblique multivariate-trivial anomalies.
- **Family**: 101 local / 74 global (58/42), median base_rate 0.37.
- **Multimodal**: 0.71 of features marginally multimodal (Hartigan dip); 0.51 jointly multimodal
  (discriminant-axis dip); median 12 GMM BIC components. (Random-projection dip is a CLT artifact -
  do not use it; the discriminant-direction dip is the valid joint test.)

## Validation findings (figures in `experiments/figs/`)

- **Dropped-for-easy-separation** (OR_solvable, `FIG_stream_orsolvable.png`): anomalies form clearly
  separated clusters/regions - correctly removed.
- **Kept datasets with separable clusters** (`stream_suspects.py`): the marginal Stage-2a is blind to
  JOINT separation, so ~21/182 kept datasets have anomalies a simple kNN aces (multivariate-trivial;
  every feature in-band but the combination is far from normal). Stage-2b (PC-OR) catches ~half of
  these (survivor kNN 0.85 -> 0.55, 9-10/21 fixed). Residual ~12 hide in low-variance PCs or are
  nonlinearly separable; catching them would require a detector-defined rule, deliberately not done.
- **Dropped-but-embedded** anomalies were dropped for FEW-DISTINCT (n_eff<100), a count issue, not
  easiness (56/82 few-distinct drops are embedded/hard with frac_triv<0.3).

## Selector leaderboard + faithful UOMS baselines (streamlined benchmark, 175 datasets)

Construct-matched on the two-stage hardened test (`rank.py`, `ifr.py`, `udr.py`; regret on ap_norm,
macro = family-balanced):
- **NoMaS matched (ours) 0.149**, MV 0.148, **NoMaS beta=1 (ours) 0.151**, EM 0.159 - all beat random.
- **UDR 0.216** (seed-stability on the iForest sub-pool) - NOT significantly different from random
  (p=0.27), exactly Ma et al's finding. **IFOREST-R 0.218** (faithful 81-config average, n_estimators x
  max_features). consensus/ModelCentrality/HITS 0.222-0.225. random 0.228.
- **Our methods beat IFOREST-R and UDR at p<0.001** (matched beats IFOREST-R on 113/175 datasets).
This COUNTERS Ma et al ("none significantly different from random model selection ... all significantly
worse than random-config iForest"): on a hard-anomaly, local-vs-global benchmark with a diverse pool,
selection genuinely beats the iForest bar. UDR is ill-defined on the full mixed pool (deterministic
detectors LOF/KNN/HBOS/COPOD/ECOD/PCA have perfect seed-stability), so it is run faithfully on the
stochastic iForest sub-pool only.

**Pseudo-anomaly selection retried (`pseudo_cluster.py`) - fails again, confirming the mechanism.**
On the streamlined (strongly cluster-structured: 12 GMM modes) benchmark: smallest-cluster pseudo-
anomalies within-dataset rho -0.06 (no signal), EDGE pseudo-anomalies rho -0.20 (ANTI-correlated,
regret 0.233 > random), vs beta=1 synthetic rho +0.284 (regret 0.143). Holding out IN-distribution
normal points measures normal-structure separation, not anomaly detection - even on cluster-structured
data. The win is specifically OUT-OF-distribution synthesis, not any anomaly proxy. Result CSVs:
STREAM_RANK/IFR/UDR/PSEUDO.csv.

## Reproduce

`experiments/streamline/pipeline_final.py` (definitive selection) -> `STREAM_FINAL2_{ALL,SET}.csv`;
`score2.py` (solvability+family); `stream_multimodality.py` (marginal+joint modality); `rank.py`
(selector leaderboard). Analysis: `stream_suspects.py`, `stream_pca_or.py`, `stream_hist_bins.py`,
`stream_combined.py`/`stream_sequential.py`. Viz: `viz_grid.py`, `viz_dropped.py`, `make_eda.py`.
