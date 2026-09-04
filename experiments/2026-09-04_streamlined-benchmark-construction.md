# Streamlined HADB construction (side project, isolated)

**Date:** 2026-09-04. A single, principled, detector-free pipeline that rebuilds the benchmark FROM
SOURCE, replacing the accumulated ad-hoc filters (max|z|, Wu-Keogh, per-dataset rule, low-spread) of
the canonical build. **This is an alternative construction; the canonical 199-dataset benchmark and its
results are untouched.** All code + result CSVs live in `experiments/streamline/`.

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

## Reproduce

`experiments/streamline/pipeline_final.py` (definitive selection) -> `STREAM_FINAL2_{ALL,SET}.csv`;
`score2.py` (solvability+family); `stream_multimodality.py` (marginal+joint modality); `rank.py`
(selector leaderboard). Analysis: `stream_suspects.py`, `stream_pca_or.py`, `stream_hist_bins.py`,
`stream_combined.py`/`stream_sequential.py`. Viz: `viz_grid.py`, `viz_dropped.py`, `make_eda.py`.
