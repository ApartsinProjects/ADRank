# HADB - Hard Anomaly-detection Benchmark for unsupervised model selection

A benchmark for **label-free selection of anomaly detectors** where trivial anomalies and
low-spread datasets are filtered out, leaving only datasets on which choosing the right
detector, without labels, genuinely matters.

## Composition (three-way, leak-free build; per-dataset rule filter all modalities, 2026-09-04)

- **199 datasets**, effective N = 197 after dedup; two modalities.
- Tabular 116 (adbench+dami, OddBench, OvrBench) + time series 83 (TSB-AD-U, UCR, TSB-AD-M).
- True-best family: 136 local / 52 global / 11 other (global-vs-local selection preserved).
- (Earlier builds: 292 before the per-dataset triviality-rule filter; 223 when it was tabular-only.)
- Protocol: normals split 60/20/20 **train / validation / test**, disjoint in raw points
  (time series: contiguous-block split + overlap purge). Detectors fit on TRAIN; label-free
  selectors compute their criterion on VALIDATION (normals only); ground-truth AUC/AP/ap_norm
  on TEST (held-out normals + hard anomalies). No selector ever sees the set it is scored on.
- Primary metric: `ap_norm = (AP - base_rate)/(1 - base_rate)`.
- Filters: Wu & Keogh one-liner triviality (time series, permutation-calibrated for
  multivariate), max|z| + LinRes double-hard triviality (tabular, per-anomaly), base-rate cap
  0.25, mixed-label duplicate removal, floor (best < 0.10) and low-spread (< 0.10) exclusion,
  and a **per-dataset triviality-rule filter** across ALL modalities: drop datasets whose
  surviving hard anomalies are still separable by a simple per-feature rule - max|z| (Gaussian)
  OR HBOS-lite (empirical histogram, catches skewed/bimodal/categorical rarity the Gaussian
  misses) - at test AUC > 0.85, computed in the space the detectors use. Tabular:
  `hadb_trivial_rules.py`. Time series (uni + multivariate, on window features):
  `ts_trivial_rules.py`. This closes a gap where the per-anomaly / raw-signal filters let
  through datasets a simple RULE still solves in feature space (NetworkFlow max|z| AUC 0.98;
  several OPPORTUNITY series 0.98-1.00). 97 datasets removed (73 tabular + 24 TS).

## Layout

- `build_hadb_v3.py`, `hadb_oddbench.py`, `hadb_ovrbench.py` - tabular arms.
- `hadb_ts_final.py` (UCR + TSB-AD-U), `hadb_ts_mts.py` (multivariate) - time-series arms.
- `ts_triviality.py` - calibrated permutation triviality (multivariate); `keogh_calibrate.py`.
- `label_free_criteria.py` - EM/MV/consensus/ModelCentrality/HITS; `hadb_round2_common.py` -
  the three-way evaluator + NoMaS + score saving.
- `hadb_consolidate.py` -> `results/HADB_MANIFEST.csv`; `hadb_eda.py`, `hadb_deep_review.py`.
- `hadb_selectors_v2.py` - the leak-free selector comparison -> `results/HADB_SELECTORS_V2.csv`.
- `geom_eda.py` - where real anomalies sit -> `results/HADB_ANOMALY_GEOMETRY.csv`.
- Method investigation: `dev_common.py`, `edge_dev.py`, `synth_dev.py`, `synth_vade_dev.py`,
  `shuffle_grid.py`, `shuffle_geom.py`, `fresh_holdout.py`, `real_vs_synth.py`,
  `two_mode_holdout.py`.
- `fetch_sources.py`, `fetch_tsbad.py`, `fetch_mts_and_check_ucr.py` - corpus fetchers
  (corpora are gitignored; UCR is unlicensed, do not redistribute).

## Findings

See [experiments/2026-09-04_synthetic-anomaly-selection-investigation.md](../experiments/2026-09-04_synthetic-anomaly-selection-investigation.md)
and the pre-registration
[experiments/2026-09-03_PREREGISTRATION-hadb-selection-comparison.md](../experiments/2026-09-03_PREREGISTRATION-hadb-selection-comparison.md).
Short version: **EM (Excess-Mass) is the best label-free selector on hard anomalies**; agreement
methods degenerate on normals-only validation; pseudo-anomaly selection (NoMaS family) does not
transfer; feature-shuffle synthetic anomalies carry real but insufficient signal.

## Reproduce

Re-fetch corpora with the `fetch_*.py` scripts, then run the arms, `hadb_consolidate.py`, and
`hadb_selectors_v2.py`. Paths are currently absolute to the build scratchpad; adjust `S`/`ROOT`
constants at the top of each script to relocate.
