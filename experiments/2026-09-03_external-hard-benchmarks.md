# 2026-09-03 — NoMaS on externally curated HARD benchmarks (difficulty stratification + TSB-AD)

**Status:** completed. Two independent tests of "does NoMaS hold up when trivial anomalies
are removed". Both are honest negatives for the reframing hypothesis; one small positive
survives on evaluation quality.

## Background: the hypothesis being tested

Proposal: many benchmark anomalies are TRIVIAL single-feature extremes that every detector
catches, so the ground-truth ranking is dominated by easy cases. If global detectors win
only on such datasets, NoMaS's local bias would be an artifact of the benchmark rather
than a real deficiency, and the 40% oracle gap would be largely illusory.

Trivial definition adopted from the user's own prior work, Apartsin & Aperstein,
"Modeling Normal Is All You Need", arXiv:2607.06094, sec 5.3:

> "A trivial detector scores each window by the maximum absolute standardized per-channel
> window mean, max_ch |z| ... An anomaly window is labeled easy if this trivial score
> exceeds the 99th percentile of train-normal, and difficult otherwise."

Protocol: STRATIFY, do not filter. Positives are the subset's anomalies; negatives are the
FULL normal set.

## Test 1: difficulty stratification on OddBench (174 datasets) — HYPOTHESIS REFUTED

| test | result |
|---|---|
| trivial fraction (99th pct) | 20.8% of anomalies |
| frac_trivial, global-best vs local-best | **0.239 vs 0.168 (1.42x)** |
| same at 99.5th pct | 0.211 vs 0.109 (1.94x) |
| **family flips on the hard subset** | **6 of 153 (4%)** |
| family distribution | global 71 -> **74**, local 51 -> 51, other 31 -> 28 |
| mean detector AUC | 0.597 -> 0.582 |
| best-vs-mean spread | 0.1394 -> **0.1431, p < 0.0001** |

Global-best datasets DO carry modestly more trivial anomalies (1.4x, replicating the
corrected local-tabular estimate of 1.39x at 5x the sample). But **removing them does not
change which family wins**, and the count moves TOWARD global (71->74), opposite the
hypothesis. The local/global limitation is NOT a trivial-anomaly artifact.

**Surviving positive:** difficulty stratification makes the ranking significantly more
discriminative (spread +0.004, p < 0.0001, n=153). Small, but a real argument that
difficulty-stratified evaluation is better evaluation - independent of the family question.

### RETRACTION recorded

An earlier pass using a FIXED `max|z| >= 3` threshold reported a 3.3x enrichment and was
announced as "confound ruled out". That was wrong. The max over d features exceeds 3 by
chance in high dimensions, so the fixed cutoff inflates the trivial fraction with
dimensionality (measured 0.32 at d<=10 rising to 0.65 at d>50). The dimensionality confound
WAS checked, but with the same bad threshold inside both bands, so the check could not
detect that the threshold itself manufactured the effect. The published,
dimension-adaptive threshold gives 1.4x, not 3.3x.

## Test 2: NoMaS on TSB-AD-U (externally curated hard benchmark) — NEGATIVE, with caveats

TSB-AD (Liu & Paparrizos, NeurIPS 2024 D&B, Apache-2.0) re-curated 1,070 series from 40
sources specifically to remove flawed exemplars. 200 series hash-sampled (NOT
alphabetically - the OddBench sampling-bias lesson); 95 survived the windowing guards.

| | regret |
|---|---|
| NoMaS | 0.1143 |
| random pick | **0.1030** |
| reduction | **-11.1%** |

56 wins / 39 losses / 0 ties, p = 0.24. NoMaS wins MORE OFTEN than it loses but the mean is
worse: it wins small and loses catastrophically.

Entirely source-dependent:

| source | n | NoMaS | random | reduction |
|---|---|---|---|---|
| IOPS | 5 | 0.0106 | 0.0582 | **+82%** |
| Exathlon | 7 | 0.0541 | 0.0737 | +27% |
| UCR | 26 | 0.1198 | 0.1361 | +12% |
| WSD | 20 | 0.0627 | 0.0680 | +8% |
| SMD | 11 | 0.1126 | 0.0734 | **-53%** |
| OPPORTUNITY | 10 | 0.3911 | 0.1932 | **-102%** |

OPPORTUNITY alone drags the mean negative. On UCR - the source Wu & Keogh built to be
non-trivial - NoMaS is positive (+12%). The local bias reappears: true best is global on 28
series, NoMaS picks global on 15.

### Caveats that materially limit this result

1. **The window descriptor is the paper's acknowledged weak point.** This used
   `_window_features` (28 summary statistics); the Limitations already state the TS arm
   "needs a richer window descriptor than a handful of summary statistics." A negative is
   ambiguous between "NoMaS fails on hard anomalies" and "28 summary stats cannot represent
   these signals." **The descriptor is the thing to fix if the TS arm is to hold up.**
2. **Only 95 of 200 sampled series survived.** Most YAHOO series (259 of 870 in TSB-AD) are
   ~1400 points, giving 85 windows at w=64/stride=16, below the 200-window floor. The
   sample skews to long series and excludes the largest source, so the promised
   "curated-Yahoo vs curated-UCR" comparison could NOT be made.
3. Same 2.6x-weak harness as all of today's regret numbers.

### Incidental finding: unrealistic anomaly density inside TSB-AD

The first TSB-AD run crashed because a series had only 11 normal windows: with
`min_count=1`, a contiguous anomaly region makes every overlapping window anomalous. That
is Wu & Keogh's second named flaw (unrealistic anomaly density) appearing INSIDE the
benchmark curated to remove such flaws. Skipped series and their windowed anomaly rates are
logged in `tsbad_nomas_skipped.csv`.

## Decision recorded: do NOT build HADB

A "Hard Anomaly Detection Benchmark" was designed and scripted (`build_hadb.py`, using the
published trivial definition, drop-trivial-rows policy, >=20 hard anomalies, rate in
[0.001, 0.25]). It was NOT built, for three independent reasons:

1. **The argument is published**: Röchner, Klüttermann, Kammler, Rothlauf, Müller, Schlör
   (2025), "We Need to Rethink Benchmarking in Anomaly Detection", arXiv:2507.15584 —
   verbatim: "a trivial algorithm that only checks for extreme values in individual
   features performs competitively with state-of-the-art deep learning methods."
2. **The construction is published**: Emmott, Das, Dietterich, Fern, Wong,
   "A Meta-Analysis of the Anomaly Detection Problem", arXiv:1503.01158 — defines **point
   difficulty**, relative frequency, clusteredness, feature relevance as benchmark
   dimensions. The canonical tabular difficulty stratification, from 2015.
3. **Our own data says it would not help**: 0/36 family flips locally, 6/153 on OddBench.

Plus: TSB-AD already did the curation for time series, and MacrOData (2,446 datasets) took
the scale niche. See `scout_synthetic_anomaly_selection.md` and the dataset scout below.

## Reference: Wu & Keogh triviality definition (verified)

Renjie Wu, Eamonn J. Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed
and are Creating the Illusion of Progress", IEEE TKDE 35(3):2421-2429, 2023,
DOI 10.1109/TKDE.2021.3112126, arXiv:2009.13807. Definition 1, verbatim:

> "A time series anomaly detection problem is *trivial* if it can be solved with a single
> line of standard library MATLAB code. We cannot 'cheat' by calling a high-level built-in
> function such as *kmeans* or *ClassificationKNN* or calling custom written functions. We
> must limit ourselves to basic vectorized primitive operations, such as *mean*, *max*,
> *std*, *diff*, etc."

Brute-force search over their one-liner parameters solves **316 of 367 (86.1%)** Yahoo
series. Their four flaws: triviality, unrealistic anomaly density, mislabeled ground truth,
run-to-failure bias. Their verdict: "The community should abandon the Yahoo, Numenta, NASA
and OMNI benchmark datasets."

Note their definition is **dataset-level and solution-based**; ours (and Apartsin &
Aperstein's) is **point-level and statistic-based**. A solution-based criterion adapts to
whatever makes a dataset easy; a statistic-based one only catches marginal extremity.

## Citations the paper should add

- **Goswami, Challu, Callot, Minorics, Kan**, "Unsupervised Model Selection for Time-series
  Anomaly Detection", ICLR 2023 (notable-top-25%), arXiv:2210.01078. Nearest prior art in
  spirit: inject synthetic anomalies, rank detectors by AUC on them, no labels. NOT
  currently cited, and a UOMS-literate reviewer would expect it.
- **Röchner et al. 2025** (above) and **Emmott et al. 2015** (above) for the difficulty /
  trivial-anomaly framing in Limitations.
- **Pinet et al. 2026**, arXiv:2606.02670: across 8 public multivariate benchmarks, no
  cross-channel rupture occurs without an accompanying univariate deviation; on 6 of 8, at
  least half of labelled anomalies deviate univariately 89-100% of the time. The strongest
  single citation for "benchmark anomalies are dominated by marginal excursions."
- Optional: **Ma, Zhao, Zhang, Akoglu**, "The Need for Unsupervised Outlier Model
  Selection", SIGKDD Explorations 25(1) 2023, DOI 10.1145/3606274.3606277 — benchmarks
  label-free internal criteria over 39 tasks x 297 models and finds they underperform.
  Good motivation citation for the gap NoMaS fills.

## Library / licensing note for any future candidate-pool work

- **PyOD 3.6.5** (BSD-2) now ships **61 detectors** and a `TimeSeriesOD` wrapper that turns
  any PyOD detector into a windowed TS detector - the cheapest route to a large
  multi-hyperparameter pool. Installed here: 3.6.1.
- **dtaianomaly 0.5.1** (KU Leuven, MIT) exposes `window_size` on essentially every
  detector - best for window sweeps.
- **aeon 1.5.0** (BSD-3) has a `PyODAdapter` and covers ROCKAD/MERLIN/DWT_MLEAD.
- **LEGAL BLOCKER: `alibi-detect` is now Business Source License 1.1**, not Apache -
  non-commercial except non-profit education. Do not use in a published method.
- Dead or blocked: Merlion (repo archived 2026-03, needs a JDK), ADTK (2020), Kats (2022),
  TODS (PyPI does not match repo), luminol/banpei, TimeEval (Docker-only), DeepOD (pins
  `torch<1.13.1`, will not resolve on Python 3.14).
- Dataset licensing: **DAMI/LMU carries NO license statement** (confirmed absent) and ODDS
  could not be verified. MacrOData/OddBench is CC BY 4.0. This validates the Zenodo
  deposit's decision not to rehost any corpus.
