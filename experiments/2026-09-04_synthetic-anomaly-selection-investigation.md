# Label-free selection on HADB: what works, what doesn't, and why

**Date:** 2026-09-03/04. Governed by the pre-registration
([2026-09-03_PREREGISTRATION-hadb-selection-comparison](2026-09-03_PREREGISTRATION-hadb-selection-comparison.md)).
All numbers are leak-free: criteria computed on VALIDATION, scored on TEST (three-way split).
The decisive metric throughout is the **within-dataset Spearman correlation** between a
selector's score and the true test ap_norm - regret alone repeatedly gave false positives on
small samples and is reported only alongside the correlation.

## Headline (UPDATED 2026-09-04 after a bug-hunt + benchmark re-filter)

The picture changed materially. See the "Bug-hunt corrections" section below.

- On the FIRST (292-dataset) benchmark, **EM (Excess-Mass, Goix 2016) was the best label-free
  selector and nothing beat it** - this reverses Ma et al. 2023 (consensus won, EM/MV did not).
- But two things were wrong: (a) the synthetic-anomaly selector was fitting detectors on the
  wrong data (validation subset, not train), handicapping it; (b) 57 tabular datasets (18-31%)
  were still solvable by the max|z| triviality RULE and should have been filtered.
- **On the CLEANED benchmark (235 datasets, max|z|-trivial removed) with the corrected
  protocol, feature-shuffle SYNTHETIC beats EM overall** (regret 0.146 vs 0.211) and DOMINATES
  local-best datasets (0.045 vs 0.196, wins 22/37). EM's apparent overall win was partly
  inflated by near-trivial global datasets. Synthetic's one remaining blind spot is global-best
  datasets, for a precise structural reason (below).

The rest of this document records the original comparison and the mechanisms; the bug-hunt
section at the end supersedes the original headline.

## The selector comparison (leak-free, 292 datasets, `HADB_SELECTORS_V2.csv`)

Regret on ap_norm, lower better; the label-free family judged on validation.

| selector | regret | vs random | notes |
|---|---|---|---|
| oracle_best | 0.000 | - | floor |
| global_fixed | 0.191 | reference | label-cheating (uses benchmark labels) |
| **EM** | **0.216** | **beats, Holm p<0.001** | best deployable |
| MV | 0.220 | beats, Holm p<0.001 | |
| iforest_random | 0.274 | tabular only | |
| consensus / ModelCentrality / HITS | ~0.278 | no | degenerate on normals-only val |
| random | 0.286 | - | |
| anti_oracle | 0.491 | - | ceiling |

**Why the agreement methods collapse:** on normals-only validation every detector agrees
(normals look normal), so the agreement graph is uninformative and they pick the most central
detector every time - IForest on 207/287 datasets, norm_rank 0.46 (= random). EM/MV retain
per-dataset signal because they measure normal-region concentration, which varies by detector.

## Pseudo-anomaly selection (NoMaS family) FAILS - and why

Tested: cluster-holdout (NoMaS), edge (tail points), embedded (small clusters among others),
and a two-channel edge+embedded mix. All hold out NORMAL points as pseudo-anomalies.

Decisive check (within-dataset Spearman of the criterion vs true ap_norm):

| method | within-dataset rho | argmax pick true-rank pct |
|---|---|---|
| pseudo-anomaly (embedded) | **-0.107** | 0.541 (WORSE than random) |
| two-channel mix | -0.076 | 0.507 |

**Verdict: no signal (slightly negative), picks worse than random.** Regret-based "wins"
(oc_embed 0.232 < EM) were small-sample + metric artifacts; the rank check exposed them.
**Mechanism:** separating a held-out pocket of NORMAL points is a different task from detecting
a real anomaly. Detectors that ace the pseudo-task overfit to normal-cluster structure and do
not transfer. The optimal two-channel mix weight (0.70 embedded / 0.30 edge) did match the real
anomaly geometry (below), but that only confirms the geometry story - it did not rescue
selection.

## Where real hard anomalies sit (`HADB_ANOMALY_GEOMETRY.csv`)

Placing real hard anomalies on the normal manifold: **inside 0.52, between-clusters 0.18,
edge 0.08, outside 0.23.** Median radial-pct 88 (elevated, not extreme - consistent with
surviving the triviality filters), median reconstruction-pct 77. 70% are inside/between, not at
the outer edge - which is why edge/cluster pseudo-anomalies (all at the rim) mis-rank detectors.

## Synthetic anomalies: the one direction with real signal

Instead of holding out normals, GENERATE anomalies. Feature-shuffling (copy a normal, overwrite
a random subset of features with values from other normals - breaks the joint structure, keeps
the marginals):

| generator | within-dataset rho (fresh holdout) | regret vs EM |
|---|---|---|
| pseudo-anomaly | -0.107 | - |
| PCA between-cluster | +0.102 | - |
| **feature-shuffle** | **+0.254** (median +0.305, >0 on 20/30) | 0.182 vs EM 0.157 |

The correlation **replicated and strengthened on fresh corpora** (ovrbench + tsbad_u, never used
in dev) - so it is real signal, NOT dev-overfitting. But shuffle-synthetic **ties/loses to EM on
regret** (p=0.36): it has better rank-correlation yet occasional catastrophic picks. EM is
weaker-correlation but safer. The EM+shuffle ensemble ties EM (0.153 vs 0.157, p=0.68) - no
meaningful gain. Filtering (VaDE recon, or cluster membership) did NOT help; the generator is
the signal, not the filter.

**Two-mode (shuffle + displacement) does not help either.** A displacement generator (move a
normal toward/past another cluster centre in PCA latent) was added to cover the anomaly type
shuffling cannot reach. It generates poor anomalies on its own (rho +0.12, regret 0.40), and the
two-mode pool is WORSE than shuffle alone (rho +0.20 vs +0.26, regret 0.22 vs 0.18) and still
loses to EM (two-mode 0.22 vs EM 0.16, p=0.053). The displacement axis is understood (below) but
not closeable with a simple generator - crude displacement lands either marginally-extreme
(trivial) or inside another cluster (normal-looking).

## Why synthetic differs from real (`real_vs_synth.py`)

The mechanism, measured directly (24 datasets):

| | median radial-pct | median recon-pct |
|---|---|---|
| REAL hard anomalies | **86 (displaced)** | 64 |
| SYNTHETIC shuffle | 64 (central) | **72 (off-manifold)** |

**Two geometrically different kinds of anomalous.** Real anomalies are DISPLACED (moved away
from the normal region). Shuffle synthetics are STRUCTURALLY BROKEN but stay CENTRAL - because
shuffling preserves marginals exactly (every feature value is a real normal value), a shuffled
point can never leave the normal range on any single feature; it can only be in the wrong
combination. Shuffling geometrically CANNOT produce a displaced anomaly (4% "outside" vs 23% for
real). And the mismatch predicts failure: corr( (real_radial - synth_radial), shuffle_rho )
= **-0.290** - shuffle works worse exactly where real anomalies are more displaced than the
synthetics can reach.

One-line answer: **synthetic shuffle anomalies violate structure but stay put; real hard
anomalies move. Shuffling reproduces the "wrong combination" but not the "wrong place," and
detectors need both.**

## Design that carries to the paper (contribution 3)

- **EM is the surprising winner**; a full leak-free comparison shows every alternative
  (agreement, pseudo-anomaly incl. NoMaS, synthetic, ensembles) fails to beat it significantly.
- **Two verified negative mechanisms**: agreement methods degenerate on normals-only; pseudo-
  anomaly selection does not transfer (held-out normals != real anomalies).
- **One verified positive secondary result**: structure-violating synthetic anomalies
  (feature-shuffle) produce a score that genuinely correlates with detector quality
  (rho +0.25, fresh-holdout confirmed) - it just does not exceed EM because it misses the
  displacement axis.
- The paper's contribution 3 is the honest "the problem is harder than believed; here is the
  surprising baseline that works, and here is why the intuitive approaches fail" story, fully
  supported by the benchmark. Wins-only discipline governed the NoMaS method paper; a benchmark
  paper reports what it finds.

---

# Bug-hunt corrections (2026-09-04) - these supersede the original headline

The user pushed to verify the negative synthetic result. The hunt found two real bugs and a
benchmark-filter gap, and the corrected result flips the headline.

## Correction 1: synthetic was fit on the wrong data (protocol bug)

The dev synthetic experiments fit detectors on 70% of the VALIDATION set (~14% of normals),
while the ground truth and EM use detectors fit on TRAIN (60%). So synthetic ranked
differently-fit detectors than the ones the ground truth measures. Fixing it (fit on TRAIN,
score validation-normals vs synth) lifted within-dataset rho from +0.27 to +0.30. Validated
against `hadb_round2_common._fit_score` (ground truth: fit Xtr, score Xte) and the EM path
(fit Xtr, score Xval). Scripts: `corrected_synth.py`.

## Correction 2: NoMaS itself violates the protocol

`nomas_scores` (hadb_round2_common line 62) re-fits detectors on validation subsets
(`m.fit(Xval[sub.train_idx])`), not the train-fit detector EM ranks. So in the benchmark
comparison NoMaS ranked differently-fit detectors than EM did - inconsistent, and it
contributed to NoMaS's poor showing. It is partly architectural (cluster-holdout inherently
re-fits), so it cannot simply use the train-fit detector; noted as a limitation.

## Correction 3: the triviality filter missed a per-dataset check (benchmark re-filter)

The max|z| triviality filter is per-DATAPOINT (drop anomalies above the 99th-pct max|z| cut).
It never checked whether the max|z| RULE still separates the SURVIVORS of a dataset. Audit
(`trivial_refilter.py`): the max|z|-rule test AUC over 185 included tabular datasets has median
0.71 but **29 datasets > 0.90 and 55 > 0.80** - near-trivial by the simple rule (NetworkFlow
0.98, MouseInteractionPhase 0.99, ...). Re-filtering at max|z|-rule AUC > 0.85 drops **57
tabular datasets** (24 of them global-best), leaving **235 (128 tabular + 107 TS)** ->
`HADB_MANIFEST_REFILTERED.csv` (`include_v2`). HBOS is a real pool detector, so HBOS-solvable
but max|z|-hard datasets (Orbital, UserEvent) are KEPT - they are legitimate global-vs-local
selection datasets.

## The corrected result: synthetic beats EM on the cleaned benchmark

Corrected protocol, cleaned benchmark, sample of 60 datasets (`cleaned_synth_vs_em.py`):

| | regret | within-dataset rho |
|---|---|---|
| **feature-shuffle synthetic** | **0.146** | +0.31 |
| EM | 0.211 | +0.13 |

paired Wilcoxon p=0.28 (high variance), but the family split is decisive:

| true-best family | synth regret | EM regret | synth wins |
|---|---|---|---|
| **local (n=37)** | **0.045** | 0.196 | **22/37** |
| global (n=19) | 0.300 | 0.248 | 6/19 |
| other (n=4) | 0.351 | 0.177 | 1/4 |

**Synthetic dominates local-best datasets and loses only on global-best ones.**

## Why synthetic still loses on global datasets - a structural limit (verified)

- **Local vs global, mechanistically** (`global_anom_geom.py`, best-config `example_local_vs_global`):
  a GLOBAL-win anomaly is **extreme in a single feature** (per-feature histogram/CDF rarity);
  a LOCAL-win anomaly is anomalous only in the **multivariate combination** (no single feature
  extreme, median 0 features with |z|>2.5 vs 0.5 for global). On global datasets even the BEST
  local config loses at EVERY threshold (NetworkFlow: best local AUC 0.71 vs HBOS 1.00;
  recall@10%FPR 0.28 vs 1.00) - not a threshold/k issue, a ranking failure.
- **Failure mode = FALSE NEGATIVES** (`fn_fp_diag.py`): on synth-loses datasets the synthetic-
  picked detector's recall@10%FPR is 0.31 vs EM's 0.62 - it MISSES the real anomalies, does not
  over-flag normals.
- **The structural limit** (`real_vs_synth` on global datasets): shuffling copies real normal
  values, so no single feature can exceed the observed range -> synthetic max|z| caps at ~1.45
  vs real 2.02. Shuffle can make the multivariate-broken part of an anomaly but NOT the
  single-feature-extreme part. So on global-best datasets it selects a local detector that then
  misses the marginally-extreme real anomalies.

## Histogram vs Gaussian triviality - analysis + open design fork

The max|z| filter assumes GAUSSIAN marginals. Analysis (`hist_vs_z`): 59% of features here are
skewed (|skew|>1) or bimodal (BC>0.55); on such features a value in the skewed tail or a
bimodal gap is empirically rare (histogram-detectable) but z-moderate - z-filtering misses it
(visualizing_soil 47% of anomalies, Orbital 27%). A histogram-based filter would be more
principled BUT it would remove exactly the anomalies HBOS catches -> remove the global-win
datasets -> collapse the local-vs-global selection challenge into a local-only benchmark.
This is a genuine fork in the benchmark's identity, left open:
- keep max|z| triviality (+ the per-dataset re-filter) -> local-vs-global selection benchmark,
  EM's family-agnosticism is the finding;
- switch to histogram triviality -> multivariate-hard-only benchmark, synthetic/local wins,
  but no local-vs-global story.

## Router / measured-locality / family-fair evaluation (2026-09-04)

Follow-up thread on whether to *route* local-vs-global per dataset, and how to evaluate fairly.

### β-router and matched selector are a wash on the pooled metric
- Density-reweighted resampling at exponent β spans the probe from typical/multivariate
  (β=+1, local-type synthetics) to marginal-tail (β=-4, global-type). Per dataset, `a1[d]`/`a4[d]`
  = each detector's separation AUC on the β=+1 / β=-4 synthetics.
- **`local_ev = max_d a1[d] - max_d a4[d]`** is a label-free dataset descriptor: high = local-type
  synthetics more separable -> local family wins; low = marginal-tail wins. It predicts the true
  winning family at **AUC 0.795**, vs 0.62 for the best static multimodality scalar.
- A **matched selector** (route to argmax `a1` if `local_ev>τ` else argmax `a4`, τ=-0.14 calibrated)
  and the **max-spread router** both FAIL to beat fixed β=1 on the pooled (micro) regret: holdout
  matched 0.121 vs β1 0.124, p=0.500 (same pick on 29/30 datasets). Reason: β=1 is already near-optimal
  on the majority (local) datasets, so routing only changes the pick on the global minority.
- **Fixed β=1 synthetic remains the simple method that beats EM** (holdout p=0.023; representative p=0.003).

### Family-fair (macro) evaluation flips the router verdict
The pooled mean is a MICRO-average dominated by the local-heavy composition, so it rewards matching
the majority family. Under a **family-balanced macro-average** (weight local and global equally):
- holdout macro: **matched 0.136 < β1 0.162 < EM 0.206**.
- full-73 stratified bootstrap: macro(matched)-macro(EM) = -0.064, 95% CI [-0.121,-0.018], P=0.997 (solid);
  macro(matched)-macro(β1) = -0.028, 95% CI [-0.077,+0.008], P=0.917 (suggestive, global-stratum-limited).
- Per-family (holdout): global n=9 matched 0.176 vs β1 0.258 (matched rescues); local n=21 matched 0.097
  vs β1 0.067 (matched slightly over-routes). The gains/costs cancel in micro, separate in macro.
- **Recommendation:** adopt the family-balanced macro-average as the benchmark's primary metric,
  report per-family tables, run the full 199-set benchmark to settle matched-vs-β1 at 95%.

### Multimodality is a WEAK, non-visual axis; measured locality is the strong one (162 datasets)
`MODALITY_PERF.csv` (all tabular+uni-TS, detectors fit on train, scored on val-normals vs synthetics):
Spearman of each descriptor with the (best-local - best-global) ap_norm family advantage:
- silhouette +0.13, GMM BIC-gain +0.13, bimodality-coef -0.00, **local_ev +0.38**.
- So static/visual multimodality barely organizes family dominance (why the density heatmap shows no
  crisp clusters - modes overlap); the measured probe organizes it ~3× better.
- Splitting methods by static-modality median shows ~no difference (β1 0.14 both groups). Against
  `local_ev` bins the methods separate: β1 degrades toward unimodal/global, matched cushions it,
  EM flat-poor. Figures: `FIG_scatter_panels.png` (detector + ranking scatter), `FIG_modality_illustration.png`
  (mode-separating projection, the robust way to view multimodality), `FIG_umap_two_examples.png`.

### Mechanism from per-dataset EDA (`inspect_datasets.py`)
- "Global" here often means low-density in a MULTIMODAL marginal, not a Gaussian tail
  (UserEventAnomalies: HBOS wins yet anomalies have median max|z|=1.25, 0 features >3σ - histogram
  rarity, not z-extremeness; vindicates the histogram-filter direction).
- β=1 also fails on some LOCAL datasets - a synthetic-FIDELITY problem, not routing: DiamondClarity
  (true-best LOF ap=0.76, β1 picks HBOS ap=-0.03 on a near-tie in a1); RacingMotion (true-best KNN_k3
  ap=0.99 invisible to every probe). The residual failures are synthetic-vs-real fidelity, which
  no router fixes.

### EM's systematic mis-ranking (detector-independent dataset metric; `em_systematic_failure.py`)
Per (dataset, detector) `em_err = pct_rank(EM) - pct_rank(true ap_norm)` (>0 = EM over-rates). Marginal
over 162 datasets (5029 obs), the error is ZERO-SUM and structured: EM **over-rates IForest/LODA by
+0.153** (p=1e-36) and **under-rates local detectors by -0.050** (p=3e-13); global unbiased on average.
Conditioned on the normals-only multimodality metric: on MULTIMODAL datasets EM over-rates IForest/LODA
(+0.173) and under-rates the local detectors that actually win (-0.032) -> it picks IForest instead of
the correct local detector. This is a mechanism-level indictment of EM independent of our method
(EM's excess-mass criterion rewards IForest/LODA's globally-concentrated scores). Fig `FIG_em_systematic.png`.

### Normal-shape -> anomaly-location -> family chain (`shape_anomaly_relation.py`)
Directionally correct but WEAK at every link: multimodal train normals -> anomalies in local voids
(iso rho +0.20) and less marginal (marg rho -0.13 to +0.10); marginal-edge -> global wins (AUC 0.57);
local-void -> local wins (AUC 0.53). Every link AUC 0.53-0.57, so the end-to-end geometric prediction
is weak (rho ~0.15) - which is WHY no static geometric statistic routes and the measured probe (local_ev
0.795) is needed. Measuring modality on TRAIN vs VAL barely differs (+0.155 vs +0.129).

### Why EM beats the extreme probe on GLOBAL datasets (`global_why_em.py`, 56 global sets)
NOT a fidelity gap: beta=-4 synthetics reach 99% of real extremeness (max|z| 2.45 vs 2.48, both single-
feature). The problem is NON-DISCRIMINATION: within the global family beta=-4 separation ranks detectors
at rho +0.037 (EM +0.075) - both ~0. A single-feature marginal extreme is separated ~equally by every
global detector, so it carries no signal about which one generalizes. EM wins by 0.025 not by ranking
better (a4 ranks the full pool better) but by a steadier top pick. "EM wins on global" = "both fail, EM
fails less."

### Generator design-space sweep - EVERY knob beyond beta tested, all negative
The generator has knobs: beta (extremeness), frac (# features), bins (marginal resolution), cap, base/feature
policy. Results:
- **frac (beta x n_features grid, `grid_beta_frac.py`):** maps anomaly type cleanly (local AUC climbs with
  feature-count, local-vs-global gap peaks at beta=1 + ~0.5d features = +0.24) but NO ranking gain
  (rho flat ~0.35 everywhere) and does NOT discriminate global (within-global rho +0.12 best).
- **elbow-beta (`elbow_analysis.py`):** the beta where a detector flips hard->easy does NOT characterize
  family (AUC 0.47, below chance) - it tracks the hyperparameter (k), not local/global. Max-discrimination
  beta is at the HARD end (+2) on every dataset, confirming the informative probe is hard/multivariate.
- **bins (`bins_test.py`):** finer bins DO target interior multimodal gaps (interior frac 0.66->0.76) and
  improve within-global rho +0.03->+0.10 at beta=-4, BUT hurt overall ranking (+0.199->+0.110). Net not a fix.
- **ensembles (`ensemble_test.py`):** MEAN ensembles significantly worse (dilution; neutral 0.237 vs best-
  individual 0.449 ap_norm); MAX/union ensemble only TIES beta=1 selection (p=0.96). On hard anomalies the
  families are anti-correlated, so combination underperforms selection.
CONCLUSION: ranking the GLOBAL family label-free is a fundamental limit, not a knob-tuning problem. beta=1
(local/multivariate probe) is the method; the global residual resists the entire generator design space and EM alike.

### Leaderboard (162 datasets, construct-matched; `leaderboard.py` -> `LEADERBOARD.csv`, `FIG_leaderboard.png`)
Regret on ap_norm (micro | macro=family-balanced):
- oracle 0.000 | 0.000
- **NoMaS matched-router (ours) 0.138 | 0.153** (vs EM p=0.002)
- best-fixed LOF_k20 (label-cheating) 0.118 | 0.165
- **NoMaS beta=1 (ours) 0.143 | 0.163** (vs EM p=0.022)
- beta=-4 extreme 0.185 | 0.190
- EM 0.196 | 0.193
- random 0.271 | 0.267
Both our methods beat EM significantly; matched-router is the best deployable method under the family-fair
macro. best-fixed LOF_k20 edges micro but cheats (uses benchmark to pick the global-best detector) and
loses on macro. CAVEAT / OPEN: MV/consensus/ModelCentrality/HITS are only from the earlier 292-set run
(MV 0.220, agreement ~0.278=random), NOT construct-matched on this 162-subset; UDR and a proper IFOREST-R
are still pending. A complete construct-matched leaderboard with all UOMS baselines is the remaining task.

### FULL construct-matched UOMS leaderboard (162 datasets; `full_leaderboard.py` -> `FULL_LEADERBOARD.csv`)
All UOMS criteria (em, mv, consensus, model_centrality, hits) are PRECOMPUTED columns in the arm CSVs
(validation-side, 3-way protocol) - the earlier 292-vs-162 mismatch was only that `hadb_selectors_v2.py`
used the old manifest. Re-run on the SAME 162 datasets, one common pool, regret on test ap_norm
(micro | macro=family-balanced); beta=-4 dropped (internal probe, not a method):
- oracle 0.000 | 0.000
- **NoMaS matched (ours) 0.134 | 0.147** (vs EM p=0.002, vs MV p=0.001)
- **NoMaS beta=1 (ours) 0.141 | 0.161** (vs EM p=0.025, vs MV p=0.017)
- MV 0.188 | 0.185 ; EM 0.195 | 0.192 (EM~MV, p=0.68)
- consensus/UDR, ModelCentrality, HITS, iforest_random all 0.258-0.261 macro = **NOT distinguishable from
  random** (vs-random p=0.05-0.20) - they degenerate on normals-only validation, as expected.
Our two methods are the only ones that beat EM/MV; the agreement methods tie random on the current benchmark.
(best_fixed LOF_k20's number is pool-sensitive - it falls out of the criteria-restricted common pool on some
datasets and hits a random fallback; its fair "always LOF_k20" value is ~0.12 micro from `leaderboard.py`.)

### GLOBAL-wins datasets are single-feature / marginal-TRIVIAL (the key benchmark-integrity finding)
Under the RANKING metric (ap_norm, base-rate-matched), a single best feature vs the best detector
(`difficulty_ap.py`): local gap +0.138, **global gap -0.020** (a single oracle-selected feature BEATS the
whole detector pool on global datasets; 71% of global within 0.05 of oracle vs 34% local, p=0.0000). Reason:
global anomalies are single-feature concentrated, and global detectors DILUTE the one informative feature by
aggregating all features. (Under AUC the gap was only 0.012 vs 0.100 - AP is the correct, stricter metric.)
What is special about global datasets (`characterize_global.py`): AXIS-ALIGNED (single-feature AUC collapses
after a random rotation; AUC-sep 0.72) and LOW feature correlation of the normals (0.22 vs 0.32; AUC-sep 0.71,
**label-free**). Multimodal-histogram-rarity REFUTED (hist_gain wrong-signed). Why EM still edges the beta=-4
probe on global (`global_why_em.py`): beta=-4 synthetics match real extremeness (99%) but are non-
discriminative (every global detector separates a single-feature extreme equally; within-global rho ~0).

### Trivial detector = per-feature CALIBRATED OR (`calibrated_or.py` -> `CALIBRATED_OR.csv`)
The user's trivial baseline: threshold every feature by ITS OWN rarity (empirical two-sided ECDF tail /
histogram density), flag on the single most-rare feature (max_j = "at least one threshold met"). Calibration
matters: naive max|z| OR gets ap_norm 0.202 on global (noise features fire in high-d), calibrated OR gets
0.302. Multivariate-necessity gap (best_det - OR): local 0.229 vs global 0.083 - the OR nearly solves global
(single-feature) but leaves a large gap on local (genuinely multivariate). The OR (max) still trails the
FITTED marginal detector (HBOS sums marginal evidence) by ~0.07, so two trivial tiers exist: single-feature
(OR/max) and multi-marginal-additive (HBOS/sum).

### Cleaned-benchmark leaderboard (`cleaned_leaderboard.py`) - honest null on margin
Applying the OR filter (drop mv_gain<0.05: 38 datasets, 14 local + 24 global -> 124 remaining) and re-running:
matched macro 0.147->0.168, EM 0.192->0.203 - every method gets worse (harder benchmark), and our margin over
EM SHRINKS slightly (gap 0.045->0.035, p=0.002->0.011) because the filter also removes local datasets where we
won easily. Our win SURVIVES (matched vs EM p=0.011, vs MV p=0.010; agreement still ~random) but is not
amplified. So the filter is justified on INTEGRITY grounds (a hard-anomaly benchmark should not contain
single-feature-trivial datasets), NOT as a way to inflate the result. Filter-2 (marginal-necessity, local-only)
would inflate the margin by construction (EM collapses on local) - deliberately NOT done. Key point: our lead
comes from the genuinely-multivariate datasets, which survive filtering.

Scratch scripts (session scratchpad, not committed): `router_confirm.py`, `multimodality_test.py`,
`measure_locality.py`, `matched_selector.py`, `compute_modality_perf.py`, `inspect_datasets.py`,
`viz_modality2.py`, `make_scatter_panels.py`, `em_systematic_failure.py`, `shape_anomaly_relation.py`,
`global_why_em.py`, `elbow_analysis.py`, `grid_beta_frac.py`, `bins_test.py`, `ensemble_test.py`,
`leaderboard.py`, `full_leaderboard.py`, `difficulty_test.py`, `difficulty_ap.py`, `characterize_global.py`,
`marginal_filter.py`, `calibrated_or.py`, `cleaned_leaderboard.py`.

## Reproduce

Code in `hadb/`, results in `hadb/results/`. Corpora re-fetched with `hadb/fetch_*.py`
(gitignored; UCR is unlicensed - do not redistribute). Original pipeline: build arms ->
`hadb_consolidate.py` -> `hadb_selectors_v2.py`. Bug-hunt: `corrected_synth.py`,
`trivial_refilter.py` (-> `HADB_MANIFEST_REFILTERED.csv`), `cleaned_synth_vs_em.py`,
`em_wins_diag.py`, `fn_fp_diag.py`, `global_anom_geom.py`, `real_vs_synth.py`,
`within_family_synth.py`. Shared reconstruction: `dev_common.py`.
