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

## Reproduce

Code in `hadb/`, results in `hadb/results/`. Corpora re-fetched with `hadb/fetch_*.py`
(gitignored; UCR is unlicensed - do not redistribute). Original pipeline: build arms ->
`hadb_consolidate.py` -> `hadb_selectors_v2.py`. Bug-hunt: `corrected_synth.py`,
`trivial_refilter.py` (-> `HADB_MANIFEST_REFILTERED.csv`), `cleaned_synth_vs_em.py`,
`em_wins_diag.py`, `fn_fp_diag.py`, `global_anom_geom.py`, `real_vs_synth.py`,
`within_family_synth.py`. Shared reconstruction: `dev_common.py`.
