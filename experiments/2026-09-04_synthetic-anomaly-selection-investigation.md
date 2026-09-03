# Label-free selection on HADB: what works, what doesn't, and why

**Date:** 2026-09-03/04. Governed by the pre-registration
([2026-09-03_PREREGISTRATION-hadb-selection-comparison](2026-09-03_PREREGISTRATION-hadb-selection-comparison.md)).
All numbers are leak-free: criteria computed on VALIDATION, scored on TEST (three-way split).
The decisive metric throughout is the **within-dataset Spearman correlation** between a
selector's score and the true test ap_norm - regret alone repeatedly gave false positives on
small samples and is reported only alongside the correlation.

## Headline

On hard anomalies in the strict normals-only setting, **EM (Excess-Mass, Goix 2016) is the best
label-free selector, and nothing we built beats it significantly.** This reverses Ma et al.
(SIGKDD Explorations 2023), where consensus methods won and EM/MV did not. The reversal is
mechanistic and verified, not incidental.

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

## Reproduce

Code in `hadb/`, results in `hadb/results/`. Corpora re-fetched with `hadb/fetch_*.py`
(gitignored; UCR is unlicensed - do not redistribute). Pipeline: build arms -> `hadb_consolidate.py`
-> `hadb_selectors_v2.py` / `hadb_deep_review.py`. Method investigation: `dev_common.py` +
`edge_dev.py` / `synth_dev.py` / `shuffle_grid.py` / `fresh_holdout.py` / `real_vs_synth.py` /
`two_mode_holdout.py`.
