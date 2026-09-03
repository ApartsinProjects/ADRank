# Pre-registration: HADB selection-method comparison

**Status:** PRE-RESULTS for NoMaS and for the label-free literature baselines (EM, MV,
consensus, ModelCentrality, HITS). Written before any of those has been run on HADB.
**Date:** 2026-09-03. The git commit that adds this file is the timestamp of record; it
predates the NoMaS-on-HADB and round-2 baseline results.
**Purpose:** lock the metric, the baselines, the statistical test, the significance
threshold, and the decision rules for contribution 3 of the paper, so no choice in the
comparison can be made after seeing the numbers.

This session retracted eight-plus mechanism findings that dissolved under a control run
afterward. This document exists so the selection comparison cannot join them: everything that
could be tuned to a favourable answer is fixed here, in advance.

---

## 1. What is frozen before the test

**The benchmark.** HADB, three-way build consolidated 2026-09-03 (see deviation D1):
292 included datasets, effective N = 287 (dedup groups), two modalities (tabular 185, time
series 107), six corpora (adbench+dami, OddBench, OvrBench, TSB-AD-U, UCR, TSB-AD-M).
Inclusion rule: zone = live (best_ap_norm >= 0.10) AND spread_ap_norm >= 0.10 AND
base_rate <= 0.25. The manifest (`HADB_MANIFEST.csv`) and per-dataset rankings are frozen; no
dataset is added, removed, or re-filtered after this date to change a comparison outcome.

**The candidate pool per dataset.** 23 detectors for time series, up to 36 for tabular. A
selector picks exactly one candidate per dataset, without labels. The pool is not changed to
favour any selector.

**The protocol (three-way, leak-free).** Normals split 60/20/20 into TRAIN / VALIDATION /
TEST, disjoint in raw points (for time series, contiguous-block split with an overlap purge).
Detectors fit on TRAIN. Every label-free selector computes its criterion on VALIDATION
(normals only, no anomalies). Ground-truth AUC / AP / ap_norm is computed on TEST (held-out
normals + all hard anomalies). A selector never sees the set it is scored on. Three seeds,
averaged per (dataset, detector).

## 2. Primary metric and secondaries

**Primary:** mean **regret on ap_norm** = best_ap_norm - ap_norm(selected), averaged over the
307 included datasets, where ap_norm = (AP - base_rate)/(1 - base_rate). Lower is better;
0 = picked the label-derived best.

**Secondaries** (reported, never substituted for the primary): normalized rank of the selected
detector in [0,1] (0 = best, 1 = worst); percentage of datasets where the exact best was
picked; and regret computed on AUC and on pAUC@10 as robustness checks. The primary is fixed
now precisely so a secondary cannot be promoted after the fact because it reads better.

## 3. Selectors

**Baselines (already run on the three-way build, 2026-09-03; the frozen setup, not the test):**

| selector | deployable | regret (overall) | beats random? | role |
|---|---|---|---|---|
| oracle_best | no | 0.000 | — | floor / sanity anchor |
| global_fixed (LOF_k3 tab / LOF_k10 ts) | no (uses labels) | 0.191 | — | reference bar to approach |
| **EM** (Excess-Mass, Goix) | yes | **0.216** | **YES**, Holm p<0.001 | best deployable baseline |
| **MV** (Mass-Volume, Goix) | yes | **0.220** | **YES**, Holm p<0.001 | |
| iforest_random | yes | 0.274 | tabular only | cheap default (UOMS IFOREST-R) |
| HITS / ModelCentrality / consensus | yes | ~0.278 | no (degenerate on normals-only val) | agreement-based, collapse to picking IForest |
| random | yes | 0.286 | — | the bar every real method must clear |
| anti_oracle | no | 0.491 | — | ceiling / sanity anchor |

Criteria computed on VALIDATION scores, evaluated by TEST regret. Finding (verified,
mechanistic): under normals-only validation, agreement-based criteria (consensus / MC / HITS)
degenerate to picking one central detector (IForest on 207/287 datasets) and tie random, while
density-concentration criteria (EM / MV) retain per-dataset signal and beat random. This is a
reversal of Ma et al. (transductive), where consensus worked and EM/MV did not.

**Method under test:** NoMaS (cluster-holdout pseudo-anomaly separability), run with its
default configuration. **No NoMaS hyperparameter is tuned on HADB.** If more than one NoMaS
variant is evaluated, the single primary variant is declared before the run in the deviations
log below; any additional variant is exploratory and labelled as such.

## 4. Hypotheses (directional, stated before the run)

- **H1 (primary).** NoMaS achieves lower mean regret than `random`.
- **H2.** NoMaS achieves lower mean regret than the best *deployable* baseline, now fixed as
  **EM at 0.216** (chosen by its own regret before NoMaS is run, not by comparison to NoMaS).
- **H3.** NoMaS closes at least half the gap between `random` and `global_fixed`
  (i.e. regret <= 0.239 = 0.286 - 0.5 x (0.286 - 0.191)).
- **H4 (per modality).** H1 holds separately on tabular and on time series.

## 5. Statistical test and threshold

Paired **Wilcoxon signed-rank** test on per-dataset regret, two-sided, **alpha = 0.05**.
Effect reported as median per-dataset regret difference plus win/tie/loss counts.
**Multiple comparisons:** Holm correction across the family of NoMaS-vs-baseline tests
(H1, H2, H4-tabular, H4-timeseries, and one test per round-2 baseline). Effective N for the
test is 302 dedup groups, not 307 rows: correlated members (PageBlocks, Stamps, Waveform,
Wilt) are collapsed to one before the paired test.

## 6. Decision rules (both branches pre-committed)

The wins-only discipline governs *method* claims; a benchmark paper must report what it finds.
Both outcomes are therefore committed here so neither is a post-hoc story.

- **WIN.** H1 holds after Holm correction (NoMaS < random, p < 0.05) AND NoMaS beats every
  deployable baseline. Claim: "NoMaS is the first label-free selector to beat random on hard
  anomalies." H2/H3/H4 sharpen the claim; they are not required for it.
- **PARTIAL.** H1 holds but NoMaS does not beat all deployable baselines, or holds on one
  modality only. Report the scoped claim exactly as measured (e.g. "NoMaS wins on time series,
  ties on tabular"), no broadening.
- **NULL / LOSS.** H1 fails (NoMaS not distinguishable from, or worse than, random). Claim:
  "on hard anomalies, no label-free selector we tested - ours included - beats random", which
  is itself the benchmark's central finding and is reported as such. NoMaS is not dropped from
  the paper to manufacture a win elsewhere.

**Falsification condition (the thing that would say NoMaS does not work here):** NoMaS regret
>= random regret, or a two-sided Wilcoxon p >= 0.05, on the primary metric over the 302
effective datasets.

## 7. Analysis constraints (anti-p-hacking)

- One config, one pass. NoMaS is run once at its default settings; the primary metric is
  computed once. No iterating the method against HADB regret.
- No metric shopping: the primary is regret on ap_norm, fixed in section 2.
- No dataset dredging: subgroup results beyond the pre-declared per-modality split (H4) are
  exploratory and labelled so.
- Every selector uses the identical frozen pool and protocol; a selector is never given a
  different candidate set.
- Sanity invariants re-checked every run: oracle regret == 0; anti_oracle >= all selectors
  per dataset; random norm_rank ~ 0.5. A violated invariant blocks reporting until resolved.

## 8. Deviations log

Any departure from sections 1-7, with date and reason, is recorded here. All entries below
predate the NoMaS run (H1-H4 are still untested).

- **D1 (2026-09-03) Two-way -> three-way rebuild.** The first build had only train/test, so a
  selection criterion computed on test would leak. Rebuilt with a 60/20/20 train/val/test
  split of normals; criteria moved to validation. Detectors now fit on 60% (was 80%), so the
  ground truth shifted and the manifest re-froze at 292 datasets / eff-N 287 (was 307 / 302).
  This is a correctness fix made before any NoMaS result existed.
- **D2 (2026-09-03) Stability baseline dropped.** The split-half stability baseline was
  removed in the three-way refactor (it needed an extra sub-split and was already characterised
  as worse-than-random). Not one of the label-free literature methods; excluded from the
  comparison.
- **D3 (2026-09-03) OvrBench capped at 131 scored.** One pathological dataset hung the arm
  (>13 min, full CPU); the run was killed and the 131 completed datasets kept. A per-dataset
  timeout guard is a TODO for a clean re-run. Does not affect other arms.
- **D4 (2026-09-03) EM/MV volume convention.** EM/MV are evaluated in the unit box
  (min-max-scaled features, volume_support = 1) because only the per-dataset detector ORDERING
  is consumed; absolute EM/MV magnitudes are not compared across datasets. Validated on
  synthetic data (good detector beats noise; FPR-style sanity checks pass).
