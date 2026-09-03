# 2026-09-03 — Code + intermediate-results audit of every experiment run today

Systematic bug hunt over all 12 experiment scripts and 10 intermediate result files.
Run BEFORE any of today's numbers are treated as findings.

## Clean

- All AUCs within [0,1]; no negative regret anywhere (regret = best - picked cannot be
  negative, and is not); no NaN/inf in pseudo_auc / true_auc / regret.
- One flagged "7368 duplicate rows" in `edge_full_all.csv` was a FALSE POSITIVE of the
  audit script: that file keys on `group`, not `cluster`, and the key list omitted it.

## Checked and CLEARED: unequal detector panels

~9% of cells carry 8 detectors instead of 9. The worry was that if the TRUE-BEST detector
failed on a cell, the pick cannot be right and regret is inflated unevenly across arms.

**Not a bias.** Every delta arm sees the IDENTICAL panel per cluster (0 of 424 clusters
differ across deltas). Failures are deterministic, not stochastic: CBLOF fails on 5 cells
and PCA on 34, the same cells at every delta. Arms are therefore compared on the same
detector set.

Minor note: PCA (a GLOBAL detector) failing on 34/2544 cells slightly under-represents the
global family in those cells. 1.3% of cells; negligible.

## Real, bounded impact: best-detector flips across experiments

5 of 38 datasets have a different true-best detector in different experiments; 4 of 38 have
a different FAMILY label, which makes the global/local splits inconsistent between
experiments.

| dataset | top-2 gap | verdict |
|---|---|---|
| 43_WDBC | 0.0000 | exact tie, harmless |
| 23_mammography | 0.0022 | near-tie, harmless |
| 28_pendigits | 0.0149 | near-tie, harmless |
| Cardiotocography | 0.0330 | near-tie, harmless |
| **26_optdigits** | **0.1716** | GENUINE inconsistency |

**Root cause:** every script recomputes `true_auc` by fitting on its own subsample
(MAX_N 4000 vs 5000, different RNG). Median spread of a detector's true_auc across
experiments is **0.066**, worst **0.302**.

**Scope of the damage:** this matters for CROSS-experiment comparisons (e.g. comparing a
gap measured in gen_valid against one measured in displace_full). It CANCELS in the
within-experiment paired comparisons, which is where every p-value today comes from, since
both arms share one subsample and one true_auc.

**Fix for future runs:** compute true_auc ONCE per dataset on the full data, cache it, and
have every experiment read the cache.

## SERIOUS #1: the harness is 2.6x weaker than the real pipeline

Measured against the real pipeline's k-means arm on the same DAMI datasets:

| | mean regret |
|---|---|
| my experimental harness (delta=0 baseline) | **0.0875** |
| real NoMaS pipeline | **0.0335** |

Worst cases: PageBlocks 0.1906 vs 0.0077 (25x), Stamps 0.2323 vs 0.0243 (10x).

Cause: the harness uses K=20, <=10 regimes, 1 seed, a 5000-row subsample, and NO
auto-calibration. The real method uses K in {30,50}, M=20 sampled subsets, 5 seeds, and
discriminative regime weighting.

**Consequence:** every REGRET number produced today is measured against a strawman
baseline, i.e. an unfair comparison in which the baseline sees less compute and less
information. Improvements over it are NOT evidence about the published method. The GAP /
mechanism measurements are unaffected (they concern which family scores higher on the
pseudo-task, not how good the baseline is).

This is why the refutations stand while the one positive result does not yet.

## SERIOUS #2: displacement delta is not scale-normalized

`Cd = C + delta * v * sd` where v is a UNIT vector in standardized space. The per-feature
move is therefore delta / sqrt(d). Across the 49 datasets (d from 3 to 1555), delta=4 means
anywhere from **0.101 to 2.309 std per feature - a 23x range**.

So "delta=4 is optimal" is not a meaningful cross-dataset statement, and pooling regret
across datasets mixes very different interventions.

**Fix:** normalize the displacement to each dataset's own spread (e.g. delta x median
distance-to-centroid) rather than implicitly by sqrt(d).

## Net effect on today's conclusions

- **All refutations STAND.** They rest on gap measurements and near-total tie counts,
  neither of which the two serious issues touch.
- **The one positive result (displacement +44.9%, p=0.003, n=49) is NOT yet a finding.**
  It must be re-run (a) inside the REAL pipeline, (b) with scale-normalized delta, and
  (c) after the validity/clipping check clears.

## Retractions issued today (recorded so they are not repeated)

1. Within-family Spearman claim — did not replicate, sign inverted on the dev set.
2. Geometry router AUC 0.891 — rested on 4 positives.
3. HDBSCAN/VaDE structure convergence — refuted, rho = -0.076, p = 0.85; was cherry-picked
   from 2 of 9 datasets.
4. "Generated anomalies cut global-best regret 68%" — the out-of-range validity artifact.
5. Edge-tag crossover "result" — inverted from +4.6% to -4.1% when scaled from 13 to 36
   datasets.
