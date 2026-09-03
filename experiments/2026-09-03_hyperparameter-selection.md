# 2026-09-03 — Can NoMaS select HYPERPARAMETERS, not just detector families?

**Status:** completed. Precondition PASSES (the problem is real and large); the method
FAILS, with a mechanistically-explained, scoped boundary that is paper-facing.

## Why this was the best-motivated idea of the day

- Today's regret decomposition showed **60% of regret is WITHIN-family**, which routing,
  offset debiasing and edge-stratification structurally cannot reach. Hyperparameter
  selection is within-family by construction.
- The marginal-extremity principle that explains all six cross-family failures does NOT
  apply: LOF(k=5) vs LOF(k=50) are both local detectors judged on a local pseudo-task, so
  the task is FAIR for this comparison.
- The paper's own Limitations names it: "combining NoMaS with per-detector hyperparameter
  search is left to future work."

## Setup

25 variants across 6 PyOD detectors on 15 tabular datasets (9 DAMI + 6 ADBench):
LOF and KNN at k in {5,10,20,35,50,100}; IForest n_estimators in {50,100,200};
HBOS n_bins in {5,10,20,50}; PCA n_components in {0.3,0.5,0.9}; CBLOF n_clusters in
{4,8,16}. K=30 clustering, 12 held-out-cluster regimes, single seed.
Script: `hyperparam_pool.py`.

## PRECONDITION: PASSES. Hyperparameters matter MORE than family choice.

Within-detector true-AUC spread across the hyperparameter grid:

| detector | mean spread | max |
|---|---|---|
| **LOF** | **0.2030** | **0.5328** |
| HBOS | 0.0874 | 0.3031 |
| KNN | 0.0873 | 0.2238 |
| PCA | 0.0685 | 0.2208 |
| CBLOF | 0.0312 | 0.0968 |
| IForest | 0.0198 | 0.0806 |

Mean 0.0829. **The library default is the best choice on only 3 of 15 datasets.**

## RESULT: NoMaS is WORSE THAN RANDOM at picking hyperparameters

| selector | regret |
|---|---|
| NoMaS pick | **0.0498** |
| library default | 0.0414 |
| **random pick from the grid** | **0.0379** |

vs default: 32 wins / 32 losses, **p = 0.4458** (an exact coin flip).
Expanding the pool HURTS overall: full 25-variant pool 0.1832 vs defaults-only 0.1652.

## MECHANISM: the pseudo-task's optimum is ANTI-CORRELATED with the truth

| detector | median k truly best | median k NoMaS picks | Spearman(pseudo, true) |
|---|---|---|---|
| **LOF** | **100** | **5** | **-0.387** |
| **KNN** | **100** | **5** | **-0.417** |
| HBOS | 10 | 10 | +0.093 |
| CBLOF | 16 | 16 | +0.000 |
| PCA | - | - | +0.071 |
| IForest | - | - | **+0.333** |

LOF: pseudo picks a larger k than truth on only 1/15 datasets. KNN: 0/15.

**Negative information is worse than none**, which is exactly why NoMaS underperforms a
random draw. IForest, the only detector with a positive correlation, is also the only one
where NoMaS beats the default (0.0039 vs 0.0094).

### Why (and it rhymes with the marginal-extremity principle)

Holding out a whole cluster removes a CONTIGUOUS REGION from training. A held-out point's
5 nearest training neighbours are then far away, so the hole is obvious to small-k methods;
at k=100 the 100th neighbour is distant for every point and the score washes out. Small k
therefore wins the PSEUDO-task.

Real anomalies are usually ISOLATED POINTS among normals, where large-k smoothing gives a
stabler density estimate and wins the TRUE task.

The pseudo-anomaly's **spatial extent** (a whole cluster) does not match real anomalies'
extent (single points), and the optimal neighbourhood size depends on exactly that. This
appears STRUCTURAL: to make a point anomalous by holdout you must remove its neighbourhood,
which necessarily creates a hole the size of that neighbourhood.

**Prediction error to record:** the mechanism was predicted in advance but with the SIGN
BACKWARDS - the expectation was that the pseudo-task would prefer LARGER k. It prefers
much smaller k. The mechanism is real; the direction was wrong.

## Paper-facing value

The Limitations line can move from "left to future work" to a scoped, evidenced boundary:

> NoMaS does not extend to neighbourhood-size selection for density-based detectors: the
> pseudo-task's optimal neighbourhood size is anti-correlated with the true optimum
> (Spearman -0.39 on LOF, -0.42 on KNN over 15 datasets), because a held-out cluster is a
> contiguous void that small-k methods detect sharply while real anomalies are isolated
> points that large-k smoothing serves better. Selection over partition-based detectors
> (Isolation Forest, +0.33) is unaffected.

This is more useful than a fix would have been: it is specific, mechanistic, and it warns
practitioners away from a natural but wrong extension.

## Caveat

Same weakened harness as all of today's regret numbers (K=30, 12 regimes, 1 seed, no
auto-calibration; measured 2.6x worse than the real pipeline). The RELATIVE comparison
(NoMaS vs default vs random on one harness) is fair; the absolute regrets are not
comparable to published numbers. The anti-correlation finding does not depend on the
harness strength.
