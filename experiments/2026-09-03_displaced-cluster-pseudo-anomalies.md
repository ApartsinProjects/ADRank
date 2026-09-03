# 2026-09-03 — Displaced-cluster pseudo-anomalies (RETRACTED: out-of-range artifact)

## FINAL VERDICT: REFUTED. The +44.9% was an artifact. Do not pursue.

The validity check (pre-registered, prediction stated in advance) confirmed it:

| delta | arm | in_range | int_ok | gap | regret vs base | p |
|---|---|---|---|---|---|---|
| 1 | raw | **0.357** | **0.047** | -0.074 | +41.7% | **0.003** |
| 1 | clipped | 1.000 | 1.000 | -0.562 | +20.1% | **0.328** |
| 4 | raw | **0.094** | **0.050** | -0.071 | +62.5% | **0.003** |
| 4 | clipped | 1.000 | 1.000 | -0.424 | +22.3% | **0.751** |

At delta=4 only **9.4%** of displaced points are valid records; 95% break integrality.
Marginal detectors (HBOS/ECOD/COPOD) flag impossible values for free. Constrain the points
to be legal and the gap largely returns (-0.07 -> -0.42) and the win becomes
NON-SIGNIFICANT (12W/14L/18T, p=0.751 - more losses than wins).

Second, independent refutation: displacement applied to the weak harness reaches 0.0511 on
DAMI, while the REAL pipeline WITHOUT displacement is already at **0.0335**. Displacement
partially recovers what the ensemble + auto-calibration already provide; it does not add.

**Status:** RETRACTED; superseded by the unified principle below.

---

**Historical record of the (now refuted) result follows.**

**Status:** significant at n=49; ONE artifact check outstanding before it can be believed
**Origin:** user proposal. The held-out cluster sits INSIDE the global envelope, so global
detectors correctly score it low and are structurally handicapped (see
2026-09-03_local-global-ceiling-routing.md). Translating the cluster OUTWARD turns it into
an exterior anomaly that global detectors can win legitimately.

## Design

Only the SCORED cluster is displaced; the TRAINING SET IS UNTOUCHED, so no detector is ever
fit on distorted data. Displacement is applied in the RAW feature space (where detectors
are fit), along the direction from the global mean to the cluster mean, in units of
per-feature std. Sweeping delta gives each detector a competence curve across anomaly types
(interior/local -> far exterior/global).

Pre-registered kill criterion: some delta must close the global-local gap materially
WITHOUT saturating. **PASSED.**

## Mechanism (49 tabular datasets, DAMI + ADBench)

| delta | mean pseudo-AUC | frac > 0.95 | global-local gap |
|---|---|---|---|
| 0.0 | 0.636 | 0.068 | **-0.932** |
| 0.5 | 0.736 | 0.181 | -0.200 |
| 1.0 | 0.787 | 0.246 | -0.107 |
| 2.0 | 0.860 | 0.402 | **-0.055** |
| 4.0 | 0.922 | 0.667 | -0.115 |
| 8.0 | 0.956 | 0.842 | -0.212 |

The family handicap is essentially eliminated at delta=2 and REOPENS at delta=8, exactly as
the saturation account predicts.

## Payoff: SIGNIFICANT, and it STRENGTHENED with power

| variant | regret | vs delta=0 | W/L/T | p |
|---|---|---|---|---|
| delta=0 (current) | 0.1506 | - | - | - |
| delta=0.5 | 0.1018 | +32.4% | 18/5/26 | **0.004** |
| delta=1 | 0.0937 | +37.8% | 21/8/20 | **0.008** |
| delta=2 | 0.1026 | +31.8% | 23/11/15 | **0.009** |
| **delta=4** | **0.0829** | **+44.9%** | 25/13/11 | **0.003** |
| delta=8 | 0.0870 | +42.2% | 25/13/11 | **0.010** |
| spectrum (avg all delta) | 0.1144 | +24.0% | 17/6/26 | **0.006** |

At n=10 this was p=0.383; at n=49 it is p=0.003. Helps BOTH families:
global-best (n=22) 0.1567 -> 0.0626 (-60%), local-best (n=22) 0.0955 -> 0.0538 (-44%).
This contradicts the n=10 sample, where displacement HURT local-best datasets.

## OUTSTANDING ARTIFACT CHECK (do not report the win until this clears)

The validity experiment (2026-09-03, gen_valid.csv) found that unconstrained generated
points were only **41.6% in-range** and **2.1% integrality-preserving**, and that CLIPPING
them to legality **destroyed the gap closure (-0.020 -> -0.450)**. Interpretation: the
closure came from marginal detectors (HBOS/ECOD/COPOD) getting a free giveaway from
impossible values, not from measured skill.

Raw-space displacement has the SAME exposure: moving a cluster 4 sigma outward pushes
features past their observed limits. `displace_validity.py` re-runs delta in {0,1,4} with a
CLIPPED (validity-projected) arm alongside the raw arm, and logs in_range / int_ok.

- If the CLIPPED arm keeps the win -> the effect is real.
- If clipping destroys it -> the win is the artifact and must be retracted.

A second unexplained anomaly: delta=4 gives the best regret while being heavily saturated
(0.922 mean pseudo-AUC, 66.7% above 0.95). A saturated task producing the best selection
contradicts the mechanism story and needs an explanation before belief.

## Novelty (scouted, see scout_synthetic_anomaly_selection.md)

Novel as a conjunction: real held-out cluster + additive SWEPT displacement + label-free
selection. Nearest collision is ADBench's "clustered" generator (mu = alpha*mu, alpha=5
FIXED, GMM-sampled synthetic points, labelled benchmarking) which we already cite 17x.
Nearest in spirit is Goswami et al. ICLR 2023 (injection-for-selection, time series), which
the paper does NOT currently cite and should.

BINDING PHRASING RULE: do NOT claim prior work uses "a single fixed magnitude" - Goswami's
public grid sweeps 4 scale factors, then POOLS across them. Accurate framing: severity is
varied only as unreported nuisance randomization and pooled away; it is never a controlled
axis.
