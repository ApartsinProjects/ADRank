# 2026-09-03 — CONCLUSION: why the local/global ceiling is structural

One principle explains every negative result from a full day of experiments. This is the
day's actual finding, and it is paper-facing.

## The principle

> **Global detectors detect MARGINAL EXTREMITY. Any pseudo-anomaly that is a VALID record
> is marginally normal by construction, and is therefore invisible to them. So one cannot
> construct valid pseudo-anomalies that fairly rank global detectors.**

The two requirements are in direct opposition:

- "the pseudo-anomaly makes sense in the original space" = every feature value lies inside
  its observed range = **marginally normal**
- "a global detector can catch it" = some feature value is **marginally extreme**

No point satisfies both. HBOS bins per feature; ECOD uses per-feature empirical CDFs;
COPOD models marginal tails. All three score a marginally-normal record as normal, whatever
its joint structure. Meanwhile LOF/KNN measure neighbour distance, so an unusual COMBINATION
is highly anomalous to them. Valid synthetic anomalies are therefore intrinsically
LOCAL-flavoured.

## The principle predicts all six independent failures

| mechanism | result | explained by the principle |
|---|---|---|
| Permutation generators | gap **-1.9** vs -0.92 baseline | every value real => nothing marginally extreme => maximally invisible to global detectors |
| Clipping generated points to valid | gap -0.02 -> **-0.45** | clipping restores marginal normality, destroying the closure |
| Displaced clusters | p=0.003 -> **p=0.751** when clipped | the win lived in the 90% of points that were IMPOSSIBLE records |
| Latent / density filters | all within **0.03** of each other | a filter selects WHICH candidate, never WHAT KIND |
| Decoder generation (VaDE proxy) | pAUC **0.548** | pullback onto the manifold makes points marginally normal AND non-anomalous |
| Family routing / offset debiasing | AUC 0.577; p=0.86 | attacks the 40% between-family share while **60%** of regret is within-family |

## Quantified scope of the limitation (all verified, invariants passing)

- Oracle family-routing headroom: **40%** of regret on OddBench (0.0972 -> 0.0584), 33% on
  the dev benchmark. The other 60% is within-family and routing cannot reach it.
- NoMaS picks a global detector **8%** of the time; the truth is global **36%** of the time;
  family hit rate 43.7%.
- Cost concentration: regret 0.062 where a local detector is best, **0.144** where a global
  one is.
- Modality scoping: global-best share is images **0%**, text **1.5%**, tabular 15%,
  time-series 18%, DAMI tabular **42%**. On deep embeddings NoMaS's local preference is
  CORRECT, not a bug.

## What this is worth to the paper

It converts the Limitations claim from an assertion into a mechanism plus a proof-sketch:
the limit is structural, the reason is identified, and six independent remedies were
falsified. Per the wins-only rule none of the failed remedies belong in the paper; the
PRINCIPLE and the quantified scope do.

Suggested Limitations framing (do not enumerate the failed attempts):
the pseudo-anomaly construction determines which detector family it can rank. Held-out
clusters are interior to the normal distribution, so they measure local competence; global
detectors, which key on marginal extremity, are not measurable by any pseudo-anomaly that
is itself a valid record. This bounds the method on tabular data whose best detector is
global (42% of DAMI), and does not arise on deep embeddings (images 0%, text 1.5%).

## Process record: six retractions in one day

1. Within-family Spearman claim - did not replicate, sign inverted.
2. Geometry router AUC 0.891 - rested on 4 positives.
3. HDBSCAN/VaDE structure convergence - rho -0.076, p=0.85; cherry-picked from 2 of 9.
4. "Generated anomalies cut global-best regret 68%" - out-of-range artifact.
5. Edge-tag crossover - inverted from +4.6% to -4.1% when scaled 13 -> 36 datasets.
6. Displacement +44.9% - out-of-range artifact; non-significant once validity-constrained.

Every one was a real mechanism whose payoff evaporated under a control. The controls that
caught them: power scaling, validity measurement, baseline-fidelity checking, and
replication on independent data. **Two process failures worth not repeating:** the harness
baseline was never checked against the real pipeline until late in the day (it was 2.6x
weaker, invalidating every regret comparison as evidence about NoMaS), and delta was never
scale-normalized (per-feature displacement varied 23x across datasets).
