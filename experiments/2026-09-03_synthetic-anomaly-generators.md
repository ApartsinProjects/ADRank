# 2026-09-03 — Synthetic anomaly generators for label-free selection (REFUTED except displacement)

**Status:** completed. Decoder-generation, permutation, and filtering all refuted.
Displacement is tracked separately in 2026-09-03_displaced-cluster-pseudo-anomalies.md.

**Goal:** the held-out cluster sits INSIDE the global envelope, so global detectors
correctly score it low and are structurally handicapped (gap -0.93). Can a synthetic
anomaly generator produce pseudo-anomalies that are fair to both detector families?

Reference: global-local z gap of the CURRENT method (real held-out cluster) = **-0.93**.
More negative = global detectors more handicapped. Zero = fair.

## 1. Decoder generation (VaDE-as-generator, tested via cheap proxies)

Proposal: train a generative model, sample in latent space near-but-outside the clusters,
DECODE to produce anomalies. Clustering left untouched.

Tested with a linear decoder (PCA inverse) and a genuine NONLINEAR decoder (small torch
autoencoder = the faithful VaDE proxy).

| generator | mean pAUC | gap |
|---|---|---|
| real | 0.659 | -0.868 |
| gmm1/2/4 (linear decode) | 0.598-0.737 | **-0.08 to -0.02** |
| ae1/2/4 (NONLINEAR decode) | **0.548-0.615** | +0.35 to +0.43 |

**DECODER PULLBACK CONFIRMED.** The AE-decoded points score 0.548 mean pseudo-AUC with 51%
of scores below 0.6: the decoder maps off-cluster latent samples back ONTO the normal
manifold, so they largely are not anomalies. This is intrinsic to decoders and VaDE's
decoder, being better trained, would do it MORE. **The nonlinear decoder also gave
essentially identical regret to the linear one** while producing worse anomalies, so VaDE
as a generator buys nothing. Do not run it.

## 2. Validity: the gap closure was an ARTIFACT

Measured whether generated points MAKE SENSE in the original feature space.

| generator | in_range | int_ok | mean pAUC | gap |
|---|---|---|---|---|
| real | 0.884 | 1.000 | 0.654 | -0.926 |
| pca4 (unconstrained) | **0.416** | **0.021** | 0.734 | **-0.020** |
| clip4 (projected valid) | 1.000 | 1.000 | 0.688 | **-0.450** |
| shuffle (permutation) | 1.000 | 1.000 | 0.645 | **-1.912** |
| xmix | 0.994 | 1.000 | 0.601 | -1.762 |
| qpush | 0.909 | 1.000 | **0.879** | -1.266 |

Unconstrained generated points are only 41.6% in-range and 2.1% integrality-preserving:
they are mostly IMPOSSIBLE records. **Clipping them to legality destroys the gap closure
(-0.020 -> -0.450).** So the closure came from marginal detectors (HBOS/ECOD/COPOD) getting
a free giveaway from out-of-range values, not from measured skill.

**RETRACTION:** an earlier reported "generated anomalies cut global-best regret 68%" was
this artifact. Withdrawn.

## 3. Permutation + filter (valid by construction) — REFUTED

Permutation keeps every value a genuinely observed value, so records are valid in the
original space. Difficulty dial = fraction of features permuted; selection = latent
distance BAND (not a top-q tail, which saturates).

Both knobs WORK: band 50-70 gives mean pAUC 0.58-0.66, band 90-100 gives 0.82-0.83,
saturation stays <=14%. But **all nine conditions make the gap WORSE**:

| frac | band 50-70 | band 70-90 | band 90-100 |
|---|---|---|---|
| 0.25 | -1.585 | -1.437 | -1.315 |
| 0.50 | -1.740 | -1.704 | -1.607 |
| 1.00 | -1.806 | -1.802 | -1.717 |

versus -0.920 for the current method. The gap worsens monotonically with permutation
fraction. Payoff: best condition +19% at **p = 0.844** (4W/4L/2T); most conditions negative.

Mechanism: breaking joint dependence produces LOCALLY sparse points, which is exactly
LOF/KNN territory. Permutation is the wrong KIND of anomaly for the global-detector problem.

## 4. Does the FILTER CHOICE matter? NO (and this kills VaDE-as-filter)

Circularity check: a panel-adjacent filter should hand its relatives an advantage, so an
independent judge (encoder latent) should differ from raw-density judges.

At matched strength q=0.2:

| filter | mean pAUC | gap |
|---|---|---|
| aelat (latent, panel-INDEPENDENT) | 0.784 | -1.712 |
| gmm (PCA density, panel-adjacent) | 0.795 | -1.731 |
| knn (raw density, panel-adjacent) | 0.842 | -1.702 |
| none | 0.643 | -1.892 |
| real | 0.663 | -1.000 |

All three filters agree within **0.03**. The circularity worry was unfounded, and more
importantly: **if a learned latent filter is indistinguishable from a k-NN distance, a
better learned filter (VaDE) will be too.** The filter is not the bottleneck; the generator
is. Filtering only raises detectability (0.64 -> 0.78-0.84) without fixing the bias.

## Implementation note (bug found and fixed)

The latent Gaussian mixture COLLAPSED with "ill-defined empirical covariance" on nearly
every dataset using full covariances. It needed standardized latents, `covariance_type
='diag'`, `reg_covar=1e-3`, and <=6 components to fit at all. This is the same
degenerate-component failure the paper documents for VaDE (effK 2-3), reproduced in a much
simpler model: latent mixtures on these small tabular datasets are intrinsically fragile.

## Harness caveat (applies to all regret numbers above)

These experiments use a WEAKENED re-implementation: K=20, <=10 regimes, 1 seed, 5000-row
subsample, no auto-calibration. Measured against the real pipeline on the same DAMI
datasets, its baseline regret is **0.0875 vs 0.0335, i.e. 2.6x worse** (PageBlocks 0.1906
vs 0.0077, Stamps 0.2323 vs 0.0243). Regret comparisons here are therefore against a
strawman and are NOT evidence about the published method. The GAP measurements are not
affected by this (they concern which family scores higher on the pseudo-task), which is why
the refutations above stand while the payoff numbers do not.

## Conclusion

Every generator except displacement either (a) produces non-anomalies via decoder pullback,
(b) closes the gap only by emitting impossible records that marginal detectors catch for
free, or (c) produces locally-flavoured anomalies that worsen the very bias being targeted.
VaDE is not indicated in either role, generator or filter.
