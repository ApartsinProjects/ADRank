# 2026-09-03 — Can the local/global ceiling be closed by label-free family routing?

**Status:** completed (routing + debiasing refuted); edge-tagging follow-up in progress
**Motivation:** the paper's structural limit (Section 5.4): cluster-holdout pseudo-anomalies
are local, so datasets whose true-best detector is global are under-served.

## Hypothesis

H0 (bound): knowing the true-best detector's FAMILY and picking within it using the existing
NoMaS score removes a large share of regret.
H1 (router): the family is predictable from label-free signal, so the bound is reachable.

**Kill criterion stated in advance:** router held-out AUC < 0.65, or oracle bound < ~0.002 AUC.

## Setup

- Data: `results/oddbench.csv` (187 unseen datasets x 3 seeds, per-detector `true_auc_json`
  and label-free `score_json`); `results/raw/modal_true_allseeds.parquet` +
  `modal_pseudo_allseeds.parquet` (69 dev datasets x 9 detectors x 5 seeds).
- Families per Section 5.4: LOCAL={LOF,KNN,CBLOF}, GLOBAL={HBOS,COPOD,ECOD,PCA},
  OTHER={IForest,LODA}.
- Label-free features only. The `rate` (contamination) column is label-derived and was
  deliberately excluded.
- Scripts: `scratchpad/route_oracle.py`, `route_router.py`, `route_debias.py`,
  `within_family.py`, `within_family_dev.py`, `route_geometry.py`, `edge_tag.py`.

## Invariants (both PASS)

- I1 recomputed regret == stored `regret` column: max|diff| = 1e-16.
- I2 routing over the full detector set == current NoMaS pick: exact match.

## Results

### Oracle bound (H0): CONFIRMED, large

| Benchmark | current regret | oracle family-routed | removable |
|---|---|---|---|
| OddBench (561 rows) | 0.0972 | 0.0584 | **40%** |
| Dev-69 (345 cells)  | 0.0351 | 0.0235 | **33%** |

The remaining 60% / 67% is WITHIN-family and routing cannot touch it.

Selection bias is severe: NoMaS picks a global detector in **8%** of OddBench rows while
the truth is global in **36%**. Family hit rate 43.7%.

### Router (H1): REFUTED

- Features = pseudo-score profile (z-scored), n, d, regime variance. Grouped 5-fold CV by
  dataset (3 seeds/dataset would otherwise leak).
- global-vs-not AUC: **0.577** (logreg), 0.544 (RF). Below the 0.65 kill threshold.
- End metric: predicted routing regret 0.1075 vs 0.1060 current = **1.4% worse**,
  Wilcoxon p = 0.33 (93 wins / 109 losses / 233 ties).

### Family-offset debiasing: REFUTED

One scalar bonus added to the global family's z-scored score (plus one for OTHER).
- In-sample best offset: only +4.2% regret reduction (vs a 40% oracle bound).
- Held-out (5-fold grouped): **-1.6%**, Wilcoxon p = 0.86; fitted offset unstable across
  folds (0.1 to 0.6). Only 52 of 561 rows change at all.
- Mechanism: the family gap has Cohen d = 0.48 (p<0.001) but within-family variance swamps
  the mean shift, so no constant can separate them.

## Retractions (recorded so they are not repeated)

1. **Within-family Spearman claim does NOT replicate.** OddBench gave local +0.42 /
   global +0.15; dev-69 gives local +0.045 / global +0.46 — inverted. Almost certainly
   tie-noise, the same phenomenon the paper documents as the reason Spearman misleads and
   regret is primary. Do not build on rank correlation here; the regret decomposition
   (40% / 33% routable) is the part that replicates.
2. **Geometry router AUC 0.891 is not a result.** The dev-59 subset has only **4**
   global-best datasets, so that LOO AUC rests on 4 positives (`log_d` AUC 0.075 looks like
   "those 4 are low-dimensional"). Untrustworthy; hypothesis remains unresolved.

## Scoping finding (useful, replicable)

Share of dataset-seed cells whose true-best detector is global-family:

| Modality | n datasets | global-best |
|---|---|---|
| Images (CV, ResNet-18) | 20 | 0.0% |
| Text (NLP, BERT) | 13 | 1.5% |
| Tabular (ADBench) | 26 | 14.6% |
| Time-series | 10 | 18.0% |
| Tabular (DAMI) | 10 | 42.0% |

The limitation is a **tabular** problem (worst on DAMI-style semantic outliers) and is
effectively absent on deep embeddings, where individual dimensions carry no standalone
meaning so marginal methods have nothing to grab. On images/text NoMaS's local preference
is CORRECT, not a bug. This also explains why the dev benchmark cannot test the router:
33 of its 69 datasets are image/text.

## Conclusion

The ceiling is real and the headroom is large (40%), but it is NOT reachable by routing:
neither a label-free family classifier nor a constant family offset works, and 60% of the
regret is within-family and out of routing's reach entirely. The productive direction is
therefore a pseudo-task that can rank global detectors AGAINST EACH OTHER, not a selector
between families.

## Follow-up: EDGE vs INTERIOR cluster tagging

Tag each candidate held-out cluster as EDGE or INTERIOR, label-free (centroid
displacement, outer-quantile fraction, marginal percentile, global-PCA reconstruction
ratio). Hypothesis: edge clusters resemble global anomalies, so global detectors can win
them legitimately and become RANKABLE. Differs from the already-failed global-tail regime
because the pseudo-anomaly stays a REAL held-out cluster; only the selection changes.

### H1 mechanism: CONFIRMED (smoke, 4 DAMI datasets, 68 clusters)

Spearman of each edge tag against the global-minus-local pseudo-AUC gap:

| tag | rho | p |
|---|---|---|
| edge_centroid | **+0.409** | 5e-04 |
| edge_marginal | **+0.388** | 1e-03 |
| edge_frac_outer | **+0.372** | 2e-03 |
| edge_recon | **+0.359** | 3e-03 |

Kill threshold was |rho| < 0.15; cleared. The more peripheral the held-out cluster, the
better global detectors do on it relative to local ones.

### H2 payoff: weak in the smoke, and the smoke was biased

Edge-cluster regret 0.0975 vs interior 0.1063 (delta +0.0088, ~8%). BUT the smoke took
the first 5 DAMI datasets ALPHABETICALLY and all 4 usable ones are global-best, so the
local-vs-global contrast was untestable. This repeats the alphabetical-sampling bias
already seen once on OddBench: **do not take first-N alphabetically.** Rerun covers all 16.

Also: using edge clusters INSTEAD of interior ones just swaps one biased task for another.
The idea's value is STRATIFICATION (rank globals on edge groups, locals on interior
groups), which targets the 60% within-family residual that routing could not reach.

### HDBSCAN arm (why it is the better instrument here)

k-means must assign every point and prefers spherical equal-sized clusters, so K=30 on a
blob carves arbitrary wedges and the edge tag partly measures wedge position in a
continuum. HDBSCAN emits a cluster only where density structure exists, so an "edge
cluster" is a genuinely separate peripheral group. It also yields a NOISE set (-1), which
with the edge tag gives a 2x2 of pseudo-anomaly shapes (cluster/noise x interior/edge).
Risk (degenerate density in high dimensions) is anti-correlated with need: the limitation
is a low-dimensional tabular problem, exactly where density clustering is reliable. The
real risk is regime COUNT (fewer groups -> higher variance), which the run measures.
sklearn 1.8.0 provides `sklearn.cluster.HDBSCAN`; the standalone `hdbscan` pkg is absent.

Scripts: `edge_full.py` (dump, both arms), `edge_analyze.py` (scoring variants V0/V1/V2/V3
with the single-stratum invariant V1 == V3).

### H1 replicated on the unbiased 13-dataset DAMI sample

Mix: 7 global-best, 4 local-best, 2 other. All four tags positive:
edge_marginal +0.376 (p=2.5e-09), edge_centroid +0.353 (p=2.5e-08),
edge_frac_outer +0.310 (p=1.2e-06), edge_recon +0.198 (p=2.2e-03).

### H2 crossover: directionally right, NOT significant, outlier-driven

Per-dataset delta = interior_regret - edge_regret (positive = edge helps):

| true best | n | edge | interior | mean delta |
|---|---|---|---|---|
| global | 7 | 0.0886 | 0.1335 | **+0.058** |
| local  | 4 | 0.0533 | 0.0265 | **-0.016** |

Interaction Mann-Whitney **p = 0.109**. Sign consistency is mediocre: global-best deltas
positive in 5/7, local-best negative in only 2/4. The mean is driven by HeartDisease
(delta +0.2026); dropping it takes the global-best mean from +0.058 to +0.034. Annthyroid
(-0.042) and Arrhythmia (-0.010) go the WRONG way despite being global-best.

Verdict: mechanism (H1) is solid; payoff (H2) is promising but not a finding at n=13.
The fix is POWER, not cleverness -> add ADBench tabular (26 datasets).

### HDBSCAN arm: NOT VIABLE on the datasets where the problem lives

Cluster counts (min_cluster_size = max(15, n/100)):

| dataset | n | d | k-means groups | HDBSCAN clusters | noise |
|---|---|---|---|---|---|
| Annthyroid | 7129 | 21 | 30 | 13 | 925 |
| Arrhythmia | 450 | 259 | 8 | **0** | 244 |
| Cardiotocography | 2114 | 21 | 24 | **2** | 45 |
| HeartDisease | 270 | 13 | 6 | **0** | 150 |

HDBSCAN declares most points noise and returns 0-2 clusters, too few to stratify.
**Correction to the pre-run reasoning:** the binding constraint is *n*, not *d*. The
argument that low dimensionality makes density clustering safe was right about
dimensionality and wrong about scale: DAMI sets are 270-2100 points, and at a principled
min_cluster_size HDBSCAN finds no structure. Making it work would need per-dataset
min_cluster_size tuning, which reintroduces the tuning the method exists to avoid.
k-means stratification remains the practical instrument.

### FINAL: edge-stratified scoring REFUTED at adequate power

36 datasets (DAMI + ADBench tabular), balanced 20 local-best / 16 global-best, k-means arm:

| variant | regret | vs V0 |
|---|---|---|
| V0 baseline | 0.1460 | - |
| V3 control (within-group z-score only) | 0.1439 | +1.4% |
| V1 family-stratified | **0.1519** | **-4.1%**, 3W/7L/26T, p=0.169 |
| V2 agnostic best-stratum | 0.1467 | -0.5%, 5W/7L/24T, p=0.695 |

Global-best subset (n=16): V0=0.1644 vs V1=0.1648. No gain where it should help most.

**Sign flip with power:** V1 read +4.6% on 13 DAMI datasets and -4.1% on 36. The DAMI-only
"crossover" (interaction p=0.109, driven by HeartDisease) was small-sample noise. The
z-scoring control gain also shrank from +9% to +1.4%.

HDBSCAN arm qualifies on only 10 datasets (mean 5.9 groups) and shows nothing either
(V1 +3.8% p=0.75, V2 -7.5% p=0.50).

## Overall conclusion

| mechanism | verdict |
|---|---|
| Oracle family-routing headroom | REAL, 40% of regret (OddBench), 33% (dev) |
| Label-free family router | REFUTED (AUC 0.577; routing 1.4% worse, p=0.33) |
| Family-offset debiasing | REFUTED (held-out -1.6%, p=0.86) |
| Edge-tag MECHANISM | CONFIRMED (rho +0.20..+0.38, p to 2.5e-9) |
| Edge-STRATIFIED scoring | REFUTED (-4.1%, p=0.169, n=36) |
| HDBSCAN instrument | Not viable (0 clusters on 3/13 even with adaptive mcs) |
| VaDE latent | Pre-existing null (p=0.23); no new support |

The substantive result: **the headroom is real and large, the edge signal demonstrably
exists in the data, and it still does not convert into better selection.** Four independent
label-free mechanisms fail to reach a 40% oracle gap. This supports the paper's existing
framing of the limit as STRUCTURAL rather than as a fixable gap, now backed by four
falsified fixes instead of one.

Per the wins-only rule this stays in the registry and out of the paper. Its paper-facing
value is that the Limitations claim is now much better evidenced.

### Retraction #3 (recorded)

Proposed that HDBSCAN finding ~0 clusters on Wilt/Waveform converges with VaDE's low effK
there, implying VaDE's "collapse" was correct rather than a weighting artifact.
**Refuted:** Spearman(HDBSCAN k, VaDE effK) = -0.076, p=0.85, n=9 (WDBC: 0 vs 8.67;
PageBlocks: 21 vs 3.17). The two-dataset agreement was cherry-picking. The paper's
weighting-artifact diagnosis stands unchallenged.

### Genuinely new observation worth keeping

HDBSCAN finds NO density-separated clusters on most DAMI sets even at n=4819 (Wilt 0
clusters / 4562 noise; Waveform 0 / 3343). The groups NoMaS holds out are therefore
largely **k-means partitions of a continuum, not genuine density clusters**. This does not
invalidate the method (it works empirically) but it reframes what the pseudo-task is, and
it explains why a group's POSITION is the main thing that distinguishes it.

**Status: completed. Direction closed.**
