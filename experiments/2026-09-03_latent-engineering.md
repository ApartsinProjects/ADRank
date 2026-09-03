# 2026-09-03 — Does engineering a latent with clearer cluster structure improve selection?

**Status:** completed (premise refuted); UMAP arm confirming
**Motivation:** if the pseudo-task's weakness is that k-means partitions a continuum
(see 2026-09-03_local-global-ceiling-routing.md), maybe a latent with genuine cluster
structure (VaDE, better hyperparameters) would fix it.

## Hypothesis

Chain: better latent -> clearer cluster structure -> better detector selection.
The second link is load-bearing and testable WITHOUT any GPU work.

**KILL CRITERION (pre-registered):** mean within-dataset Spearman(separation, regret) must
be NEGATIVE (better separation -> lower regret) and sign-consistent on >= 2/3 of datasets.

## Why this was run model-free instead of by tuning VaDE

VaDE is expensive, unstable, already a documented null in this project (p=0.23), and it
confounds "richer latent" with "harder optimization". Generating several DIFFERENT latents
per dataset isolates the link at ~0 compute. If separation does not track regret, the line
is closed regardless of which model builds the latent.

## Setup

12 tabular datasets (DAMI + ADBench), balanced by true-best family. Per dataset, up to 7
latents of the same normal-only data: raw standardized, PCA-4/8/16/32, Gaussian random
projection-16, RBF kernel-PCA-16 (plus a UMAP arm, below). Each latent: MiniBatchKMeans
K=20, up to 12 held-out-cluster regimes, 9 detectors, mean pseudo-AUC -> pick -> regret.
Separation measured label-free (silhouette, Calinski-Harabasz, negated Davies-Bouldin).
Labels used only for the true-AUC evaluation column.
Scripts: `latent_sep.py`, `latent_umap.py`, `latent_analyze.py`.

## Result: PREMISE REFUTED (sign is backwards)

Within-dataset Spearman(separation, regret) across latents:

| metric | mean rho | negative on |
|---|---|---|
| silhouette | **+0.189** | 1/5 datasets |
| Calinski-Harabasz | **+0.296** | 1/5 datasets |

Positive rho = better-separated latents give WORSE selection. Consistent on 4/5 datasets
(SpamBase +0.777 on both metrics). KILL CRITERION FAILS on both.

Only 5 of 12 datasets admitted >=3 distinct latents with varying regret; low-dimensional
sets (Pima d=8, Wilt d=5, 14_glass d=7) cannot support many distinct PCA variants. n=5 is
underpowered, but the DIRECTION is consistent and is the opposite of the hypothesis.

## The decisive number: the whole line has a small ceiling

| | regret (6 datasets with pca16) |
|---|---|
| PCA-16 default | 0.0894 |
| pick best-separated latent (label-free) | 0.0816 (+8.7%, 1W/2L/3T, **p=1.000**) |
| **ORACLE best latent per dataset** | **0.0792** |

Perfect per-dataset latent choice buys **~11%** of regret. For contrast, the oracle
family-routing gap is 40%. The label-free proxy for reaching even this 11% is noise.

## Supporting observation: the latent usually does not change the pick

Regret is IDENTICAL across all latents on 20_letter (0.0000), Waveform (0.0000),
Wilt (0.0000), PageBlocks (0.1906), and 5 of 6 latents on Cardiotocography (0.2181).
Changing the representation frequently leaves the selected detector untouched.

## UMAP arm (sharpest test): DECISIVE REFUTATION

UMAP maximizes apparent separation and is documented to EXAGGERATE it, so it is the
strongest test: if the premise held, UMAP should win outright. Two settings
(n_neighbors=15 default, n_neighbors=5 maximally local), 12 datasets.

| | mean silhouette | mean regret |
|---|---|---|
| best non-UMAP latent | 0.236 | 0.0970 |
| best UMAP latent | **0.467 (+97%)** | **0.1050** |

**10 of 12 datasets are an EXACT TIE** (1 better, 1 worse), paired Wilcoxon **p = 1.000**.

Per-dataset: 14_glass 0.0824->0.0824, 20_letter 0.0000->0.0000, 22_magic.gamma
0.0747->0.0747, Annthyroid 0.0626->0.0626, Cardiotocography 0.2181->0.2181, PageBlocks
0.1906->0.1906, Pima 0.1216->0.1216, WDBC 0.0703->0.0703, Waveform 0.0000->0.0000,
Wilt 0.0000->0.0000. Only SpamBase (worse, 0.1240->0.2258) and Stamps (better,
0.2193->0.2140) move at all.

Doubling cluster separation changes the selected detector on 2 of 12 datasets, in
inconsistent directions. Manufacturing cluster structure does not improve selection.

## Caveat (stated honestly)

The oracle is over the 7 generated latents, so it is not a mathematical bound on ALL
possible latents; VaDE could produce something outside their span. But it converges with
three independent results: within-dataset Spearman(effK, regret) = +0.079 across 6 VaDE
configs on 4 datasets (opposite signs per dataset: -0.471, +0.464, -0.176, +0.500); the
project's existing effK-gated VaDE null (0.023 vs 0.027, p=0.23); and the many-ties
observation above.

Also note from the existing VaDE sweeps: the configs producing the MOST cluster structure
(`vade_default`, `vade_joint150`, effK 17-30) win on NO dataset, while `vade_protectclust`
(effK 3-10) wins on 3 of 4. More structure was already associated with worse selection.

## Conclusion

Do NOT invest in VaDE hyperparameter tuning for this purpose. The load-bearing link is
refuted with the sign backwards, and the line's oracle ceiling is ~11% versus a 40% gap
elsewhere.

## Byproduct worth keeping (paper-facing)

**PCA-16 is fixed by fiat in the Method and never ablated** (0 latent-dimension ablations
in the paper). This experiment supplies that missing ablation: across 7 alternative
latents on 12 datasets, none beats PCA-16 significantly (best label-free alternative
1W/2L/3T, p=1.000), and the oracle alternative gains only 11%. PCA-16 is a defensible
default, and there is now evidence for it rather than assertion.
