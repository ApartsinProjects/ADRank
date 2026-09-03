# Experiment registry — index

One line per experiment. Newest first. Status: completed / in_progress / superseded-by-#N.

## 2026-09-03 — HADB benchmark + selection comparison

| # | File | What | Status |
|---|---|---|---|
| P | [PREREGISTRATION-hadb-selection-comparison](2026-09-03_PREREGISTRATION-hadb-selection-comparison.md) | Locks metric/baselines/test/decision-rules for contribution 3 BEFORE NoMaS runs | pre-results — **commit locks the timestamp** |
| B | (scratchpad `hadb_*` + `HADB_MANIFEST.csv`) | HADB built: 307 datasets / eff-N 302, 2 modalities, 8 audit bugs fixed | benchmark ready |
| S1 | (scratchpad `HADB_SELECTORS_V1.csv`) | Reference selectors: random 0.294, global_fixed 0.194, iforest_random 0.285 (ties random), stability 0.319 (worse) | completed — no deployable baseline beats random |
| I | [synthetic-anomaly-selection-investigation](2026-09-04_synthetic-anomaly-selection-investigation.md) | Full leak-free selector comparison + why pseudo/synthetic anomalies fail | completed — **EM wins**; agreement methods degenerate, pseudo-anomaly (NoMaS) doesn't transfer, shuffle-synthetic has real signal (rho +0.25) but ties EM; real anomalies are DISPLACED, synthetics STRUCTURE-BROKEN |
| — | (`hadb/` code + `hadb/results/`) | The three-way leak-free benchmark: 292 datasets / eff-N 287, EM 0.216 beats random p<0.001 | benchmark ready; code preserved in repo |

## 2026-09-03 — the local/global ceiling session

| # | File | Question | Status | Verdict |
|---|---|---|---|---|
| 1 | [CONCLUSION-marginal-extremity-principle](2026-09-03_CONCLUSION-marginal-extremity-principle.md) | Why is the local/global ceiling structural? | completed | **THE DAY'S FINDING.** One principle explains six independent failures |
| 2 | [local-global-ceiling-routing](2026-09-03_local-global-ceiling-routing.md) | Can label-free family routing close the ceiling? | completed | Oracle headroom **40%** is real; router (AUC 0.577) and offset debias (p=0.86) REFUTED; 60% of regret is within-family |
| 3 | [synthetic-anomaly-generators](2026-09-03_synthetic-anomaly-generators.md) | Can generated pseudo-anomalies rank global detectors fairly? | completed | REFUTED. Decoder pullback (pAUC 0.548), permutation worsens the gap (-1.9), filter choice irrelevant (within 0.03) |
| 4 | [displaced-cluster-pseudo-anomalies](2026-09-03_displaced-cluster-pseudo-anomalies.md) | Does displacing a held-out cluster fix the gap? | completed | **RETRACTED.** +44.9% (p=0.003) was an out-of-range artifact; non-significant once validity-constrained (p=0.751) |
| 5 | [latent-engineering](2026-09-03_latent-engineering.md) | Does a latent with clearer cluster structure help? | completed | REFUTED. UMAP doubles separation (+97%), changes regret on 2/12 datasets, p=1.000. Supplies the paper's **missing PCA-16 ablation** |
| 6 | [hyperparameter-selection](2026-09-03_hyperparameter-selection.md) | Can NoMaS select hyperparameters, not just families? | completed | FAILS, but with a **scoped, mechanistic boundary**: pseudo-optimum is ANTI-correlated with truth (LOF -0.39, KNN -0.42); IForest +0.33 |
| 7 | [external-hard-benchmarks](2026-09-03_external-hard-benchmarks.md) | Does NoMaS hold up when trivial anomalies are removed? | completed | Trivial-anomaly reframing REFUTED (6/153 family flips). TSB-AD negative (-11%) but confounded by the thin window descriptor |
| 8 | [audit-of-todays-experiments](2026-09-03_audit-of-todays-experiments.md) | Are today's numbers trustworthy? | completed | Two serious issues: harness **2.6x weaker** than the real pipeline; delta not scale-normalized (23x range) |
| 9 | [scout_synthetic_anomaly_selection](scout_synthetic_anomaly_selection.md) | Prior art for injection-based label-free selection | completed | Novel as a conjunction; **binding phrasing rule** recorded re: Goswami severity sweep |

## What is paper-ready from this session

Four items need no further experiments:

1. **The 40% oracle bound** (#2) — quantifies the Limitations claim; invariants pass.
2. **Modality scoping** (#2) — global-best share: images 0%, text 1.5%, tabular 15%,
   time-series 18%, DAMI 42%. The bias is tabular-only; on deep embeddings the local
   preference is CORRECT.
3. **The PCA-16 ablation** (#5) — the Method fixes PCA-16 by fiat with no ablation; 7
   alternative latents + UMAP across 12 datasets, none significantly better, oracle
   alternative worth 11%.
4. **The hyperparameter boundary** (#6) — replaces "left to future work" with a specific,
   mechanistic, evidenced limit.

Plus citations to add: Goswami et al. ICLR 2023, Röchner et al. 2025, Emmott et al. 2015,
Pinet et al. 2026. See #7.

## Process lessons (cost: 7 retractions in one day)

Every retraction was a real mechanism whose payoff evaporated under a control:

1. Within-family Spearman — did not replicate, sign inverted on the dev set.
2. Geometry router AUC 0.891 — rested on 4 positives.
3. HDBSCAN/VaDE structure convergence — rho -0.076, p=0.85; cherry-picked from 2 of 9.
4. "Generated anomalies cut global-best regret 68%" — out-of-range validity artifact.
5. Edge-tag crossover — inverted from +4.6% to -4.1% when scaled 13 -> 36 datasets.
6. Displacement +44.9% — out-of-range artifact; p=0.751 once validity-constrained.
7. Trivial-anomaly 3.3x enrichment — fixed `max|z|>=3` threshold artifact; the published
   dimension-adaptive threshold gives 1.4x and changes no family outcomes.

**The two process failures worth not repeating:**

- **The harness baseline was never checked against the real pipeline until late.** It was
  2.6x weaker (0.0875 vs 0.0335 on the same DAMI datasets), which invalidates every regret
  comparison as evidence about NoMaS. This check costs two minutes and should run FIRST.
- **Mechanism findings were reported before their controls ran.** Five of seven retractions
  would have been caught earlier by validity, power, and threshold-sensitivity checks that
  were cheap and available.

**For future runs:** compute `true_auc` ONCE per dataset on full data and cache it (the
per-script resampling caused a 0.066 median / 0.302 worst spread and one best-detector
flip); use the real pipeline as the baseline; scale-normalize any displacement parameter;
never take first-N datasets alphabetically (this bias bit the project twice).
