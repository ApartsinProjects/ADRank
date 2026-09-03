# Scout: prior art for synthetic/injected anomalies in label-free detector selection

Date: 2026-09-03. Two web-researcher passes. Every claim below carries a URL.
Purpose: assess novelty of "displace a real held-out cluster by a swept magnitude and use
it as a label-free detector-ranking criterion", and find citations the paper is missing.

## Verdict

**Novel as a conjunction.** No paper found does: *real* held-out normal cluster +
*additive, swept* displacement + used as a *label-free selection* criterion. The two
ingredients exist separately.

## Citation the paper is MISSING (add regardless of the displacement work)

**Mononito Goswami, Cristian Challu, Laurent Callot, Lenon Minorics, Andrey Kan.
"Unsupervised Model Selection for Time-series Anomaly Detection." ICLR 2023
(notable-top-25%). arXiv:2210.01078.** https://arxiv.org/abs/2210.01078

Closest prior art in SPIRIT: inject synthetic anomalies, rank detectors by AUC on them,
no labels. 17 surrogate metrics (5 prediction-error, 9 synthetic-injection, 3 centrality),
rank-aggregated over 5000+ models. The paper currently cites MetaOD, Goix (EM/MV),
ADBench and Steinbuss but NOT this. A reviewer versed in UOMS would expect it.

## The severity question (settles how we may phrase novelty)

Goswami et al. **do vary injection magnitude, then pool it away**. Their released grid:
`scale: [0.25,0.5,2,4]`, `speedup: [0.25,0.5,2,4]`, `wander: [-0.3,-0.1,0.1,0.3]`,
`noise_std: [0.05]` (single). `evaluate_model_synthetic_anomalies` loops all grid settings
x n_repeats and **concatenates** into one pooled array per anomaly type -> one F1 per type.
No per-severity curve, no severity-vs-selection analysis. The words "severity", "sweep",
"difficulty", "local anomaly", "global anomaly" do not appear in the paper.
Source: [anomaly_parameters.py](https://raw.githubusercontent.com/mononitogoswami/tsad-model-selection/master/src/tsadams/model_selection/anomaly_parameters.py),
[model_selection_utils.py](https://raw.githubusercontent.com/mononitogoswami/tsad-model-selection/master/src/tsadams/utils/model_selection_utils.py)

**BINDING PHRASING RULE.** Do NOT write "prior work uses a single fixed magnitude" — false
and checkable against their public repo. Accurate framing: *prior work varies injection
magnitude only as unreported nuisance randomization and pools across it, treating each
anomaly type as one scalar surrogate metric; severity is never a controlled axis.*
Caveat: the magnitudes come from GitHub; the paper itself gives no numeric values
("a user-defined parameter" throughout, no parameter table).

## Nearest collision on the MECHANISM: ADBench (already cited 17x in our paper)

Han, Hu, Huang, Jiang, Zhao. ADBench, NeurIPS 2022 D&B. https://arxiv.org/abs/2206.09426
All four synthetic types discard real anomalies, fit a **GMM to the normals**, then perturb:
- **local**: scale covariance, Sigma = alpha*Sigma, alpha=5
- **global**: Unif(alpha*min(X^k), alpha*max(X^k)) per feature, alpha=1.1
- **dependency**: vine copula with dependence destroyed
- **clustered**: **scale the mean vector, mu = alpha*mu, alpha=5** <- displaces a cluster

Differentiation to state explicitly (we cite ADBench as our benchmark, so this must be
clean): ADBench samples *synthetic* points from a *fitted GMM*, uses a *single fixed*
alpha, and builds a *labelled benchmark*. We relocate a *real* held-out cluster by an
*additive swept* distance and use it for *label-free selection*. ADBench never sweeps
alpha and never uses these for model selection.

Ancestor: Steinbuss & Bohm, TKDD 15(4) 2021, DOI 10.1145/3441453 (already cited). Same
taxonomy; scaled covariance / copula / expanded bounding box. Verified: they do NOT
displace clusters and do NOT relocate real data.

## Other relevant, mostly non-threatening

- **SWSA** (Selection With Synthetic Anomalies), Fung, Qiu, Li, Rudolph, arXiv 2310.10461.
  Ranks IMAGE detectors label-free using CutPaste + diffusion interpolation (gamma fixed
  at 0.7). Strongest "select detectors with synthetic anomalies" precedent; image domain,
  magnitude-free. VENUE UNCONFIRMED (arXiv footnote suggests ICML; an IEEE Xplore record
  10970745 also exists) - verify before citing.
- **AutoTSAD**, Schmidl, Naumann, Papenbrock, PVLDB 17(11):2987-3002, 2024.
  DOI 10.14778/3681954.3681978. Proxy metrics on synthetic data for TS selection.
- **The Need for UOMS**, Ma, Zhao, Zhang, Akoglu, SIGKDD Explorations 25(1) 2023,
  DOI 10.1145/3606274.3606277. Benchmarks internal label-free criteria on 39 tasks x 297
  models and finds they underperform. Good motivation citation for our gap.
- **ELECT**, Zhao, Zhang, Akoglu, ICDM 2022, arXiv 2211.01834. Transfer-based selection.
- **Emmott et al.**, ODD'13, DOI 10.1145/2500853.2500858. Class-holdout benchmark
  construction controlling point difficulty / relative frequency / clusteredness. This is
  the closest thing to "cluster holdout" but it is labelled BENCHMARK construction, not
  label-free selection, so it does not pre-empt us.
- Training-time injection (NSA arXiv 2109.15222, CutPaste, Outlier Exposure) synthesizes
  anomalies to TRAIN a detector, not to rank detectors. No threat.

## No paper found that

Hides a *cluster of normals* and uses it to rank detectors label-free (our core NoMaS
construct). Nothing surfaced in two passes.

## Unresolved gaps (do not assert these without checking)

- SWSA venue (ICML vs IEEE).
- Steinbuss & Bohm exact page range (ACM returned 403).
- Idan, ECAI 2024 (arXiv 2410.14579) method internals - unknown whether injection-based.
- DOPING and Campos et al. (DAMI 2016) not verified; never surfaced by search.
