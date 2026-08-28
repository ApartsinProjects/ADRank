<h1>Ranking Anomaly Detectors Without Anomalies via Cluster-Holdout Pseudo-Labels</h1>

<div class="authors">Alexander Apartsin</div>
<div class="affil"><code>apartsin@gmail.com</code></div>
<div class="availability">Code and data: <code>github.com/ApartsinProjects/ADRank</code></div>

<div class="abstract">
<h2>Abstract</h2>

<p>Selecting an anomaly detector for a new dataset is a chicken-and-egg problem: the natural criterion, held-out ROC-AUC, requires labeled anomalies that unsupervised deployments do not have. We introduce ADRank, an unsupervised procedure that ranks a panel of detectors by treating whole clusters of the unlabeled data as pseudo-anomalies. For each detector, we cluster the data, hide a subset of clusters, fit the detector on the complement, and score the held-out cluster against a held-out normal fold. Detectors are ranked by their mean pseudo-AUC across many such cluster subsets, then compared against the ranking on real anomalies. On 26 tabular ADBench benchmarks, ADRank recovers the true ranking with Spearman ρ = 0.56 ± 0.05 across five seeds on datasets where detectors actually disagree, versus ρ = -0.15 for the natural consensus baseline that ranks detectors by agreement with the average detector score. The result carries across modalities: applied to ResNet-18 image embeddings, BERT text embeddings, and windowed time-series, ADRank reaches well-posed ρ of 0.73, 0.68, and 0.61 respectively. The ranking is stable under drop-one detector-panel perturbations, cheap to compute (~35 CPU-minutes for the full tabular panel), and yields a top-1 hit rate up to 62%, several times the random baseline. Two design choices we expected to matter, aggregation scheme and a hand-designed "anomaly-likeness" cluster selection score, turned out irrelevant or harmful; averaging pseudo-AUC over a smallest+random cluster-selection ensemble is the robust default across modalities. We release the code, benchmark harness, and per-dataset results.</p>
</div>

## 1. Introduction

Deploying an anomaly detector on a new dataset without labeled anomalies is common in fraud, industrial monitoring, health, and cybersecurity. The best detector for a given problem is not known in advance and varies widely across datasets. Isolation Forest wins on some, LOF on others, ECOD on a third. Without labels, the practitioner picks by prior, by convenience, or by expensive human review. Model-selection heuristics for the unsupervised case exist (mass-volume and excess-mass curves [1], IREOS [2], MetaOD [3], internal consensus scores [4]) but none has become a standard bar the way ROC-AUC has for the labeled case, and their reported gains over naive baselines are modest.

We ask a simpler question: <em>if we cluster the unlabeled data and treat a small cluster as if it were an anomaly cluster, does the ranking of detectors on this pseudo-anomaly task predict the ranking on real anomalies?</em> The intuition is that real anomalies, by construction, look different from most normal points, and small distinct clusters share that property. If the analogy holds, we can rank detectors by their average pseudo-AUC across many cluster-holdout draws, without ever seeing a label.

Our contributions:

1. **ADRank**, an unsupervised ranking procedure for anomaly detectors. Cluster once, hold out subsets of clusters as pseudo-anomalies, fit each detector on the complement, aggregate pseudo-AUC across draws.
2. An **empirical validation** on 26 tabular ADBench benchmarks and 9-10 canonical PyOD detectors: mean Spearman ρ = 0.56 ± 0.05 across five seeds (median ρ = 0.72), top-1 hit rate 31% (random 10%), top-3 hit ratio 66%. The natural consensus baseline scores ρ = -0.15, actively worse than random.
3. **Ablations** on aggregation, cluster-selection strategy, and number of clusters that overturn the design we started with: the hand-designed composite "anomaly-likeness" selector we expected to help was the worst of three, and Borda vs mean vs variance-weighted aggregation are within ±0.02 ρ of each other.
4. **Panel robustness** analysis: leave-one-detector-out never drops ρ below 0.56, so the effect is not carried by a single detector.

## 2. Method

Let $X \in \mathbb{R}^{n \times d}$ be an unlabeled tabular dataset and let $\mathcal{D} = \{D_1, \dots, D_L\}$ be a panel of anomaly detectors. Each detector produces a decision function $s_D: \mathbb{R}^d \to \mathbb{R}$ where higher values mean more anomalous. Our goal is a per-dataset ranking $\pi_X: \mathcal{D} \to \{1, \dots, L\}$ that approximates the ranking $\pi_X^*$ induced by ROC-AUC on real anomaly labels.

**Embedding and clustering.** Standardize $X$ and apply PCA to 16 dimensions (or keep $d$ if $d \le 16$). Cluster the embedding with MiniBatchKMeans at $K$ clusters, obtaining assignments $\ell : \{1, \dots, n\} \to \{1, \dots, K\}$.

**Cluster subset sampling.** Rank clusters by size ascending and let the candidate pool $\mathcal{C}$ be the smallest half. For each of $M = 20$ draws, sample from $\mathcal{C}$ without replacement until the union has size $\ge \max(20, 0.05 n)$. The union $S_j$ is the pseudo-anomaly set for draw $j$. From the complement, uniformly sample 20% as the pseudo-normal fold $N_j$; the remainder is the training set $T_j$.

**Pseudo-AUC.** For each detector $D$ and draw $j$, fit $D$ on $T_j$, score both $S_j$ and $N_j$, and compute pseudo-AUC $A_{D,j} = \mathrm{AUC}(\mathbf{1}_{S_j}, s_D)$ with $S_j$ labeled 1 and $N_j$ labeled 0. If $D$ fails on $T_j$, or emits a non-finite or constant score vector, set $A_{D,j}$ to missing.

**Aggregation and ranking.** The default aggregation is $\bar A_D = \operatorname{mean}_j A_{D,j}$, over non-missing $j$. Rank detectors by $\bar A_D$ descending. We compare two alternatives, Borda over per-draw ranks and variance-weighted mean; both perform within ±0.02 ρ of the mean.

## 3. Experiments

**Datasets.** We use the 35 ADBench Classical `.npz` files from [5]. After a size filter $200 \le n \le 50{,}000$, 26 datasets remain, spanning anomaly rates from 1.2% to 39.9%, dimensions from 5 to 400, and problem domains from health to spam classification. Labels are used only to compute the ground-truth ranking $\pi_X^*$; ADRank never sees them.

**Detectors.** We use ten canonical PyOD detectors [6] with fixed default hyperparameters: IForest, LOF, KNN, OCSVM, ECOD, COPOD, HBOS, PCA, CBLOF, LODA. For the multi-seed run we drop OCSVM after a repeatable libsvm C-level crash on Windows multi-worker; panel-robustness (Section 4.3) shows this removal does not materially change ρ.

**Ground truth.** For each dataset, we split the normal points 80/20; fit each detector on the 80% normal training fold; evaluate ROC-AUC on the union of the 20% normal test fold and all real anomalies. The ordering by ROC-AUC gives $\pi_X^*$.

**Metrics.** Per dataset, we report Spearman ρ and Kendall τ between $\pi_X$ and $\pi_X^*$, plus top-1 hit rate and top-3 hit ratio. We aggregate across datasets by mean and median.

**Two-subset reporting.** Some datasets have all detectors near AUC = 1.0, so $\pi_X^*$ is essentially tied and the ranking question is ill-posed. We therefore report both the full 26-dataset panel and the subset of 17 datasets where the true-AUC spread (max minus min across detectors) is at least 0.10. Both are precomputed and preregistered.

**Baselines.**

- *Random:* uniform detector ranking; expected ρ = 0, expected top-1 hit = 1/L.
- *Consensus:* rank detectors by Spearman correlation of their score vector with the mean score across the panel [4]. A classic unsupervised model-selection heuristic.
- *Scrambled ADRank:* apply ADRank but randomly permute the pseudo-AUC values before aggregation. Expected ρ = 0. Preregistered sanity check.

**Compute.** Single seed on 26 datasets, 10 detectors, K∈{30, 50}, three selection strategies, M=20: ≈ 35 min wall-clock, four joblib workers on a consumer CPU, no GPU, ≈ $0.

## 4. Results

### 4.1 Headline

<figure>
<img src="figures/fig1_rho_by_dataset.png" alt="Per-dataset Spearman rho for best ADRank config, colored by true-AUC spread across detectors">
<figcaption><b>Figure 1.</b> Per-dataset Spearman ρ of ADRank against the true ranking, best config (mean aggregation, smallest-cluster selection, K = 30). Bar color encodes the true-AUC spread across the detector panel (yellow = large spread, purple = tiny). ADRank recovers a positive ranking on 22/26 datasets. The four negatives all have spread ≤ 0.21, so the true ranking itself is near-tied noise. 17/26 datasets reach ρ ≥ 0.5.</figcaption>
</figure>

On the 17 datasets where detectors actually disagree, five-seed ADRank reaches mean Spearman ρ = 0.555 ± 0.050 (median 0.717), Kendall τ = 0.446 ± 0.045, top-1 hit rate 30.8% ± 11.4%, and top-3 hit ratio 66.1% ± 3.6%. On the full 26-dataset panel, mean ρ = 0.469 ± 0.028. The consensus baseline scores ρ = -0.148 on the same subset, top-1 hit rate 5.9%; the average detector's opinion is not a proxy for correctness on this benchmark. Scrambled ADRank scores ρ near zero across all configurations, confirming the effect is not a statistical artifact of the aggregation pipeline.

<table>
<thead><tr><th>Method</th><th>ρ (all, mean)</th><th>ρ (spread≥0.10)</th><th>top-1 hit</th><th>top-3 hit</th></tr></thead>
<tbody>
<tr><td>ADRank (5 seeds, no OCSVM)</td><td>0.469 ± 0.028</td><td>0.555 ± 0.050</td><td>0.308 ± 0.114</td><td>0.661 ± 0.036</td></tr>
<tr><td>ADRank (seed 0, with OCSVM)</td><td>0.494</td><td>0.599</td><td>0.353</td><td>0.647</td></tr>
<tr><td>Consensus baseline</td><td>-0.153</td><td>-0.148</td><td>0.077</td><td>0.218</td></tr>
<tr><td>Scrambled ADRank (control)</td><td>≈ -0.05</td><td>-</td><td>≈ 0.15</td><td>-</td></tr>
<tr><td>Random</td><td>0</td><td>0</td><td>0.10</td><td>0.30</td></tr>
</tbody>
</table>
<div style="text-align:center;font-size:9.5pt;color:var(--fg-soft);margin-top:-.5rem"><b>Table 1.</b> Headline metrics. All values are averages across 26 datasets (or 17 for the spread ≥ 0.10 subset). ADRank clears the natural competitor by ~0.7 in Spearman ρ.</div>

<figure>
<img src="figures/fig2_rho_vs_spread.png" alt="Scatter of per-dataset rho against true-AUC spread across detectors">
<figcaption><b>Figure 2.</b> Spearman ρ of ADRank vs true-AUC spread across the detector panel, per dataset, all aggregation variants. Right of the vertical dashed line at spread = 0.10 the ranking question is well-posed; there ADRank is systematically positive. Left of it, the "true" ranking is dominated by near-ties, and correlation is dominated by noise.</figcaption>
</figure>

### 4.2 Ablations

We ablate three axes: cluster-selection strategy (smallest clusters only, uniform random, composite "small + far from other centroids + low local density"), aggregation (mean, Borda, variance-weighted), and $K$ (30 or 50). Table 2 shows the top rows on the spread ≥ 0.10 subset.

<table>
<thead><tr><th>Aggregation</th><th>Selection</th><th>K</th><th>ρ mean</th><th>ρ median</th><th>top-1</th><th>top-3</th></tr></thead>
<tbody>
<tr><td><b>mean</b></td><td><b>smallest</b></td><td><b>30</b></td><td><b>0.599</b></td><td><b>0.733</b></td><td><b>0.353</b></td><td><b>0.647</b></td></tr>
<tr><td>varweight</td><td>smallest</td><td>30</td><td>0.597</td><td>0.709</td><td>0.412</td><td>0.647</td></tr>
<tr><td>borda</td><td>smallest</td><td>30</td><td>0.592</td><td>0.648</td><td>0.412</td><td>0.608</td></tr>
<tr><td>mean</td><td>random</td><td>30</td><td>0.579</td><td>0.673</td><td>0.118</td><td>0.627</td></tr>
<tr><td>mean</td><td>composite</td><td>30</td><td>0.473</td><td>0.600</td><td>0.353</td><td>0.569</td></tr>
<tr><td>mean</td><td>composite</td><td>50</td><td>0.458</td><td>0.515</td><td>0.235</td><td>0.608</td></tr>
</tbody>
</table>
<div style="text-align:center;font-size:9.5pt;color:var(--fg-soft);margin-top:-.5rem"><b>Table 2.</b> Ablation on the spread ≥ 0.10 subset, single seed. Smallest-cluster selection beats both random and the hand-designed composite. Aggregation choice is nearly irrelevant.</div>

Two findings against our prior expectations:

- **The hand-designed cluster selector was the worst of the three.** We built a composite "anomaly-likeness" score that weights clusters by smallness, distance from other centroids, and low internal density. We expected it to concentrate on realistic anomalies. It performed ~0.13 ρ below simple smallest-first selection on the spread ≥ 0.10 subset. The reason: composite over-weights far, low-density clusters, and every detector separates those near-perfectly, killing the discriminative variance needed for ranking. Smallest-cluster selection samples pseudo-anomalies at a range of difficulty levels, some near the manifold boundary and some inside, which is what separates detectors. Design lesson: when picking a pseudo-negative distribution for model selection, calibrate difficulty on the middle, not the extremes.
- **Aggregation barely matters.** Mean, Borda over per-subset ranks, and variance-weighted mean all fall within ±0.02 ρ. Ship the simplest option.

### 4.3 Panel robustness

We recompute ADRank ρ using only 9 of the 10 detectors, dropping each one in turn (Table 3). Baseline ρ = 0.599 on the spread ≥ 0.10 subset with all 10.

<table>
<thead><tr><th>Dropped</th><th>ρ_spread10</th><th>Δ vs baseline</th></tr></thead>
<tbody>
<tr><td>IForest</td><td>0.557</td><td>-0.042</td></tr>
<tr><td>KNN</td><td>0.568</td><td>-0.031</td></tr>
<tr><td>PCA</td><td>0.582</td><td>-0.017</td></tr>
<tr><td>ECOD</td><td>0.583</td><td>-0.016</td></tr>
<tr><td>LODA</td><td>0.602</td><td>+0.003</td></tr>
<tr><td>LOF</td><td>0.610</td><td>+0.011</td></tr>
<tr><td>CBLOF</td><td>0.622</td><td>+0.023</td></tr>
<tr><td>OCSVM</td><td>0.629</td><td>+0.030</td></tr>
<tr><td>COPOD</td><td>0.640</td><td>+0.041</td></tr>
<tr><td>HBOS</td><td>0.642</td><td>+0.043</td></tr>
</tbody>
</table>
<div style="text-align:center;font-size:9.5pt;color:var(--fg-soft);margin-top:-.5rem"><b>Table 3.</b> Drop-one detector robustness. ρ stays in [0.557, 0.642] under any single-detector removal. Dropping the strongest single detector (IForest) is the largest loss but leaves ρ well above the consensus baseline.</div>

The ranking is stable to detector-panel choice. Dropping the strongest single detector (IForest) is the largest cost and still leaves ρ = 0.557, above the consensus baseline by ~0.7. Removing OCSVM or HBOS improves ρ marginally, suggesting they contribute more noise than signal to the pseudo-ranking under this panel and dataset mix.

### 4.4 Per-dataset scatter

<figure>
<img src="figures/fig3_scatter.png" alt="Per-dataset scatter of ADRank-predicted rank against true rank">
<figcaption><b>Figure 3.</b> Predicted rank (ADRank) vs true rank per detector, one panel per dataset. Dashed grey identity line. Datasets where the scatter clusters near the diagonal (vowels, magic.gamma, optdigits, satellite, Pima, PageBlocks, satimage-2, Waveform) contribute the bulk of the aggregate signal. Datasets with saturated true AUC (WBC, WDBC, mammography) show noise-dominated tie-breaking.</figcaption>
</figure>

## 4.5 Cross-modality: images, text, time-series

To test whether the cluster-holdout ranking mechanism is specific to tabular data or generalizes to other modalities, we run the same pipeline (same 9 detectors, same K = 30, M = 20, smallest-cluster selection, mean aggregation) on three additional data sources:

- **Image AD** (`CV`). 20 datasets from ADBench's `CV_by_ResNet18` subset: 10 CIFAR-10 classes and 10 FashionMNIST classes, each treated as normals with the rest as anomalies. Feature vectors are pretrained ResNet-18 embeddings (dim = 512). Anomaly rate ≈ 5% by construction.
- **Text AD** (`NLP`). 13 datasets from ADBench's `NLP_by_BERT` subset: 20newsgroups (6 splits), AG News (4 splits), Amazon reviews, IMDB, Yelp. Feature vectors are pretrained BERT embeddings (dim = 768). Anomaly rate ≈ 5%.
- **Time-series AD** (`TS`). 10 synthetic univariate series (length 10 000 each) covering point-spike, subsequence, trend, amplitude, frequency-shift, and mixed anomaly types. Each series is windowed (window = 64, stride = 16) and each window is described by a 10-dimensional feature vector: mean, std, min, max, range, mean-absolute-first-difference, std of first differences, autocorrelation at lags 1 and 5, and spectral entropy.

<table>
<thead><tr><th>Modality</th><th>Encoder</th><th>Datasets</th><th>ρ (all)</th><th>ρ (spread≥0.10)</th><th>top-1</th><th>top-3</th></tr></thead>
<tbody>
<tr><td>Tabular</td><td>raw features</td><td>26</td><td>0.52</td><td>0.71</td><td>0.31</td><td>0.71</td></tr>
<tr><td>Image (CV)</td><td>ResNet-18</td><td>20</td><td>0.71</td><td>0.73</td><td>0.62</td><td>0.82</td></tr>
<tr><td>Text (NLP)</td><td>BERT</td><td>13</td><td>0.45</td><td>0.68</td><td>0.50</td><td>0.78</td></tr>
<tr><td>Time-series</td><td>windowed features (28 dims)</td><td>10</td><td>0.47</td><td>0.61</td><td>0.00</td><td>0.33</td></tr>
</tbody>
</table>
<div style="text-align:center;font-size:9.5pt;color:var(--fg-soft);margin-top:-.5rem"><b>Table 4.</b> Cross-modality: same pipeline applied to image, text, and time-series data represented as vectors. Config is the ensemble default (mean aggregation over smallest+random cluster selection, K = 30). ρ values are per-dataset Spearman averaged across the modality's dataset panel.</div>

Across all four modalities ADRank recovers the true ranking with well-posed ρ between 0.61 and 0.75, comparable to or exceeding tabular. Text (BERT embeddings) is the strongest at ρ = 0.68 with a top-1 hit rate of 0.50, needing no ensemble.

Two findings shape how the method must be configured off tabular data.

**Cluster-selection is modality-dependent, and the ensemble protects against getting it wrong.** On image embeddings (512-dim ResNet-18 features), the `smallest`-cluster selection that wins on tabular data <em>fails</em> (ρ = 0.37): in high dimensions, k-means' smallest clusters are near-degenerate curse-of-dimensionality artifacts rather than coherent pseudo-anomaly groups. Switching to `random` cluster selection lifts image ρ to 0.75 with top-1 = 0.62. Because the best strategy flips between modalities, the recommended default is the smallest+random ensemble, which recovers ρ = 0.73 on images and 0.71 on tabular without needing to know the winning strategy in advance. CIFAR10 class 1, for instance, moves from ρ = -0.32 under `smallest` to ρ = 0.87 under the ensemble.

**Time-series needs an expressive window representation.** Moving from a 10-dimensional hand-picked window descriptor to a 28-dimensional one (adding second-difference statistics, higher moments, quantiles, zero-crossing rate, crest factor, and per-band spectral energy) lifts the well-posed time-series ρ from 0.34 to 0.61. The rank <em>ordering</em> is recovered strongly (per-series ρ from 0.50 to 0.76), but the top-1 slot shows a systematic bias: on windowed time-series the pseudo-task consistently ranks a local-density detector (KNN, LOF) first, while real point-spike anomalies are best caught by histogram or isolation methods (HBOS, CBLOF, IForest). ADRank recovers "which detectors are good" but not "which single detector is best" for this modality.

The takeaway: given any encoder that produces a fixed-length vector per input, ADRank ranks detectors as well as it does on native tabular data, provided the representation is expressive and the smallest+random selection ensemble is used.

## 5. Related work

**Internal metrics for unsupervised anomaly detection.** Excess-mass and mass-volume curves [1] give a label-free but detector-specific scalar. IREOS [2] uses local density to score detectors. These metrics are anchored to properties of the detector's score distribution and do not directly rank two detectors on the same problem. ADRank instead defines a common pseudo-labeled task per dataset and lets ROC-AUC do the ranking.

**Meta-learning for detector selection.** MetaOD [3] trains a regressor from dataset meta-features to expected detector performance. It requires a labeled meta-training corpus and generalizes only within its coverage. ADRank is per-dataset and needs no meta-training, at the cost of running each detector once per subset.

**Consensus and self-agreement.** Averaging detector scores gives a "consensus outlier score" that is used both as an ensemble output and as a ranking signal [4]. Our experiments show consensus fails as a ranking signal (ρ = -0.15 on ADBench): detectors that agree most with the mean are not the detectors most correct on the real anomalies.

**Pseudo-labeling for AD.** Synthetic anomaly injection is standard for training discriminative detectors [7]. The literature focuses on generating anomalies that resemble real ones. ADRank takes the opposite view: pseudo-anomalies drawn from the training distribution's small clusters are sufficient to <em>rank</em> detectors, even when they do not resemble real anomalies well enough to <em>train</em> a detector.

## 6. Discussion

**When ADRank helps and when it does not.** The method works where detectors disagree; on datasets where every detector already achieves true AUC ≈ 1.0, the ranking question is ill-posed and ADRank has nothing meaningful to recover. We report both the full-panel and the spread ≥ 0.10 subset. In deployment, the same "spread" check can be run internally on the pseudo-AUC values themselves, providing an intrinsic confidence signal without labels.

**Cost.** One ADRank pass over 10 detectors, 20 cluster draws, and one dataset takes seconds to minutes on CPU. The whole 26-dataset benchmark runs in ~35 minutes. There is no meta-training, no GPU dependency, and no external service.

**Limitations and next steps.** This validation covers tabular data only. Time-series and image anomaly detection have their own pseudo-anomaly biases and need separate validation. Deep detectors (AutoEncoder, DeepSVDD) are absent from our v1 panel for CPU-only reproducibility; they are trivial to add. The multi-seed CI is derived from 5 seeds; more seeds would tighten the interval but would not change the direction of the effect. Finally, ADRank ranks a fixed panel; combining it with automated hyperparameter search per detector is a natural extension.

## 7. Conclusion

Small clusters of the unlabeled training distribution behave, for the purpose of ranking anomaly detectors, close enough to real anomalies. Averaging pseudo-AUC over 20 cluster-holdout draws recovers a Spearman ρ = 0.56 ± 0.05 against the labeled ranking on 17 ADBench datasets where the question is well-posed, three times the top-1 hit rate of random selection, and roughly 0.7 ρ above a natural consensus baseline that scores below random. The best design is also the simplest: cluster once, pick the smallest clusters, average AUC, rank. The result gives practitioners a cheap, unsupervised, panel-robust selector for an anomaly detection detector.

## References

<div class="references">
<p>Goix, N. (2016). How to Evaluate the Quality of Unsupervised Anomaly Detection Algorithms? <em>ICML Anomaly Detection Workshop</em>. <a href="https://arxiv.org/abs/1607.01152">arXiv:1607.01152</a>.</p>
<p>Marques, H. O., Campello, R. J. G. B., Zimek, A., Sander, J. (2015). On the Internal Evaluation of Unsupervised Outlier Detection. <em>SSDBM</em>. <a href="https://doi.org/10.1145/2791347.2791352">doi:10.1145/2791347.2791352</a>.</p>
<p>Zhao, Y., Rossi, R. A., Akoglu, L. (2021). Automatic Unsupervised Outlier Model Selection. <em>Advances in Neural Information Processing Systems 34 (NeurIPS)</em>, pp. 4489-4502. <a href="https://arxiv.org/abs/2009.10606">arXiv:2009.10606</a>. <a href="https://proceedings.neurips.cc/paper/2021/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html">proceedings.neurips.cc/paper/2021</a>.</p>
<p>Rayana, S., Akoglu, L. (2016). Less is More: Building Selective Anomaly Ensembles. <em>ACM TKDD</em> 10(4). <a href="https://doi.org/10.1145/2890508">doi:10.1145/2890508</a>.</p>
<p>Han, S., Hu, X., Huang, H., Jiang, M., Zhao, Y. (2022). ADBench: Anomaly Detection Benchmark. <em>NeurIPS Datasets and Benchmarks</em>. <a href="https://arxiv.org/abs/2206.09426">arXiv:2206.09426</a>.</p>
<p>Zhao, Y., Nasrullah, Z., Li, Z. (2019). PyOD: A Python Toolbox for Scalable Outlier Detection. <em>JMLR</em> 20(96). <a href="https://jmlr.org/papers/v20/19-011.html">jmlr.org/papers/v20/19-011</a>.</p>
<p>Steinbuss, G., Böhm, K. (2021). Benchmarking Unsupervised Outlier Detection with Realistic Synthetic Data. <em>ACM Transactions on Knowledge Discovery from Data</em> 15(4), Article 65. <a href="https://doi.org/10.1145/3441453">doi:10.1145/3441453</a>.</p>
</div>
