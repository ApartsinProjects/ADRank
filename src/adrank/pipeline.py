"""ADRank v1 pipeline.

Single-file implementation of:
  - dataset loading (synthetic + local .npz)
  - detector registry (PyOD, fixed defaults)
  - latent embedding + clustering
  - cluster-composite anomaly-likeness score and subset sampling
  - pseudo-AUC evaluation per (dataset, detector, subset)
  - rank aggregation (Borda, mean, variance-weighted)
  - correlation metrics vs the true (label-based) ranking

Labels y are read from the loaders solely to compute the ground-truth ranking
inside `true_rank_from_labels`. Nowhere else in the pipeline is y consumed.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau

# PyOD detectors
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA as PYOD_PCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA

warnings.filterwarnings("ignore")

SEED = 0


# ----------------------------- data loading ---------------------------------

@dataclass
class Dataset:
    name: str
    X: np.ndarray  # (n, d)
    y: np.ndarray  # (n,) with 0=normal, 1=anomaly, ONLY read by true_rank_from_labels


def _synth_gaussian_blobs(n_normal=1500, n_anom=60, d=8, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(4, d))
    X_norm = np.vstack([rng.normal(c, 0.6, size=(n_normal // 4, d)) for c in centers])
    X_anom = rng.uniform(-8, 8, size=(n_anom, d))
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([np.zeros(len(X_norm)), np.ones(n_anom)]).astype(int)
    idx = rng.permutation(len(X))
    return Dataset("synth_blobs", X[idx], y[idx])


def _synth_two_moons_far(n_normal=1500, n_anom=60, d=6, seed=0) -> Dataset:
    from sklearn.datasets import make_moons
    rng = np.random.default_rng(seed)
    Xm, _ = make_moons(n_samples=n_normal, noise=0.05, random_state=seed)
    pad = rng.normal(0, 0.1, size=(n_normal, d - 2))
    X_norm = np.hstack([Xm, pad])
    X_anom = rng.uniform(-3, 3, size=(n_anom, d))
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([np.zeros(len(X_norm)), np.ones(n_anom)]).astype(int)
    idx = rng.permutation(len(X))
    return Dataset("synth_two_moons", X[idx], y[idx])


def _synth_ring(n_normal=1500, n_anom=60, d=5, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n_normal)
    r = 3 + rng.normal(0, 0.1, size=n_normal)
    Xr = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    pad = rng.normal(0, 0.05, size=(n_normal, d - 2))
    X_norm = np.hstack([Xr, pad])
    X_anom = rng.uniform(-1, 1, size=(n_anom, d))  # inside the ring
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([np.zeros(len(X_norm)), np.ones(n_anom)]).astype(int)
    idx = rng.permutation(len(X))
    return Dataset("synth_ring", X[idx], y[idx])


def _synth_high_dim_gauss(n_normal=1500, n_anom=60, d=20, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    X_norm = rng.normal(0, 1, size=(n_normal, d))
    X_anom = rng.normal(0, 3, size=(n_anom, d))  # same mean, higher variance
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(int)
    idx = rng.permutation(len(X))
    return Dataset("synth_high_dim_var", X[idx], y[idx])


def _synth_local_dense(n_normal=1500, n_anom=60, d=5, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    # normal: one big diffuse gaussian; anomalies: tight cluster off-center
    X_norm = rng.normal(0, 1, size=(n_normal, d))
    X_anom = rng.normal(4, 0.1, size=(n_anom, d))
    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(int)
    idx = rng.permutation(len(X))
    return Dataset("synth_local_dense", X[idx], y[idx])


SYNTH_LOADERS = [
    _synth_gaussian_blobs,
    _synth_two_moons_far,
    _synth_ring,
    _synth_high_dim_gauss,
    _synth_local_dense,
]


def load_synthetic(seed: int = SEED) -> List[Dataset]:
    return [f(seed=seed) for f in SYNTH_LOADERS]


def load_npz_dir(path: str) -> List[Dataset]:
    """Load every .npz in `path` with keys 'X' and 'y'. Filename (sans .npz) is dataset name."""
    out: List[Dataset] = []
    if not os.path.isdir(path):
        return out
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".npz"):
            continue
        arr = np.load(os.path.join(path, fn), allow_pickle=False)
        X = np.asarray(arr["X"], dtype=np.float64)
        y = np.asarray(arr["y"], dtype=int).ravel()
        if X.shape[0] < 200 or X.shape[0] > 50000:
            # skip tiny or huge sets in v1 for uniform runtime
            continue
        out.append(Dataset(name=fn[:-4], X=X, y=y))
    return out


# ----------------------------- detectors ------------------------------------

def _detector_factory(n_features: Optional[int] = None) -> Dict[str, Callable[[], object]]:
    # PyOD defaults, fixed random_state where supported
    factory = {
        "IForest": lambda: IForest(random_state=SEED, n_estimators=100),
        "LOF": lambda: LOF(n_neighbors=20, novelty=True),
        "KNN": lambda: KNN(n_neighbors=10, method="largest"),
        "OCSVM": lambda: OCSVM(kernel="rbf", gamma="scale", nu=0.1),
        "ECOD": lambda: ECOD(),
        "COPOD": lambda: COPOD(),
        "HBOS": lambda: HBOS(n_bins=20),
        "PCA": lambda: PYOD_PCA(random_state=SEED),
        "CBLOF": lambda: CBLOF(random_state=SEED, n_clusters=8, alpha=0.9, beta=5),
        "LODA": lambda: LODA(n_bins=10, n_random_cuts=100),
    }
    # Deep detectors (torch) are opt-in via ADRANK_DEEP=1 because they are slow.
    if os.environ.get("ADRANK_DEEP", "") == "1":
        from pyod.models.auto_encoder import AutoEncoder
        from pyod.models.deep_svdd import DeepSVDD
        factory["AutoEncoder"] = lambda: AutoEncoder(
            epoch_num=30, batch_size=256, lr=1e-3, random_state=SEED, verbose=0)
        # DeepSVDD requires the feature count at construction time.
        if n_features is not None:
            factory["DeepSVDD"] = lambda: DeepSVDD(
                n_features=n_features, use_ae=False, epochs=30, batch_size=256,
                random_state=SEED, verbose=0)
    return factory


def detector_names(n_features: Optional[int] = None) -> List[str]:
    exclude = set(os.environ.get("ADRANK_EXCLUDE_DETECTORS", "").split(","))
    exclude.discard("")
    return [n for n in _detector_factory(n_features).keys() if n not in exclude]


def fit_and_score(name: str, X_train: np.ndarray, X_score: np.ndarray) -> Optional[np.ndarray]:
    """Fit a detector on X_train and return decision scores on X_score. Higher = more anomalous.
    Returns None on any failure or if the scores contain non-finite values.
    """
    import contextlib, io
    factories = _detector_factory(n_features=X_train.shape[1])
    try:
        if name not in factories:
            return None
        model = factories[name]()
        # deep detectors (torch) print per-epoch loss; silence it to keep logs sane
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train)
        s = model.decision_function(X_score)
        s = np.asarray(s, dtype=np.float64)
        if not np.all(np.isfinite(s)):
            return None
        if np.nanstd(s) < 1e-12:  # degenerate constant scores
            return None
        return s
    except Exception:
        # Detectors can genuinely fail on degenerate subsets (constant column, singular cov, etc.)
        return None


# ----------------------------- embedding + clustering -----------------------

def embed(X: np.ndarray, dim: int = 16) -> np.ndarray:
    """Standardize and PCA-reduce (or return standardized X if d <= dim)."""
    Xs = StandardScaler().fit_transform(X)
    d = Xs.shape[1]
    if d <= dim:
        return Xs
    pca = PCA(n_components=dim, random_state=SEED)
    return pca.fit_transform(Xs)


def cluster(Z: np.ndarray, K: int, seed: int = SEED) -> np.ndarray:
    """MiniBatchKMeans labels."""
    K_eff = min(K, max(2, Z.shape[0] // 20))
    km = MiniBatchKMeans(n_clusters=K_eff, random_state=seed, n_init=5, batch_size=256)
    return km.fit_predict(Z)


def cluster_composite_scores(Z: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """For each cluster, compute an anomaly-likeness composite in [0,1].
    Higher = more anomaly-like (small + far from other centroids + low local density).
    """
    n = len(Z)
    uniq = np.unique(labels)
    centroids = np.array([Z[labels == c].mean(axis=0) for c in uniq])

    # size
    sizes = np.array([(labels == c).sum() for c in uniq], dtype=float)
    size_component = 1.0 - sizes / n  # smaller = higher

    # mean distance to k nearest OTHER centroids (k = min(3, |clusters|-1))
    if len(uniq) < 2:
        # degenerate single-cluster case (e.g. a collapsed VaDE clustering): no
        # "other" centroids, so the far-from-others component is undefined -> 0.
        far_component = np.zeros(len(uniq))
    else:
        k = max(1, min(3, len(uniq) - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(centroids)
        dists, _ = nn.kneighbors(centroids)
        far_component = dists[:, 1:].mean(axis=1)  # drop self

    # local density: mean distance from cluster members to their centroid
    density_component = np.array([
        np.linalg.norm(Z[labels == c] - centroids[i], axis=1).mean()
        for i, c in enumerate(uniq)
    ])

    def mm(v):
        lo, hi = v.min(), v.max()
        return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)

    composite = (mm(size_component) + mm(far_component) + mm(density_component)) / 3.0

    return pd.DataFrame({
        "cluster": uniq,
        "size": sizes.astype(int),
        "size_component": size_component,
        "far_component": far_component,
        "density_component": density_component,
        "composite": composite,
    }).sort_values("composite", ascending=False).reset_index(drop=True)


# ----------------------------- subset sampling ------------------------------

@dataclass
class Subset:
    subset_id: int
    train_idx: np.ndarray          # indices to fit the detector on
    pseudo_anom_idx: np.ndarray    # held-out clusters treated as "anomalies"
    pseudo_norm_idx: np.ndarray    # held-out points from complement treated as "normals"


def sample_subsets(
    labels: np.ndarray,
    cluster_scores: pd.DataFrame,
    n_points: int,
    M: int,
    target_frac: float = 0.05,
    normal_holdout_frac: float = 0.2,
    selection: str = "composite_top_quartile",  # or 'random' or 'smallest'
    seed: int = SEED,
) -> List[Subset]:
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_points)

    cluster_sizes = dict(zip(cluster_scores["cluster"], cluster_scores["size"]))

    if selection == "composite_top_quartile":
        q = np.quantile(cluster_scores["composite"], 0.75)
        candidate_clusters = cluster_scores.loc[cluster_scores["composite"] >= q, "cluster"].tolist()
    elif selection == "smallest":
        # smallest half of clusters as candidates
        candidate_clusters = cluster_scores.sort_values("size").head(max(2, len(cluster_scores) // 2))["cluster"].tolist()
    elif selection == "random":
        candidate_clusters = cluster_scores["cluster"].tolist()
    elif selection == "hard":
        # clusters embedded AMONG others (small distance to nearest other centroids):
        # these are locally-sparse rather than globally-isolated pseudo-anomalies, so
        # detecting them exercises local density normalization (e.g. LOF over KNN).
        candidate_clusters = cluster_scores.sort_values("far_component").head(
            max(2, len(cluster_scores) // 2))["cluster"].tolist()
    else:
        raise ValueError(selection)

    target_size = max(20, int(target_frac * n_points))

    subsets: List[Subset] = []
    for j in range(M):
        # greedy: pick clusters (without replacement) until pseudo_anom_idx ~ target_size
        remaining = list(candidate_clusters)
        rng.shuffle(remaining)
        chosen: List[int] = []
        cur = 0
        for c in remaining:
            if cur >= target_size:
                break
            chosen.append(c)
            cur += cluster_sizes[c]
        if not chosen:
            continue

        pseudo_anom_mask = np.isin(labels, chosen)
        complement_mask = ~pseudo_anom_mask
        complement_idx = all_idx[complement_mask]

        rng.shuffle(complement_idx)
        n_holdout = max(50, int(normal_holdout_frac * len(complement_idx)))
        n_holdout = min(n_holdout, len(complement_idx) // 3)  # keep enough for training
        pseudo_norm_idx = np.sort(complement_idx[:n_holdout])
        train_idx = np.sort(complement_idx[n_holdout:])

        subsets.append(Subset(
            subset_id=j,
            train_idx=train_idx,
            pseudo_anom_idx=all_idx[pseudo_anom_mask],
            pseudo_norm_idx=pseudo_norm_idx,
        ))
    return subsets


# ----------------------------- pseudo evaluation ----------------------------

def pseudo_auc_for_dataset(
    ds: Dataset,
    K: int = 50,
    M: int = 20,
    selection: str = "composite_top_quartile",
    seed: int = SEED,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    """For each detector and each subset, return pseudo-AUC on (pseudo_anom vs pseudo_norm).

    `precomputed`, if given, is (Z, labels): an externally-computed latent embedding
    and cluster labels (e.g. from VaDE), used instead of the default PCA-16 +
    MiniBatchKMeans. This lets us swap the latent/clustering while holding the rest
    of the pipeline fixed.
    """
    if precomputed is not None:
        Z, labels = precomputed
    else:
        Z = embed(ds.X, dim=16)
        labels = cluster(Z, K=K, seed=seed)
    scores = cluster_composite_scores(Z, labels)
    subsets = sample_subsets(
        labels, scores, n_points=ds.X.shape[0],
        M=M, selection=selection, seed=seed,
    )

    rows = []
    for det in detector_names(ds.X.shape[1]):
        for sub in subsets:
            X_train = ds.X[sub.train_idx]
            X_score_idx = np.concatenate([sub.pseudo_norm_idx, sub.pseudo_anom_idx])
            X_score = ds.X[X_score_idx]
            y_pseudo = np.concatenate([
                np.zeros(len(sub.pseudo_norm_idx)),
                np.ones(len(sub.pseudo_anom_idx)),
            ])
            s = fit_and_score(det, X_train, X_score)
            if s is None or len(np.unique(y_pseudo)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_pseudo, s)
            rows.append({
                "dataset": ds.name,
                "detector": det,
                "subset": sub.subset_id,
                "K": K,
                "selection": selection,
                "n_train": len(sub.train_idx),
                "n_anom": len(sub.pseudo_anom_idx),
                "n_norm": len(sub.pseudo_norm_idx),
                "pseudo_auc": auc,
            })
    return pd.DataFrame(rows)


# ----------------------------- baselines ------------------------------------

def consensus_baseline_rank(ds: Dataset, seed: int = SEED) -> pd.DataFrame:
    """Baseline: score every point by the *mean* per-detector score (a consensus),
    then rank each detector by how well its own score matches the consensus.
    No labels used. This is a standard 'unsupervised model selection' straw man.
    """
    rng = np.random.default_rng(seed)
    n = ds.X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(0.8 * n)
    X_train = ds.X[idx[:cut]]
    X_score = ds.X[idx[cut:]]

    per_det = {}
    for det in detector_names(ds.X.shape[1]):
        s = fit_and_score(det, X_train, X_score)
        if s is None:
            continue
        # z-score so scales are comparable across detectors
        s = (s - s.mean()) / (s.std() + 1e-12)
        per_det[det] = s

    if len(per_det) < 2:
        return pd.DataFrame({"dataset": [ds.name], "detector": [None],
                             "consensus_agreement": [np.nan]})

    S = np.vstack(list(per_det.values()))
    consensus = S.mean(axis=0)
    rows = []
    for det, s in per_det.items():
        # Spearman between detector's score and the consensus
        rho = spearmanr(s, consensus).statistic
        rows.append({"dataset": ds.name, "detector": det, "consensus_agreement": rho})
    return pd.DataFrame(rows)


def cluster_geometry_baseline_rank(ds: Dataset, K: int = 50, M: int = 20,
                                     selection: str = "composite_top_quartile",
                                     seed: int = SEED) -> pd.DataFrame:
    """Sanity baseline: score every detector identically using ONLY the
    cluster-geometry component of the pseudo-eval (mean intra-cluster density
    of the held-out subset). If this baseline matches the pseudo-ranking's
    correlation, the signal is coming from clusters, not from the detectors.
    """
    Z = embed(ds.X, dim=16)
    labels = cluster(Z, K=K, seed=seed)
    scores = cluster_composite_scores(Z, labels)
    subsets = sample_subsets(labels, scores, n_points=ds.X.shape[0],
                             M=M, selection=selection, seed=seed)
    # detector-agnostic score: mean distance of pseudo-anom points to overall centroid
    global_centroid = Z.mean(axis=0)
    per_subset = []
    for sub in subsets:
        dists = np.linalg.norm(Z[sub.pseudo_anom_idx] - global_centroid, axis=1)
        per_subset.append(dists.mean())
    baseline_score = float(np.mean(per_subset))
    # every detector gets the same score -> ranks are ties -> correlation = 0 by construction.
    # We assign each detector this same score so the correlate_ranks step will flag `pred_all_tied`.
    return pd.DataFrame({
        "dataset": ds.name,
        "detector": detector_names(ds.X.shape[1]),
        "cluster_geom_score": baseline_score,
    })


# ----------------------------- true ranking ---------------------------------

def true_rank_from_labels(ds: Dataset, seed: int = SEED) -> pd.DataFrame:
    """Compute the ground-truth ranking: train each detector on the normal-only portion,
    score all points, evaluate ROC-AUC and PR-AUC on the full labeled set.
    This is the ONLY place y is read outside of reporting.
    """
    rng = np.random.default_rng(seed)
    normal_mask = ds.y == 0
    X_normal = ds.X[normal_mask]
    # 80/20 normal split: train on 80% normals; test = 20% normals + all anomalies
    idx = np.arange(len(X_normal))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    X_train = X_normal[idx[:cut]]
    X_test_norm = X_normal[idx[cut:]]
    X_test_anom = ds.X[~normal_mask]
    X_test = np.vstack([X_test_norm, X_test_anom])
    y_test = np.concatenate([np.zeros(len(X_test_norm)), np.ones(len(X_test_anom))])

    rows = []
    for det in detector_names(ds.X.shape[1]):
        s = fit_and_score(det, X_train, X_test)
        if s is None:
            auc, ap = np.nan, np.nan
        else:
            auc = roc_auc_score(y_test, s)
            ap = average_precision_score(y_test, s)
        rows.append({"dataset": ds.name, "detector": det, "true_auc": auc, "true_ap": ap})
    return pd.DataFrame(rows)


# ----------------------------- aggregation ----------------------------------

def _rank_within(df: pd.DataFrame, value_col: str, ascending: bool = False) -> pd.Series:
    """Return 1-based ranks of `value_col` within `df`. Higher value = rank 1 by default."""
    return df[value_col].rank(ascending=ascending, method="average")


def aggregate_pseudo(pseudo_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-(dataset, aggregation) detector ranks and scores.

    Aggregations:
      - mean       : mean pseudo-AUC across subsets
      - borda      : mean of per-subset ranks (across detectors, higher AUC = rank 1)
      - varweight  : mean pseudo-AUC weighted by per-subset variance across detectors
                     (subsets where detectors all agree carry ~0 weight)
    """
    df = pseudo_df.dropna(subset=["pseudo_auc"]).copy()

    out_rows = []

    # 1) mean AUC
    mean_df = df.groupby(["dataset", "detector"])["pseudo_auc"].mean().reset_index()
    for ds, sub in mean_df.groupby("dataset"):
        sub = sub.copy()
        sub["rank"] = _rank_within(sub, "pseudo_auc", ascending=False)
        sub["aggregation"] = "mean"
        sub["score"] = sub["pseudo_auc"]
        out_rows.append(sub[["dataset", "detector", "aggregation", "score", "rank"]])

    # 2) Borda over per-subset ranks
    df["subset_rank"] = df.groupby(["dataset", "subset"])["pseudo_auc"].rank(ascending=False, method="average")
    borda = df.groupby(["dataset", "detector"])["subset_rank"].mean().reset_index()
    for ds, sub in borda.groupby("dataset"):
        sub = sub.copy()
        # lower Borda score = better -> rank 1
        sub["rank"] = sub["subset_rank"].rank(ascending=True, method="average")
        sub["aggregation"] = "borda"
        sub["score"] = -sub["subset_rank"]  # higher score => better, for consistency
        out_rows.append(sub[["dataset", "detector", "aggregation", "score", "rank"]])

    # 3) variance-weighted mean AUC
    per_subset_var = df.groupby(["dataset", "subset"])["pseudo_auc"].var().rename("subset_var").reset_index()
    df2 = df.merge(per_subset_var, on=["dataset", "subset"])
    df2["w"] = df2["subset_var"].fillna(0.0)
    # if all subsets have zero variance, fall back to uniform
    def _wmean(g):
        w = g["w"].values
        v = g["pseudo_auc"].values
        if w.sum() < 1e-12:
            return v.mean()
        return np.average(v, weights=w)
    varw = df2.groupby(["dataset", "detector"]).apply(_wmean, include_groups=False).rename("pseudo_auc").reset_index()
    for ds, sub in varw.groupby("dataset"):
        sub = sub.copy()
        sub["rank"] = _rank_within(sub, "pseudo_auc", ascending=False)
        sub["aggregation"] = "varweight"
        sub["score"] = sub["pseudo_auc"]
        out_rows.append(sub[["dataset", "detector", "aggregation", "score", "rank"]])

    return pd.concat(out_rows, ignore_index=True)


# ----------------------------- rank correlations ----------------------------

def correlate_ranks(true_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, aggregation), compute Spearman, Kendall, top-1 and top-3 hit rates."""
    true = true_df.copy()
    true["true_rank"] = true.groupby("dataset")["true_auc"].rank(ascending=False, method="average")
    merged = agg_df.merge(true[["dataset", "detector", "true_auc", "true_rank"]],
                          on=["dataset", "detector"], how="inner")

    rows = []
    for (ds, agg), sub in merged.groupby(["dataset", "aggregation"]):
        sub = sub.dropna(subset=["true_auc", "score"])
        note = ""
        if len(sub) < 3:
            rho, tau, note = np.nan, np.nan, "n<3"
        else:
            true_spread = sub["true_auc"].max() - sub["true_auc"].min()
            pred_spread = sub["score"].max() - sub["score"].min()
            if true_spread < 1e-6:
                rho, tau, note = np.nan, np.nan, "true_all_tied"
            elif pred_spread < 1e-6:
                rho, tau, note = np.nan, np.nan, "pred_all_tied"
            else:
                rho = spearmanr(sub["true_rank"], sub["rank"]).statistic
                tau = kendalltau(sub["true_rank"], sub["rank"]).statistic
        top1_true = sub.sort_values("true_rank").head(1)["detector"].tolist()
        top1_pred = sub.sort_values("rank").head(1)["detector"].tolist()
        top3_true = set(sub.sort_values("true_rank").head(3)["detector"])
        top3_pred = set(sub.sort_values("rank").head(3)["detector"])
        rows.append({
            "dataset": ds,
            "aggregation": agg,
            "spearman_rho": rho,
            "kendall_tau": tau,
            "top1_hit": int(len(top1_true) > 0 and top1_pred == top1_true),
            "top3_hit_ratio": (len(top3_true & top3_pred) / 3.0) if top3_true else np.nan,
            "n_detectors": len(sub),
            "note": note,
        })
    return pd.DataFrame(rows)
