# -*- coding: utf-8 -*-
"""Label-free internal selection criteria for anomaly-detector model selection.

Each criterion scores every candidate detector WITHOUT labels and a selector picks the
argmax (or argmin). Two families:

  SCORE-MATRIX criteria - need only the [n_points x n_detectors] matrix of anomaly scores:
    consensus (UDR)   Spearman correlation of each detector's ranking with the mean-rank
                      consensus of the whole pool. Pick the most consensus-agreeing detector.
                      (Ma et al. 2023 "UDR"; the idea traces to rank aggregation.)
    model_centrality  average pairwise Spearman correlation of a detector to all others - its
                      centrality in the agreement graph. (Lin & Akoglu ModelCentrality.)
    hits_authority    HITS authority score on the non-negative agreement graph.

  DENSITY criteria - need each detector's scores on the data AND on uniform samples over the
  feature support (Goix et al. 2016, Excess-Mass / Mass-Volume):
    em   Excess-Mass criterion,  HIGHER is better -> pick argmax.
    mv   Mass-Volume criterion,  LOWER  is better -> pick argmin.

  ORIENTATION. PyOD decision_function returns LARGER = more anomalous. EM/MV are defined for a
  normality score (large = normal / dense), so they are computed on the NEGATED anomaly score.
  The score-matrix criteria are rank-correlation based and orientation-invariant up to a global
  flip; all detectors use the same (anomalous-large) orientation, so agreement is well defined.

  VOLUME CONVENTION. EM/MV need the Lebesgue volume of super-level sets. For SELECTION we only
  need the per-dataset ARGMAX/ARGMIN over detectors, and volume_support is a constant shared by
  all detectors on a dataset; we therefore evaluate in the unit box (min-max scaled features,
  volume_support = 1), scoring uniform [0,1]^d samples mapped back to the original feature box.
  This gives a consistent detector ORDERING, which is all the selector consumes. Absolute EM/MV
  magnitudes are not comparable across datasets and are not used as such.
"""
import numpy as np
from scipy.stats import rankdata


# ----------------------------- score-matrix criteria -----------------------------
def _rank_matrix(S):
    """S: [n_points, n_det] anomaly scores -> per-column average ranks (1..n)."""
    return np.apply_along_axis(rankdata, 0, S)


def consensus_scores(S):
    """UDR: Spearman of each detector's ranking vs the mean-rank consensus. Higher = better."""
    R = _rank_matrix(S)                     # [n, m]
    consensus = R.mean(1)                   # mean rank per point
    cr = rankdata(consensus)
    n = R.shape[0]
    out = np.empty(R.shape[1])
    for j in range(R.shape[1]):
        out[j] = _spearman_from_ranks(R[:, j], cr, n)
    return out


def _corr_matrix(S):
    R = _rank_matrix(S)
    Rc = R - R.mean(0)
    denom = np.sqrt((Rc ** 2).sum(0))
    denom[denom == 0] = 1.0
    C = (Rc.T @ Rc) / np.outer(denom, denom)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1, 1)


def model_centrality_scores(S):
    """Average pairwise Spearman correlation to all other detectors. Higher = better."""
    C = _corr_matrix(S)
    m = C.shape[0]
    np.fill_diagonal(C, 0.0)
    return C.sum(1) / max(m - 1, 1)


def hits_authority_scores(S, iters=100):
    """HITS authority on the non-negative agreement graph. Higher = better."""
    C = _corr_matrix(S)
    A = np.clip(C, 0, None)                 # non-negative adjacency
    np.fill_diagonal(A, 0.0)
    m = A.shape[0]
    auth = np.ones(m)
    for _ in range(iters):
        hub = A @ auth
        auth = A.T @ hub
        nrm = np.linalg.norm(auth)
        if nrm == 0:
            break
        auth = auth / nrm
    return auth


def _spearman_from_ranks(ra, rb, n):
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


# ----------------------------- EM / MV (Goix 2016) -----------------------------
def em_mv_scores(score_fns, X_eval, n_generated=100000, t_count=1000,
                 alpha_min=0.9, alpha_max=0.999, seed=0):
    """Compute EM and MV for a list of detector scoring callables.

    score_fns : list of callables f(Xarray)->anomaly scores (LARGER = more anomalous).
    X_eval    : [n, d] the data the detectors are evaluated on.
    Returns (em[m], mv[m]) arrays. EM higher = better; MV lower = better.
    """
    rng = np.random.default_rng(seed)
    n, d = X_eval.shape
    lo, hi = X_eval.min(0), X_eval.max(0)
    rng_span = np.where(hi > lo, hi - lo, 1.0)
    U = lo + rng.random((n_generated, d)) * rng_span
    vol = 1.0                                    # unit-box convention (see module docstring)

    t = np.linspace(0, 100, t_count)
    alpha = np.linspace(alpha_min, alpha_max, t_count)
    em_out, mv_out = [], []
    for f in score_fns:
        s_X = -np.asarray(f(X_eval), float)      # NORMALITY score
        s_U = -np.asarray(f(U), float)
        em_out.append(_em_auc(t, vol, s_U, s_X, n_generated))
        mv_out.append(_mv_auc(alpha, vol, s_U, s_X, n_generated))
    return np.array(em_out), np.array(mv_out)


def _em_auc(t, vol, s_unif, s_X, n_gen, t_max=0.9):
    n = s_X.shape[0]
    EM = np.zeros(t.shape[0]); EM[0] = 1.0
    for u in np.unique(s_X):
        EM = np.maximum(EM, (s_X > u).mean() - t * (s_unif > u).sum() / n_gen * vol)
    amax = np.argmax(EM <= t_max)
    amax = amax if amax > 0 else t.shape[0] - 1
    return float(np.trapz(EM[:amax + 1], t[:amax + 1]))


def _mv_auc(alpha, vol, s_unif, s_X, n_gen):
    n = s_X.shape[0]
    order = np.argsort(s_X)                       # ascending
    mv = np.zeros(alpha.shape[0])
    mass, cpt, u = 0.0, 0, s_X[order[-1]]
    for i in range(alpha.shape[0]):
        while mass < alpha[i] and cpt < n:
            cpt += 1
            u = s_X[order[-cpt]]
            mass = cpt / n
        mv[i] = (s_unif >= u).sum() / n_gen * vol
    return float(np.trapz(mv, alpha))


# ----------------------------- convenience: pick from a score matrix -----------------------------
CRITERIA_MATRIX = {
    "consensus": (consensus_scores, "max"),
    "model_centrality": (model_centrality_scores, "max"),
    "hits_authority": (hits_authority_scores, "max"),
}
