# -*- coding: utf-8 -*-
"""Calibrated one-liner triviality test for time series, uni- and multivariate.

WHY THIS EXISTS. The multivariate arm's first criterion declared a series trivial if the
argmax of ANY single-channel one-liner (30 parameterisations x up to 20 channels) landed
inside the labelled region. A placebo test - move the labels to a random position, see if the
criterion still fires - fired at 56%, against 4.7% for the univariate argmax test. With
20 x 30 = 600 statistics, an argmax lands in a 3.7%-density region often enough by luck that
the test was measuring MULTIPLE COMPARISONS, not triviality. So the reported "86% of TSB-AD-M
is not genuinely multivariate" was an artefact and is withdrawn.

THE FIX is a permutation test whose false-positive rate is 5% BY CONSTRUCTION.

  statistic(y) = max over (channel, parameterisation) of AUC(one-liner, y),
                 taken sign-agnostically as max(auc, 1 - auc) because a one-liner may flag
                 the anomaly as either high or low. The max carries the full multiplicity.

  null: circularly SHIFT the label vector by a random offset. A circular shift preserves the
  number, length and shape of the anomaly segments exactly - it only moves them - so the null
  has the same multiple-comparison structure as the observed statistic. K such shifts give a
  null distribution of the SAME max-statistic.

  decision: trivial iff statistic(true labels) >= the (1 - alpha) quantile of the null. Since
  the observed value is compared against its own permutation null, the probability of a
  non-trivial series exceeding the threshold by chance is exactly alpha.

FAST AUC. The AUC of a fixed score vector against any label placement is the Mann-Whitney U:
AUC = (sum of the positives' ranks - P(P+1)/2) / (P*N). Ranks of each statistic are computed
ONCE per series; every placebo is then just a sum of those ranks over the shifted positives,
so K=99 placebos over 600 statistics cost almost nothing beyond the one-time ranking.
"""
import numpy as np
from scipy.stats import rankdata

# same grid as the univariate keogh_trivial, kept faithful for defensibility
_KS = (0, 8, 32, 128, 512)
_CS = (0.0, 1.0, 3.0)


def movmean(a, k):
    """Centred moving average, half-window k//2, edge windows shortened. Vectorised."""
    if k <= 0:
        return a.astype(float)
    h = k // 2
    c = np.cumsum(np.insert(a, 0, 0.0))
    n = len(a)
    i = np.arange(n)
    lo = np.maximum(0, i - h)
    hi = np.minimum(n, i + h + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def movstd(a, k):
    if k <= 0:
        return np.zeros(len(a))
    m = movmean(a, k)
    m2 = movmean(a * a, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def oneliner_stats(x):
    """Yield the one-liner statistic vectors for one channel (labels not involved)."""
    d = np.diff(x, prepend=x[0])
    for base in (np.abs(d), d):
        for k in _KS:
            for u in (0, 1):
                if u and k == 0:
                    continue
                for c in _CS:
                    if k == 0 and c > 0:
                        continue
                    thr = np.zeros(len(base))
                    if u:
                        thr = thr + movmean(base, k)
                    if c and k:
                        thr = thr + c * movstd(base, k)
                    st = base - thr
                    if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12:
                        yield st


def _ranks_for_series(Xc):
    """Average ranks of every (channel, parameterisation) statistic. Shape [M, n]."""
    out = []
    for ch in range(Xc.shape[1]):
        for st in oneliner_stats(Xc[:, ch]):
            out.append(rankdata(st))
    return np.asarray(out) if out else np.empty((0, Xc.shape[0]))


def perm_trivial(Xc, lab, K=99, alpha=0.05, seed=0):
    """Calibrated permutation triviality test.

    Xc : (n, C) array, one column per channel (C=1 for univariate).
    lab: (n,) 0/1 labels.
    Returns (trivial: bool, detail: str, stat: float, thr: float).
    """
    lab = np.asarray(lab, int)
    n = len(lab)
    P = int(lab.sum()); N = n - P
    if P < 3 or N < 3:
        return True, "degenerate", float("nan"), float("nan")
    ranks = _ranks_for_series(Xc)
    if not len(ranks):
        return False, "no_stat", float("nan"), float("nan")
    M = len(ranks)
    const = P * (P + 1) / 2.0
    denom = P * N

    def max_stat(y):
        pos = y.astype(bool)
        rp = ranks[:, pos].sum(1)            # [M] sum of positive ranks per statistic
        auc = (rp - const) / denom
        auc = np.maximum(auc, 1.0 - auc)     # one-liner is sign-agnostic
        return float(auc.max())

    true = max_stat(lab)
    g = np.random.default_rng(seed)
    offs = g.integers(1, n, size=K)
    placebo = np.array([max_stat(np.roll(lab, int(o))) for o in offs])
    thr = float(np.percentile(placebo, 100 * (1 - alpha)))
    return bool(true >= thr), f"auc={true:.3f} thr={thr:.3f} M={M}", true, thr
