"""Time-series adapter for ADRank.

Generate synthetic univariate time series with labeled anomaly regions, extract
sliding-window features to a tabular X matrix, and export as Dataset objects
that plug directly into the existing pipeline.

Ten series cover the standard anomaly types: point (spike / dip), contextual
(normal value in the wrong regime), subsequence (unusual short pattern), trend
change, amplitude change, frequency shift. All univariate for v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .pipeline import Dataset


def _base_seasonal(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 50) + 0.3 * np.sin(2 * np.pi * t / 17)
    x += rng.normal(0, 0.1, size=n)
    return x


def _inject_point_spikes(x, n_anom, seed):
    rng = np.random.default_rng(seed + 1)
    labels = np.zeros(len(x), dtype=int)
    idx = rng.choice(len(x), size=n_anom, replace=False)
    x = x.copy()
    x[idx] += rng.choice([-1, 1], size=n_anom) * (3 + rng.random(n_anom) * 2)
    labels[idx] = 1
    return x, labels


def _inject_subseq(x, n_anom_regions, region_len, seed):
    rng = np.random.default_rng(seed + 2)
    labels = np.zeros(len(x), dtype=int)
    x = x.copy()
    for _ in range(n_anom_regions):
        start = rng.integers(0, len(x) - region_len)
        # inject a burst of higher-frequency noise
        x[start:start + region_len] += rng.normal(0, 1.5, size=region_len)
        labels[start:start + region_len] = 1
    return x, labels


def _inject_trend(x, seed):
    rng = np.random.default_rng(seed + 3)
    labels = np.zeros(len(x), dtype=int)
    x = x.copy()
    # trend anomaly: a segment where the series drifts upward
    start = rng.integers(len(x) // 3, 2 * len(x) // 3)
    length = 500
    trend = np.linspace(0, 3, length)
    x[start:start + length] += trend
    labels[start:start + length] = 1
    return x, labels


def _inject_amplitude(x, seed):
    rng = np.random.default_rng(seed + 4)
    labels = np.zeros(len(x), dtype=int)
    x = x.copy()
    start = rng.integers(len(x) // 3, 2 * len(x) // 3)
    length = 400
    # amplify the amplitude locally
    center = x[start:start + length].mean()
    x[start:start + length] = center + (x[start:start + length] - center) * 3
    labels[start:start + length] = 1
    return x, labels


def _inject_freq_shift(x, seed):
    rng = np.random.default_rng(seed + 5)
    labels = np.zeros(len(x), dtype=int)
    x = x.copy()
    start = rng.integers(len(x) // 3, 2 * len(x) // 3)
    length = 500
    t = np.arange(length)
    # replace with a shifted-frequency segment
    x[start:start + length] = np.sin(2 * np.pi * t / 12) + 0.3 * np.sin(2 * np.pi * t / 5)
    labels[start:start + length] = 1
    return x, labels


def _window_features(x: np.ndarray, w: int = 64, stride: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window of size `w` with `stride` and describe each window with a
    richer ~28-dim feature vector spanning time-domain statistics, difference
    statistics, autocorrelation structure, distribution shape, and spectral energy.
    Returns (features, window_center_indices).
    """
    n = len(x)
    starts = np.arange(0, n - w + 1, stride)
    feats = []
    for s in starts:
        win = x[s:s + w]
        mu = win.mean()
        sd = win.std()
        diffs = np.diff(win)
        d2 = np.diff(diffs)  # second differences

        # autocorrelation at several lags
        def _ac(k):
            if k >= len(win):
                return 0.0
            a = win[:-k] - win[:-k].mean()
            b = win[k:] - win[k:].mean()
            denom = (np.std(win[:-k]) * np.std(win[k:]) * len(a) + 1e-12)
            return float((a * b).sum() / denom)

        # distribution shape (standardized moments)
        z = (win - mu) / (sd + 1e-12)
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)

        # quantiles and robust spread
        q10, q25, q50, q75, q90 = np.quantile(win, [0.1, 0.25, 0.5, 0.75, 0.9])
        iqr = q75 - q25

        # zero-crossing rate of the mean-centered signal
        zc = float((np.sign(win[:-1] - mu) != np.sign(win[1:] - mu)).mean())

        # peak-to-peak and crest factor
        ptp = win.max() - win.min()
        rms = float(np.sqrt((win ** 2).mean()))
        crest = float((np.abs(win).max()) / (rms + 1e-12))

        # spectral: entropy + energy in 4 frequency bands
        fft = np.abs(np.fft.rfft(win - mu))
        power = fft ** 2
        total_power = power.sum() + 1e-12
        p_norm = power / total_power
        p_pos = p_norm[p_norm > 0]
        spec_ent = float(-(p_pos * np.log(p_pos)).sum())
        # split spectrum into 4 bands, fraction of energy each
        nb = len(power)
        band = np.array_split(power, 4)
        band_frac = [float(b.sum() / total_power) for b in band]
        # dominant frequency index (normalized)
        dom_freq = float(np.argmax(power) / (nb + 1e-12))

        feats.append([
            # time-domain (6)
            mu, sd, win.min(), win.max(), ptp, rms,
            # difference stats (4)
            np.abs(diffs).mean(), diffs.std(), np.abs(d2).mean(), d2.std(),
            # autocorrelation (4)
            _ac(1), _ac(2), _ac(5), _ac(10),
            # distribution shape (7)
            skew, kurt, q10, q50, q90, iqr, zc,
            # spectral (7)
            spec_ent, crest, dom_freq, band_frac[0], band_frac[1], band_frac[2], band_frac[3],
        ])
    return np.array(feats, dtype=np.float64), starts + w // 2


def _window_labels(labels: np.ndarray, starts: np.ndarray, w: int, min_count: int = 1) -> np.ndarray:
    """A window is anomalous if it contains at least `min_count` anomalous points.
    min_count=1 works for point anomalies; region-type anomalies naturally exceed this.
    """
    win_lab = []
    for s in starts:
        win_lab.append(int(labels[s:s + w].sum() >= min_count))
    return np.array(win_lab, dtype=int)


def _make_ts_dataset(name: str, x: np.ndarray, labels: np.ndarray,
                     w: int = 64, stride: int = 16) -> Dataset:
    starts = np.arange(0, len(x) - w + 1, stride)
    X, _ = _window_features(x, w=w, stride=stride)
    y = _window_labels(labels, starts, w=w, min_count=1)
    return Dataset(name=name, X=X, y=y)


def load_synthetic_ts(seed: int = 0) -> List[Dataset]:
    """Return 10 synthetic time-series datasets, each already windowed."""
    N = 10000  # length of each raw series -> ~625 windows at w=64, stride=16
    rng = np.random.default_rng(seed)
    datasets: List[Dataset] = []

    # 4 point-spike variants (scale with N)
    for i, n_anom in enumerate([40, 60, 80, 100]):
        x = _base_seasonal(N, seed=seed + 100 + i)
        x, lab = _inject_point_spikes(x, n_anom=n_anom, seed=seed + 100 + i)
        datasets.append(_make_ts_dataset(f"ts_point_spikes_{i}", x, lab))

    # 2 subsequence (scale with N)
    for i, params in enumerate([(6, 120), (10, 80)]):
        n_reg, reg_len = params
        x = _base_seasonal(N, seed=seed + 200 + i)
        x, lab = _inject_subseq(x, n_anom_regions=n_reg, region_len=reg_len, seed=seed + 200 + i)
        datasets.append(_make_ts_dataset(f"ts_subseq_{i}", x, lab))

    # trend / amplitude / frequency shift
    for name, fn in [("ts_trend", _inject_trend),
                     ("ts_amplitude", _inject_amplitude),
                     ("ts_freq_shift", _inject_freq_shift)]:
        x = _base_seasonal(N, seed=seed + 300)
        x, lab = fn(x, seed=seed + 300)
        datasets.append(_make_ts_dataset(name, x, lab))

    # one mixed series
    x = _base_seasonal(N, seed=seed + 400)
    x, lab1 = _inject_point_spikes(x, n_anom=30, seed=seed + 400)
    x, lab2 = _inject_subseq(x, n_anom_regions=4, region_len=80, seed=seed + 401)
    lab = np.clip(lab1 + lab2, 0, 1)
    datasets.append(_make_ts_dataset("ts_mixed", x, lab))

    return datasets
