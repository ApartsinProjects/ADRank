"""Excess-Mass (EM) and Mass-Volume (MV) internal-metric baselines [Goix 2016].

Canonical LABEL-FREE metrics for unsupervised anomaly-detection model selection.
Correct estimator: fit the detector, then score BOTH the data and a uniform
Monte-Carlo sample of the feature bounding box with the SAME fitted model. For a
score threshold t (higher score = more anomalous, so "normal" region is score<t):

  mass(t)   = fraction of DATA with score <= t            (empirical measure of the level set)
  volume(t) = vol_box * fraction of UNIFORM points with score <= t   (Lebesgue volume)
  EM(t)     = sup_t ( mass(t) - t_lambda * volume(t) ) integrated; HIGHER = better.
  MV(alpha) = volume at fixed mass alpha; LOWER = better.

A good detector concentrates the normal mass in a small volume, giving high EM
and low MV. We rank detectors by EM (desc) and MV (asc) and measure regret vs the
true ranking. EM/MV rely on Lebesgue-volume estimation by uniform sampling, which
degrades in high dimension, so on 512/768-dim embeddings the baseline is weak by
construction (a point we report, not hide).
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.pipeline import load_npz_dir, detector_names, _detector_factory
from adrank.ts import load_synthetic_ts

N_MC = 8000
N_T = 100


def em_mv_for_model(model, X):
    """Fit-already model; score data X and uniform box; return (EM_auc, MV_auc)."""
    try:
        s_data = np.asarray(model.decision_function(X), dtype=np.float64)
    except Exception:
        return np.nan, np.nan
    if not np.all(np.isfinite(s_data)) or np.nanstd(s_data) < 1e-12:
        return np.nan, np.nan
    lo, hi = X.min(0), X.max(0)
    rng = np.random.default_rng(0)
    U = rng.uniform(lo, hi, size=(N_MC, X.shape[1]))
    try:
        s_unif = np.asarray(model.decision_function(U), dtype=np.float64)
    except Exception:
        return np.nan, np.nan
    if not np.all(np.isfinite(s_unif)):
        return np.nan, np.nan
    # normalize thresholds over the combined score range
    smin = min(s_data.min(), s_unif.min()); smax = max(s_data.max(), s_unif.max())
    ts = np.linspace(smin, smax, N_T)
    # normal region = score <= t
    mass = np.array([(s_data <= t).mean() for t in ts])       # in [0,1]
    volf = np.array([(s_unif <= t).mean() for t in ts])       # volume fraction in [0,1]
    # EM: excess mass EM(t_lambda) = sup_u (mass(u) - t_lambda * volf(u)); integrate over t_lambda grid
    tl = np.linspace(0.0, 1.0, N_T)
    em_curve = np.array([np.max(mass - lam * volf) for lam in tl])
    em_auc = float(np.trapz(em_curve, tl))                    # higher = better
    # MV: volume as a function of mass; integrate over high-mass alpha in [0.9, 0.999]
    o = np.argsort(mass)
    m_s, v_s = mass[o], volf[o]
    grid = np.linspace(0.9, 0.999, 50)
    v_at = np.interp(grid, m_s, v_s)
    mv_auc = float(np.trapz(v_at, grid))                      # lower = better
    return em_auc, mv_auc


def run(datasets, tag):
    rows = []
    fac = _detector_factory
    for ds in datasets:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(ds.X)); cut = int(0.8 * len(idx))
        Xtr = ds.X[idx[:cut]]
        for det in detector_names(ds.X.shape[1]):
            try:
                model = fac(ds.X.shape[1])[det]()
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    model.fit(Xtr)
                em, mv = em_mv_for_model(model, ds.X)
            except Exception:
                em, mv = np.nan, np.nan
            rows.append({"modality": tag, "dataset": ds.name, "detector": det, "em": em, "mv": mv})
        print(f"  {tag}/{ds.name} done", flush=True)
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--modality", default="tabular"); a = ap.parse_args()
    if a.modality == "tabular":
        dss = load_npz_dir(os.path.join(ROOT, "data", "adbench"))
    elif a.modality == "dami":
        dss = [d for d in load_npz_dir(os.path.join(ROOT, "data", "dami")) if d.y.mean() <= 0.35]
    elif a.modality == "ts":
        dss = load_synthetic_ts(seed=0)
    else:
        dss = load_npz_dir(os.path.join(ROOT, "data", a.modality))
    df = run(dss, a.modality)
    out = os.path.join(ROOT, "results", "raw", f"baseline_emmv_{a.modality}.parquet")
    df.to_parquet(out)
    print(f"[emmv] wrote {out} ({len(df)} rows, EM nan {df.em.isna().mean():.2f})")


if __name__ == "__main__":
    main()
