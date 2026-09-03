# -*- coding: utf-8 -*-
"""How do SYNTHETIC (shuffle) anomalies differ from REAL hard anomalies, and does the
difference explain where shuffle-selection fails?

Per dataset: reconstruct normals, REAL hard anomalies, and SYNTHETIC shuffle anomalies; place
both anomaly sets on the same axes (radial percentile vs cluster centres, PCA reconstruction
percentile = off-manifold, max|z| = marginal extremity); then correlate the dataset-level
shuffle-selection signal (within-dataset Spearman of synth_AUC vs true) against the geometry
of its REAL anomalies. If shuffle fails where real anomalies are radially DISPLACED, that is
the mechanism: shuffle makes structure-broken-but-central points, real hard anomalies are more
displaced.
"""
import os, sys, io, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL, sample_dev, sample_holdout
from hadb_ts_final import W, STRIDE, MAX_LEN
from adrank.ts import _window_features, _window_labels
import zipfile, re


def linres(X, nm, cap=120):
    d = X.shape[1]
    if d < 2 or d > cap:
        return None
    Xn = X[nm]; R = np.zeros((len(X), d))
    for k in range(d):
        oth = [i for i in range(d) if i != k]
        try:
            m = Ridge(alpha=1.0).fit(Xn[:, oth], Xn[:, k]); R[:, k] = np.abs(X[:, k] - m.predict(X[:, oth])) / (Ridge(alpha=1.0).fit(Xn[:, oth], Xn[:, k]).predict(Xn[:, oth]).std() + 1e-9)
        except Exception:
            R[:, k] = 0.0
    return R.max(1)


def load_tab(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X);
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    nm = y == 0; anom = np.where(y == 1)[0]
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9; maxz = np.abs((X - mu) / sd).max(1)
    kz = maxz[anom] <= np.percentile(maxz[nm], 99)
    hard = anom[kz]
    return X[nm], X[hard], maxz


def load_ucr(name):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "ucr", "UCR_TimeSeriesAnomalyDatasets2021.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith(".txt") and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]; m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn)
    x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()])
    a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2
        lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    st = np.arange(0, len(x) - W + 1, STRIDE); Xw, _ = _window_features(x, w=W, stride=STRIDE)
    yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    nm = Xw[yw == 0]; mu, sd = nm.mean(0), nm.std(0) + 1e-9
    return nm, Xw[yw == 1], np.abs((Xw - mu) / sd).max(1)


def gen_shuffle(Xn, ns, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d))
    for r in range(ns):
        b = Xn[rng.integers(n)].copy(); k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        c = rng.choice(d, k, replace=False); out[r] = b; out[r, c] = Xn[rng.integers(n)][c]
    return out


def geom(Xn, Xa):
    sc = StandardScaler().fit(Xn); Zn = sc.transform(Xn); Za = sc.transform(Xa)
    if Zn.shape[1] > 16:
        p = PCA(16, random_state=0).fit(Zn); Zn, Za = p.transform(Zn), p.transform(Za)
    cen = MiniBatchKMeans(min(20, max(2, len(Zn) // 30)), random_state=0, n_init=5).fit(Zn).cluster_centers_
    rn = np.linalg.norm(Zn[:, None] - cen[None], axis=2).min(1); ra = np.linalg.norm(Za[:, None] - cen[None], axis=2).min(1)
    pf = PCA(n_components=0.95, random_state=0).fit(Zn)
    e_n = np.linalg.norm(Zn - pf.inverse_transform(pf.transform(Zn)), axis=1)
    e_a = np.linalg.norm(Za - pf.inverse_transform(pf.transform(Za)), axis=1)
    pct = lambda v, ref: np.searchsorted(np.sort(ref), v) / len(ref) * 100
    return np.median(pct(ra, rn)), np.median(pct(e_a, e_n))


def synth_auc(Xn, syn, pool):
    g = np.random.default_rng(0); idx = np.arange(len(Xn)); g.shuffle(idx); cut = int(0.7 * len(idx)); tr, ho = idx[:cut], idx[cut:]
    if len(tr) < 30 or len(ho) < 10:
        return {}
    Xe = np.vstack([Xn[ho], syn]); ye = np.r_[np.zeros(len(ho)), np.ones(len(syn))]; o = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xn[tr])
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                o[vn] = float(roc_auc_score(ye, s))
        except Exception:
            pass
    return o


DS = [("oddbench", "hadb_oddbench.csv", TAB_POOL, load_tab, sample_dev("oddbench", 8) + sample_holdout("ovrbench", 0)),
      ("ovrbench", "hadb_ovrbench.csv", TAB_POOL, load_tab, sample_holdout("ovrbench", 8)),
      ("ucr", "hadb_ts_ucr.csv", TS_POOL, load_ucr, sample_dev("ucr", 8))]

rows = []
for corpus, csv, pool, loader, names in DS:
    D = pd.read_csv(os.path.join(S, csv)); D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
    apn = D.groupby(["dataset", "variant"]).ap_norm.mean()
    for name in names:
        try:
            Xn, Xhard, maxz = loader(corpus, name) if corpus != "ucr" else loader(name)
        except Exception:
            continue
        if len(Xn) < 80 or len(Xhard) < 5:
            continue
        syn = gen_shuffle(Xn, 150)
        r_rad, r_rec = geom(Xn, Xhard)          # REAL anomaly geometry
        s_rad, s_rec = geom(Xn, syn)            # SYNTHETIC geometry
        sa = synth_auc(Xn, syn, pool)
        try:
            ap = apn.loc[name]
        except Exception:
            continue
        common = [v for v in sa if v in ap.index]
        if len(common) < 5:
            continue
        sv = np.array([sa[v] for v in common]); tv = np.array([ap[v] for v in common])
        if np.std(sv) == 0 or np.std(tv) == 0:
            continue
        rho = spearmanr(sv, tv).statistic
        rows.append(dict(dataset=name, corpus=corpus, rho=rho,
                         real_radial=r_rad, real_recon=r_rec, synth_radial=s_rad, synth_recon=s_rec))
R = pd.DataFrame(rows)
print(f"=== REAL vs SYNTHETIC anomaly geometry ({len(R)} datasets) ===")
print(f"  REAL      median radial-pct {R.real_radial.median():.0f}   recon-pct {R.real_recon.median():.0f}")
print(f"  SYNTHETIC median radial-pct {R.synth_radial.median():.0f}   recon-pct {R.synth_recon.median():.0f}")
print(f"\n  => real anomalies are more DISPLACED (radial), synthetics more OFF-MANIFOLD (recon).")
print(f"\n=== does real-anomaly displacement predict shuffle FAILURE? ===")
print(f"  corr(real_radial, shuffle_rho)          = {spearmanr(R.real_radial, R.rho).statistic:+.3f}")
print(f"  corr(real_radial - synth_radial, rho)   = {spearmanr(R.real_radial - R.synth_radial, R.rho).statistic:+.3f}")
print(f"  (negative => shuffle works WORSE when real anomalies are more radially displaced than the synthetics)")
lo = R[R.real_radial > R.real_radial.median()]; hi = R[R.real_radial <= R.real_radial.median()]
print(f"\n  shuffle rho on DISPLACED-real datasets (radial>med): {lo.rho.mean():+.3f}")
print(f"  shuffle rho on CENTRAL-real  datasets (radial<=med): {hi.rho.mean():+.3f}")
