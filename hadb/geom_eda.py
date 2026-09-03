# -*- coding: utf-8 -*-
"""Where do the REAL (test) anomalies sit relative to the normal manifold?

For each dataset: build the normal manifold (PCA embed + k-means on the normal points), then
place each HARD anomaly (the ones the benchmark actually uses) on four geometric axes, each as
a PERCENTILE against the normal points' own distribution:

  radial   distance to the nearest normal-cluster centre.
           low pct = sits where normals sit; >99 = far beyond the normal cloud.
  density  distance to its 10 nearest NORMAL points (local sparsity).
  recon    PCA reconstruction error on a 95%-variance normal subspace (off-manifold-ness).
  between  d(nearest centre)/d(2nd nearest): ~1 AND high radial = wedged between clusters.

Classification per anomaly:
  OUTSIDE  radial pct > 99            (globally isolated - a trivial anomaly by construction)
  EDGE     radial pct 90-99
  INSIDE   radial pct < 90            (geometrically indistinguishable from a normal point)
  BETWEEN  radial pct >= 90 AND between-ratio > 0.8 (equidistant to two centres)

Hypothesis (stated up front): HARD anomalies - which survived the max|z| and LinRes triviality
filters - should be disproportionately INSIDE, which would explain why edge/cluster
pseudo-anomalies (all at the OUTSIDE/EDGE) fail to predict real detection performance.
For contrast, ALL anomalies (pre-filter) are characterised too.
"""
import os, sys, io, zipfile, re, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN  # noqa: E402
from adrank.ts import _window_features, _window_labels  # noqa: E402


def linres(X, nm, cap=120):
    d = X.shape[1]
    if d < 2 or d > cap:
        return None
    Xn = X[nm]; R = np.zeros((len(X), d))
    for k in range(d):
        oth = [i for i in range(d) if i != k]
        try:
            m = Ridge(alpha=1.0).fit(Xn[:, oth], Xn[:, k])
            res = X[:, k] - m.predict(X[:, oth]); R[:, k] = np.abs(res) / (res[nm].std() + 1e-9)
        except Exception:
            R[:, k] = 0.0
    return R.max(1)


def double_hard(X, y):
    """The benchmark's hard-anomaly set: max|z| AND LinRes both below the 99th pct of normals."""
    nm = y == 0; anom = np.where(y == 1)[0]
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9
    maxz = np.abs((X - mu) / sd).max(1)
    keep_z = maxz[anom] <= np.percentile(maxz[nm], 99)
    lin = linres(X, nm)
    keep_l = (lin[anom] <= np.percentile(lin[nm], 99)) if lin is not None else np.ones(len(anom), bool)
    return anom[keep_z & keep_l], anom


def characterise(Xn, Xa):
    """Return per-anomaly percentiles on the four axes, given normals Xn and anomalies Xa."""
    if len(Xn) < 60 or len(Xa) < 3:
        return None
    sc = StandardScaler().fit(Xn)
    Zn = sc.transform(Xn); Za = sc.transform(Xa)
    if Zn.shape[1] > 16:
        p = PCA(n_components=16, random_state=0).fit(Zn); Zn, Za = p.transform(Zn), p.transform(Za)
    K = min(20, max(2, len(Zn) // 30))
    km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5).fit(Zn)
    cen = km.cluster_centers_

    def radial(Z):
        d = np.linalg.norm(Z[:, None, :] - cen[None, :, :], axis=2)
        ds = np.sort(d, axis=1)
        return ds[:, 0], (ds[:, 0] / (ds[:, 1] + 1e-9))
    rn, _ = radial(Zn); ra, betw = radial(Za)
    # density: distance to 10th nearest NORMAL
    nn = NearestNeighbors(n_neighbors=min(10, len(Zn) - 1)).fit(Zn)
    dn = nn.kneighbors(Zn)[0][:, -1]; da = nn.kneighbors(Za)[0][:, -1]
    # PCA recon error on 95%-variance normal subspace
    pf = PCA(n_components=0.95, random_state=0).fit(Zn)
    rn2 = np.linalg.norm(Zn - pf.inverse_transform(pf.transform(Zn)), axis=1)
    ra2 = np.linalg.norm(Za - pf.inverse_transform(pf.transform(Za)), axis=1)

    def pct(v, ref):
        return np.searchsorted(np.sort(ref), v) / len(ref) * 100
    return pd.DataFrame(dict(radial_pct=pct(ra, rn), density_pct=pct(da, dn),
                             recon_pct=pct(ra2, rn2), between=betw))


def classify(df):
    out = []
    for _, r in df.iterrows():
        if r["radial_pct"] > 99:
            out.append("outside")
        elif r["radial_pct"] >= 90 and r["between"] > 0.8:
            out.append("between")
        elif r["radial_pct"] >= 90:
            out.append("edge")
        else:
            out.append("inside")
    return out


# ---------- reconstruct normals + anomalies per dataset ----------
def tabular(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(),
                        np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) > 6000:
        r = np.random.RandomState(0); k = r.choice(len(X), 6000, replace=False); X, y = X[k], y[k]
    hard, allan = double_hard(X, y)
    return X[y == 0], X[hard], X[allan]


def ucr(name):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "ucr", "UCR_TimeSeriesAnomalyDatasets2021.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name]
    if not cand:
        cand = [n for n in z.namelist() if n.lower().endswith(".txt") and name.split("_")[0] in n]
    fn = cand[0]; m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn, re.I)
    x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()], float)
    a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2
        lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    starts = np.arange(0, len(x) - W + 1, STRIDE)
    Xw, _ = _window_features(x, w=W, stride=STRIDE); yw = _window_labels(lab, starts, w=W, min_count=1)
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    return Xw[yw == 0], Xw[yw == 1], Xw[yw == 1]


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include]
SAMPLE = {"tabular": list(inc[inc.corpus == "oddbench"].dataset[:14]) +
          list(inc[inc.corpus == "ovrbench"].dataset[:8]),
          "ts": list(inc[inc.corpus == "ucr"].dataset[:20])}

allrows = []
for mod, names in SAMPLE.items():
    for name in names:
        try:
            if mod == "ts":
                Xn, Xh, Xa = ucr(name)
            else:
                corp = "oddbench" if name in set(inc[inc.corpus == "oddbench"].dataset) else "ovrbench"
                Xn, Xh, Xa = tabular(corp, name)
        except Exception as e:
            continue
        dh = characterise(Xn, Xh)
        if dh is None:
            continue
        cls = pd.Series(classify(dh)).value_counts(normalize=True)
        allrows.append(dict(dataset=name, modality=mod, n_hard=len(Xh),
                            inside=cls.get("inside", 0), edge=cls.get("edge", 0),
                            outside=cls.get("outside", 0), between=cls.get("between", 0),
                            med_radial=dh.radial_pct.median(), med_recon=dh.recon_pct.median(),
                            med_density=dh.density_pct.median()))

R = pd.DataFrame(allrows)
R.to_csv(os.path.join(S, "HADB_ANOMALY_GEOMETRY.csv"), index=False)
print(f"characterised HARD anomalies over {len(R)} datasets "
      f"({(R.modality=='tabular').sum()} tabular, {(R.modality=='ts').sum()} ts)\n")
print("=== where hard anomalies sit (mean fraction across datasets) ===")
for mod in ["tabular", "ts", None]:
    sub = R if mod is None else R[R.modality == mod]
    tag = mod or "ALL"
    print(f"  {tag:8s} inside {sub.inside.mean():.2f}  edge {sub.edge.mean():.2f}  "
          f"outside {sub.outside.mean():.2f}  between {sub.between.mean():.2f}   "
          f"| median radial-pct {sub.med_radial.median():.0f}, recon-pct {sub.med_recon.median():.0f}")
print("\n  radial-pct interpretation: 50 = sits exactly where normals sit (INSIDE the cloud);")
print("  >99 = beyond the normal cloud (a detector would find it trivially).")
print(f"\n  datasets where a MAJORITY of hard anomalies are INSIDE (radial<90): "
      f"{int((R.inside>0.5).sum())}/{len(R)}")
