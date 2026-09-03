# -*- coding: utf-8 -*-
"""Do feature-shuffled synthetic points actually land like real anomalies?

Place the shuffled points on the SAME geometric axes used for the real hard anomalies
(radial percentile vs normal-cluster centres, kNN-density percentile, PCA reconstruction
percentile, between-cluster ratio) and classify each as inside / edge / outside / between.
Compare the shuffled distribution to the REAL hard-anomaly distribution measured earlier
(inside 0.52, between 0.18, edge 0.08, outside 0.23).

If shuffled points land INSIDE existing clusters, they are not anomalies and any correlation is
suspect. If they sit inside-the-manifold / between clusters like the real ones, they are valid.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import val_tabular, val_ucr, sample_dev


def shuffle_candidates(Xn, n_cand, frac=0.3, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape
    base = Xn[rng.integers(0, n, n_cand)].copy()
    ncol = max(1, int(frac * d))
    for r in range(n_cand):
        cols = rng.choice(d, ncol, replace=False)
        base[r, cols] = Xn[rng.integers(0, n, ncol), cols]
    return base


def characterise(Xn, Xa):
    sc = StandardScaler().fit(Xn); Zn = sc.transform(Xn); Za = sc.transform(Xa)
    if Zn.shape[1] > 16:
        p = PCA(n_components=16, random_state=0).fit(Zn); Zn, Za = p.transform(Zn), p.transform(Za)
    K = min(20, max(2, len(Zn) // 30))
    cen = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5).fit(Zn).cluster_centers_

    def radial(Z):
        d = np.sort(np.linalg.norm(Z[:, None, :] - cen[None, :, :], axis=2), axis=1)
        return d[:, 0], d[:, 0] / (d[:, 1] + 1e-9)
    rn, _ = radial(Zn); ra, betw = radial(Za)
    nn = NearestNeighbors(n_neighbors=min(10, len(Zn) - 1)).fit(Zn)
    dn = nn.kneighbors(Zn)[0][:, -1]; da = nn.kneighbors(Za)[0][:, -1]
    pf = PCA(n_components=0.95, random_state=0).fit(Zn)
    rn2 = np.linalg.norm(Zn - pf.inverse_transform(pf.transform(Zn)), axis=1)
    ra2 = np.linalg.norm(Za - pf.inverse_transform(pf.transform(Za)), axis=1)
    pct = lambda v, ref: np.searchsorted(np.sort(ref), v) / len(ref) * 100
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


rows = []
for corpus, loader in [("oddbench", val_tabular), ("ucr", None)]:
    for name in sample_dev(corpus, 10):
        try:
            Xn = val_ucr(name) if corpus == "ucr" else loader(corpus, name)
        except Exception:
            continue
        if len(Xn) < 80:
            continue
        synth = shuffle_candidates(Xn, 200)
        df = characterise(Xn, synth)
        cls = pd.Series(classify(df)).value_counts(normalize=True)
        rows.append(dict(dataset=name, modality=("ts" if corpus == "ucr" else "tabular"),
                         inside=cls.get("inside", 0), edge=cls.get("edge", 0),
                         outside=cls.get("outside", 0), between=cls.get("between", 0),
                         med_radial=df.radial_pct.median(), med_recon=df.recon_pct.median(),
                         med_density=df.density_pct.median()))
R = pd.DataFrame(rows)
print(f"=== where SHUFFLED synthetic points land ({len(R)} datasets) ===")
for mod in ["tabular", "ts", None]:
    sub = R if mod is None else R[R.modality == mod]
    print(f"  {(mod or 'ALL'):8s} inside {sub.inside.mean():.2f}  between {sub.between.mean():.2f}  "
          f"edge {sub.edge.mean():.2f}  outside {sub.outside.mean():.2f}  "
          f"| med radial-pct {sub.med_radial.median():.0f} recon-pct {sub.med_recon.median():.0f}")
print("\n  REAL hard anomalies (measured earlier): inside 0.52  between 0.18  edge 0.08  outside 0.23")
print("\n  match check: shuffled should NOT be ~100% inside (would mean they look normal),")
print("  and should overlap the real inside/between-heavy profile to be valid proxies.")
print(f"\n  fraction of shuffled points detectably OFF the normal manifold "
      f"(median recon-pct > 50): {int((R.med_recon > 50).sum())}/{len(R)} datasets")
