# -*- coding: utf-8 -*-
"""Do GLOBAL anomalies (where a global detector wins, local loses) have NORMAL neighbours,
or are they CLUSTERED with other anomalies?

For datasets grouped by true-best family, characterise the real hard anomalies:
  nn_normal_frac   fraction of each anomaly's 10 nearest neighbours (in the normal+anomaly
                   pool) that are NORMAL points. High => normal neighbours; low => clustered
                   with other anomalies.
  local_dens_pct   the anomaly's 10th-NN distance to NORMALS, as a percentile of the normals'
                   own 10th-NN distances. ~50 => locally as dense as a normal (local detector
                   sees nothing); high => locally sparse (local detector should catch it).
  global_rare      median over anomalies of the max-over-features marginal extremity
                   (|CDF-percentile - 50| x 2). High => globally rare in some marginal (HBOS/
                   ECOD territory).
"""
import os, sys, zipfile, re, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN
from adrank.ts import _window_features, _window_labels

LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")


def tab(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 6000:
        r = np.random.RandomState(0); k = r.choice(len(X), 6000, replace=False); X, y = X[k], y[k]
    nm = y == 0; anom = np.where(y == 1)[0]
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9; mz = np.abs((X - mu) / sd).max(1)
    hard = anom[mz[anom] <= np.percentile(mz[nm], 99)]
    return X[nm], X[hard]


def ucr(name):
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
    return Xw[yw == 0], Xw[yw == 1]


def characterise(Xn, Xa):
    sc = StandardScaler().fit(Xn); Zn = sc.transform(Xn); Za = sc.transform(Xa)
    pool = np.vstack([Zn, Za]); lab = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]
    k = min(10, len(pool) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pool)
    _, ind = nn.kneighbors(Za)
    nbr = ind[:, 1:]                              # drop self
    nn_normal_frac = float(np.mean(lab[nbr] == 0))     # fraction of anomaly neighbours that are normal
    nnn = NearestNeighbors(n_neighbors=min(10, len(Zn) - 1)).fit(Zn)
    dn = nnn.kneighbors(Zn)[0][:, -1]; da = nnn.kneighbors(Za)[0][:, -1]
    local_dens_pct = float(np.median(np.searchsorted(np.sort(dn), da) / len(dn) * 100))
    # global marginal rarity: per-feature CDF percentile extremity, max over features, median over anomalies
    ext = np.zeros(len(Za))
    for j in range(Zn.shape[1]):
        p = np.searchsorted(np.sort(Zn[:, j]), Za[:, j]) / len(Zn)
        ext = np.maximum(ext, np.abs(p - 0.5) * 2)
    return nn_normal_frac, local_dens_pct, float(np.median(ext))


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include].copy()
inc["fam"] = inc.gt_best.map(fam)
rows = []
for f in ["global", "local"]:
    sub = inc[(inc.fam == f) & (inc.corpus.isin(["oddbench", "ovrbench", "ucr"]))]
    sub = sub.sort_values("spread_ap_norm", ascending=False).head(12)
    for _, r in sub.iterrows():
        try:
            Xn, Xa = ucr(r.dataset) if r.corpus == "ucr" else tab(r.corpus, r.dataset)
        except Exception:
            continue
        if len(Xn) < 80 or len(Xa) < 5:
            continue
        nf, ld, gr = characterise(Xn, Xa)
        rows.append(dict(dataset=r.dataset, true_fam=f, nn_normal_frac=nf, local_dens_pct=ld, global_rare=gr))
R = pd.DataFrame(rows)
print(f"=== anomaly geometry by true-best family ({len(R)} datasets) ===\n")
print(f"  {'':28s}{'global-win':>12s}{'local-win':>11s}")
for c, lab in [("nn_normal_frac", "anomaly nbrs that are NORMAL"), ("local_dens_pct", "local density pct (50=normal)"),
               ("global_rare", "global marginal rarity")]:
    gw = R[R.true_fam == "global"][c].median(); lw = R[R.true_fam == "local"][c].median()
    print(f"  {lab:28s}{gw:>12.2f}{lw:>11.2f}")
print("\n  per-dataset (global-win):")
print(R[R.true_fam == "global"][["dataset", "nn_normal_frac", "local_dens_pct", "global_rare"]].round(2).head(8).to_string(index=False))
