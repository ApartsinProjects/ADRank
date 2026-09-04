# -*- coding: utf-8 -*-
"""SIDE PROJECT: consolidated final streamlined pipeline + anomaly-DIVERSITY sensor.
Filters: Stage1 OR-solvable(<0.9) | Stage2 harden anomalies(5%-FP) | Stage3 >=100 hard |
         Stage4 dedup(fingerprint) | min-normals >=800 (test-side). No base-rate cap.
Diversity sensor on the HARD anomaly set (are they distinct or near-duplicates?):
  n_eff  = greedy radius-cover count at r = median normal nearest-neighbor distance
           (effectively-distinct anomalies at the normal resolution scale)
  eff_frac = n_eff / n_hard   (1 = all distinct; <<1 = near-duplicates, e.g. overlapping TS windows)
Writes to streamline/ only."""
import os, sys, io, zipfile, re, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as P
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
ZIPS = {"ucr": "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip", "tsbad_u": "tsbad/TSB-AD-U.zip"}
Q = 0.05; TRIV1 = 0.90; MIN_HARD = 100; MIN_NORM = 800; MAX_MTS = 200
_ZIP = {}
def zf(p):
    if p not in _ZIP: _ZIP[p] = zipfile.ZipFile(os.path.join(ROOT, "data", p))
    return _ZIP[p]
def split_na(X, y):
    X = np.nan_to_num(np.asarray(X, float)); y = np.asarray(y, int).ravel(); r = np.random.default_rng(0)
    ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 6000: ni = r.choice(ni, 6000, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.6 * len(ni)):]], X[ai]
def tab_load(c, n):
    d = np.load(os.path.join(ROOT, "data", c, n + ".npz"), allow_pickle=True)
    return split_na(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]),
                    np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]))
def uni_load(name, zipname):
    z = zf(zipname); cand = [q for q in z.namelist() if os.path.basename(q) == name] or [q for q in z.namelist() if q.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(q)]
    fn = cand[0]
    if fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()]); a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    Xw, _ = _window_features(x, w=W, stride=STRIDE); st = np.arange(0, len(x) - W + 1, STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
    return Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[(yw == 1)]
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b] + 1e-9)))
    return sev
def fingerprint(Xtr, Xn):
    Z = np.vstack([Xtr, Xn]); return "%d|%s|%s" % (Z.shape[1], hashlib.sha1(np.round(np.sort(Z.mean(0)), 3).tobytes()).hexdigest()[:10], hashlib.sha1(np.round(np.sort(Z.std(0)), 3).tobytes()).hexdigest()[:10])
def diversity(Xtr, Xh):
    """n_eff = greedy radius cover of hard anomalies at r = median normal NN distance (std space)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd; Zh = (Xh - mu) / sd
    k = min(5, len(Zt) - 1); nn = NearestNeighbors(n_neighbors=2).fit(Zt[np.random.default_rng(0).choice(len(Zt), min(len(Zt), 1500), replace=False)])
    r = np.median(nn.kneighbors(Zt[:min(len(Zt), 1500)])[0][:, 1]) + 1e-9
    if len(Zh) > 600: Zh = Zh[np.random.default_rng(0).choice(len(Zh), 600, replace=False)]
    covered = np.zeros(len(Zh), bool); order = np.arange(len(Zh)); neff = 0
    nnh = NearestNeighbors(radius=r).fit(Zh)
    for i in order:
        if covered[i]: continue
        neff += 1; idx = nnh.radius_neighbors(Zh[i:i + 1], return_distance=False)[0]; covered[idx] = True
    return neff, r
def sources():
    for corp, f in CSVMAP.items():
        for name in sorted(pd.read_csv(os.path.join(S, f)).dataset.unique()): yield corp, name, "tab" if corp in ("oddbench", "ovrbench") else "uni"
    for sub in ("adbench", "dami"):
        dd = os.path.join(ROOT, "data", sub)
        if os.path.isdir(dd):
            for ds in P.load_npz_dir(dd): yield sub, ds.name, ("obj", ds)
    for name, src, Xc, lab in load_mts(MAX_MTS): yield "tsbad_m", name, ("mts", Xc, lab)
rows = []
for corp, name, how in sources():
    try:
        if how == "tab": Xtr, Xn, Xa = tab_load(corp, name)
        elif how == "uni": Xtr, Xn, Xa = uni_load(name, ZIPS[corp])
        elif how[0] == "obj": Xtr, Xn, Xa = split_na(how[1].X, how[1].y)
        else:
            Xc, lab = how[1], how[2]
            if len(Xc) < W + 10: continue
            Xw, starts = mts_window_features(Xc); yw = mts_wlabels(lab, starts); Xw = np.nan_to_num(Xw)
            pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0); Xtr, Xn, Xa = Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[(yw == 1)]
    except Exception: continue
    if len(Xtr) < 40 or len(Xn) < 20 or len(Xa) < 3: continue
    sev_n = severity(Xtr, Xn); sev_a = severity(Xtr, Xa); thr = np.quantile(sev_n, 1 - Q)
    hard_mask = sev_a <= thr; Xh = Xa[hard_mask]; nhard = len(Xh)
    rec = {"corpus": corp, "dataset": str(name)[:40], "n_norm": len(Xn), "n_hard": nhard,
           "frac_triv": float((~hard_mask).mean()), "fp": fingerprint(Xtr, Xn)}
    if nhard >= 5:
        neff, r = diversity(Xtr, Xh); rec["n_eff"] = neff; rec["eff_frac"] = neff / nhard
    else:
        rec["n_eff"] = nhard; rec["eff_frac"] = 1.0
    rows.append(rec)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "STREAM_FINAL_ALL.csv"), index=False)
f = df[(df.frac_triv < TRIV1) & (df.n_norm >= MIN_NORM) & (df.n_hard >= MIN_HARD)].sort_values("n_hard", ascending=False).drop_duplicates("fp")
f.to_csv(os.path.join(OUT, "STREAM_FINAL_SET.csv"), index=False)
print(f"=== FINAL streamlined benchmark (min_norm>={MIN_NORM}, >={MIN_HARD} hard, no base-rate cap) ===")
print(f"  {len(f)} datasets   ", f.corpus.value_counts().to_dict())
print(f"\n  ANOMALY DIVERSITY sensor (n_eff = distinct anomalies at normal resolution; eff_frac = n_eff/n_hard):")
print(f"  {'corpus':10s} {'n':>4s} {'med n_eff':>9s} {'med eff_frac':>12s} {'med n_hard':>10s}")
for c, g in f.groupby("corpus"):
    print(f"  {c:10s} {len(g):4d} {int(g.n_eff.median()):9d} {g.eff_frac.median():12.2f} {int(g.n_hard.median()):10d}")
print(f"  OVERALL   med n_eff {int(f.n_eff.median())}  med eff_frac {f.eff_frac.median():.2f}")
print(f"\n  datasets whose hard anomalies are mostly near-duplicates (eff_frac<0.5): {int((f.eff_frac<0.5).sum())}/{len(f)}")
print(f"  datasets with >=50 DISTINCT hard anomalies (n_eff>=50): {int((f.n_eff>=50).sum())}/{len(f)}   "
      f"by corpus: {f[f.n_eff>=50].corpus.value_counts().to_dict()}")
print("saved streamline/STREAM_FINAL_ALL.csv, STREAM_FINAL_SET.csv")
