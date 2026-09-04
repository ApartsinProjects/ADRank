# -*- coding: utf-8 -*-
"""SIDE PROJECT: apply the 3-stage streamlined pipeline FROM SCRATCH on the SOURCE collection
(all scored candidates in the arm CSVs, pre-include), detector-free.
  Stage 1  drop DATASET if an OR-of-per-feature-thresholds rule catches almost all anomalies (marginal-trivial).
  Stage 2  drop trivial ANOMALIES (calibrated per-feature rarity above the normals' (1-q) quantile = q FP).
  Stage 3  drop DATASET if too few HARD anomalies survive.
Reports the selection funnel per corpus. Writes to streamline/ only; never touches canonical files."""
import os, sys, io, zipfile, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
Q = 0.05; TRIV1 = 0.90; MIN_HARD = 10; MAXBR = 0.25
_ZIP = {}
def zf(p):
    if p not in _ZIP: _ZIP[p] = zipfile.ZipFile(os.path.join(ROOT, "data", p))
    return _ZIP[p]


def tab_load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    r = np.random.default_rng(0); ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 6000: ni = r.choice(ni, 6000, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.6 * len(ni)):]], X[ai]


def uni_load(name, zipname):
    z = zf(zipname)
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(n)]
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
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); hist = -np.log(dens[b] + 1e-9)
        sev = np.maximum(sev, np.maximum(tail, hist))
    return sev


ZIPS = {"ucr": "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip", "tsbad_u": "tsbad/TSB-AD-U.zip"}
rows = []
for corp, f in CSVMAP.items():
    D = pd.read_csv(os.path.join(S, f)); names = sorted(D.dataset.unique())
    for i, name in enumerate(names):
        try:
            if corp in ("oddbench", "ovrbench"): Xtr, Xn, Xa = tab_load(corp, name)
            else: Xtr, Xn, Xa = uni_load(name, ZIPS[corp])
        except Exception: continue
        if len(Xtr) < 40 or len(Xn) < 20 or len(Xa) < 3:
            rows.append({"corpus": corp, "dataset": name, "stage": "load_fail", "n_hard": 0}); continue
        sev_n = severity(Xtr, Xn); sev_a = severity(Xtr, Xa)
        thr = np.quantile(sev_n, 1 - Q); triv = sev_a > thr
        n_hard = int((~triv).sum()); frac_triv = float(triv.mean())
        br_hard = n_hard / (n_hard + len(Xn) + 1e-9)
        rows.append({"corpus": corp, "dataset": name, "n_anom": len(Xa), "n_norm": len(Xn),
                     "frac_triv": frac_triv, "n_hard": n_hard, "br_hard": br_hard})
    print(f"  {corp}: processed {len(names)}", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "STREAM_SOURCE.csv"), index=False)
ok = df[df.n_hard.notna() & df.get("frac_triv").notna()].copy() if "frac_triv" in df else df
loaded = df[df.frac_triv.notna()]
s1 = loaded[loaded.frac_triv < TRIV1]                       # stage 1: not OR-solvable
s3 = s1[s1.n_hard >= MIN_HARD]                              # stage 3: enough hard anomalies
print(f"\n=== streamlined pipeline on SOURCE ({len(df)} candidates) ===")
print(f"  loaded ok (>=40 train, >=20 norm, >=3 anom): {len(loaded)}")
print(f"  STAGE 1 keep (frac trivial < {TRIV1}):        {len(s1)}   (dropped {len(loaded)-len(s1)} OR-solvable)")
print(f"  STAGE 3 keep (>= {MIN_HARD} hard anomalies):        {len(s3)}   (dropped {len(s1)-len(s3)} too-few-hard)")
print(f"\n  per-corpus funnel (source -> loaded -> stage1 -> final):")
for c in CSVMAP:
    a = (df.corpus == c).sum(); b = (loaded.corpus == c).sum(); c1 = (s1.corpus == c).sum(); c3 = (s3.corpus == c).sum()
    print(f"    {c:10s} {a:4d} -> {b:4d} -> {c1:4d} -> {c3:4d}")
print(f"\n  FINAL streamlined benchmark: {len(s3)} datasets   (vs current include=199)")
print(f"  median hardened base rate: {s3.br_hard.median():.3f}  (cap {MAXBR})")
s3.to_csv(os.path.join(OUT, "STREAM_SOURCE_FINAL.csv"), index=False)
print("saved streamline/STREAM_SOURCE.csv, STREAM_SOURCE_FINAL.csv")
