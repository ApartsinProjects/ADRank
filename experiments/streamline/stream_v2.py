# -*- coding: utf-8 -*-
"""SIDE PROJECT v2 (isolated). Corrected two-stage triviality pipeline.
  Stage 1 (dataset): drop where a calibrated OR rule captures ~all achievable ap_norm (mv_gain < eps,
          from CALIBRATED_OR.csv) - i.e. no multivariate modeling helps.
  Stage 2 (anomaly): calibrated per-feature rarity (ECDF-tail / hist), OR'd on the most-rare feature;
          threshold = normals' (1-q) quantile so the rule has exactly q FP; drop anomalies above it
          (trivially catchable), keep the rest (marginally normal-looking -> need multivariate detection).
Reports SELECTION + COMPOSITION only. Writes to streamline/ only.
"""
import os, sys, io, zipfile, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
EPS1 = 0.05   # stage-1 mv_gain threshold
Q = 0.05      # stage-2 normal false-positive rate


def tab_load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int); X = np.nan_to_num(X)
    r = np.random.default_rng(0); ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 6000: ni = r.choice(ni, 6000, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.6 * len(ni)):]], X[ai]


def uni_load(name, zipname):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", zipname))
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
    """per-point max over features of calibrated rarity (max of ECDF-tail and hist -log density)."""
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


co = pd.read_csv(os.path.join(S, "CALIBRATED_OR.csv"))[["dataset", "truefam", "mv_gain"]]
mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv"))[["corpus", "dataset"]]
co = co.merge(mp, on="dataset", how="left")
surv1 = co[co.mv_gain >= EPS1]                       # STAGE 1
print(f"=== streamlined pipeline v2 (SIDE PROJECT) ===")
print(f"STAGE 1 (mv_gain>={EPS1}): {len(co)} -> {len(surv1)} datasets "
      f"({int((surv1.truefam=='local').sum())} local / {int((surv1.truefam=='global').sum())} global)")
rows = []
for _, r in surv1.iterrows():
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xn, Xa = tab_load(r.corpus, r.dataset)
        elif r.corpus == "ucr": Xtr, Xn, Xa = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        else: Xtr, Xn, Xa = uni_load(r.dataset, "tsbad/TSB-AD-U.zip")
    except Exception: continue
    if len(Xtr) < 40 or len(Xn) < 20 or len(Xa) < 5: continue
    sev_n = severity(Xtr, Xn); sev_a = severity(Xtr, Xa)
    thr = np.quantile(sev_n, 1 - Q)                  # STAGE 2 threshold at q FP on normals
    triv = sev_a > thr                               # anomaly trivially caught by the q-FP OR rule
    rows.append({"dataset": r.dataset, "truefam": r.truefam, "n_anom": len(Xa),
                 "fp_norm": float((sev_n > thr).mean()), "frac_anom_trivial": float(triv.mean()),
                 "n_hard": int((~triv).sum())})
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "STREAM_V2.csv"), index=False)
print(f"\nSTAGE 2 (drop anomalies a {int(Q*100)}%-FP calibrated OR catches):")
print(f"   frac anomalies trivial (removed):  local {df[df.truefam=='local'].frac_anom_trivial.mean():.2f}  "
      f"global {df[df.truefam=='global'].frac_anom_trivial.mean():.2f}")
print(f"   realized normal FP rate: {df.fp_norm.mean():.3f} (target {Q})")
print(f"   anomalies: {int(df.n_anom.sum())} -> hard {int(df.n_hard.sum())} ({100*df.n_hard.sum()/max(df.n_anom.sum(),1):.0f}% kept)")
print(f"\nSTAGE 3 (drop datasets without enough surviving HARD anomalies):")
for mh in [5, 10, 20]:
    final = df[df.n_hard >= mh]
    print(f"   min_hard={mh:2d}:  keep {len(final):3d}/{len(df)} datasets "
          f"({int((final.truefam=='local').sum())} local / {int((final.truefam=='global').sum())} global)")
final = df[df.n_hard >= 10]
final.to_csv(os.path.join(OUT, "STREAM_FINAL.csv"), index=False)
print(f"\nFINAL streamlined benchmark (min_hard=10): {len(final)} datasets "
      f"({int((final.truefam=='local').sum())} local / {int((final.truefam=='global').sum())} global)  vs original 162 (106/56)")
print("saved streamline/STREAM_V2.csv, STREAM_FINAL.csv")
