# -*- coding: utf-8 -*-
"""SIDE PROJECT (isolated - does NOT touch canonical selection/CSVs/results).
Two-stage triviality pipeline:
  Stage 1  drop DATASETS an OR-of-per-feature-thresholds rule already solves (calibrated OR ap_norm high).
  Stage 2  in survivors, drop ANOMALIES that are individually trivial by the SAME rule (a feature value
           out of the train [min,max] range, or in an empty histogram bin). Keep only in-support anomalies
           that no single-feature marginal rule catches.
This pass reports SELECTION + COMPOSITION only (no detector re-fit). Writes to streamline/ only.
"""
import os, sys, io, zipfile, re, warnings
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
TAU1 = 0.50   # stage-1: drop dataset if OR-rule ap_norm >= TAU1 (report a few)


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


def apn(y, s):
    b = y.mean(); return (average_precision_score(y, s) - b) / (1 - b + 1e-12)


def rule(Xtr, X, bins=30):
    """per-point: (#out-of-support features, oob_score for ranking). Out-of-support = out of train
    [min,max] OR in an empty train histogram bin."""
    n, d = X.shape; oob = np.zeros(n, int); sev = np.zeros(n)   # sev = max per-feature -log density (for ranking)
    for j in range(d):
        tr = Xtr[:, j]
        if np.std(tr) < 1e-12: continue
        lo, hi = tr.min(), tr.max()
        cnt, edges = np.histogram(tr, bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1)
        out = (X[:, j] < lo) | (X[:, j] > hi) | (cnt[b] == 0)
        oob += out.astype(int)
        rar = np.where(out, 20.0, -np.log(dens[b] + 1e-9)); sev = np.maximum(sev, rar)
    return oob, sev


meta = {}
for c, f in CSVMAP.items():
    d = pd.read_csv(os.path.join(S, f)); br = d.groupby("dataset").base_rate.mean()
    for ds in br.index: meta[ds] = br[ds]
mp = pd.read_csv(os.path.join(S, "MODALITY_PERF.csv")); mp["truefam"] = np.where(mp.best_local > mp.best_global, "local", "global")
rows = []
for _, r in mp.iterrows():
    if r.dataset not in meta: continue
    br = meta[r.dataset]
    try:
        if r.corpus in ("oddbench", "ovrbench"): Xtr, Xn, Xa = tab_load(r.corpus, r.dataset)
        elif r.corpus == "ucr": Xtr, Xn, Xa = uni_load(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        else: Xtr, Xn, Xa = uni_load(r.dataset, "tsbad/TSB-AD-U.zip")
    except Exception: continue
    if len(Xtr) < 40 or len(Xn) < 20 or len(Xa) < 5 or br != br: continue
    oob_a, sev_a = rule(Xtr, Xa); oob_n, sev_n = rule(Xtr, Xn)
    # base-rate matched OR ap_norm (stage-1 score) using severity ranking
    rng = np.random.default_rng(1); na = max(3, int(round(br / (1 - br) * len(Xn))))
    ia = rng.choice(len(Xa), na, replace=False) if len(Xa) > na else np.arange(len(Xa))
    y = np.r_[np.zeros(len(Xn)), np.ones(len(ia))]; sc = np.r_[sev_n, sev_a[ia]]
    or_ap = apn(y, sc)
    triv_a = oob_a > 0                     # anomaly trivial if ANY feature out-of-support
    rows.append({"dataset": r.dataset, "corpus": r.corpus, "truefam": r.truefam, "d": Xtr.shape[1],
                 "n_anom": len(Xa), "or_ap": or_ap,
                 "frac_anom_trivial": float(triv_a.mean()), "frac_norm_trivial": float((oob_n > 0).mean()),
                 "n_hard_anom": int((~triv_a).sum())})
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "STREAM_STAGE12.csv"), index=False)
print(f"=== two-stage triviality pipeline (SIDE PROJECT, {len(df)} datasets: "
      f"{int((df.truefam=='local').sum())} local / {int((df.truefam=='global').sum())} global) ===")
print(f"\nSTAGE 1  drop datasets an OR rule solves (or_ap >= tau1):")
for t in [0.4, 0.5, 0.6]:
    drop = df.or_ap >= t; sl = df[~drop]
    print(f"   tau1={t}:  drop {int(drop.sum()):3d}  -> survive {len(sl):3d}  ({int((sl.truefam=='local').sum())} local / {int((sl.truefam=='global').sum())} global)")
surv = df[df.or_ap < TAU1].copy()
print(f"\nSTAGE 2  (on {len(surv)} survivors at tau1={TAU1}) drop trivial anomalies (>=1 feature out of train support):")
print(f"   frac anomalies trivial:  local {surv[surv.truefam=='local'].frac_anom_trivial.mean():.2f}  global {surv[surv.truefam=='global'].frac_anom_trivial.mean():.2f}")
print(f"   false-positive (normals flagged trivial): {surv.frac_norm_trivial.mean():.3f}")
surv["hard_ok"] = surv.n_hard_anom >= 5
final = surv[surv.hard_ok]
print(f"   datasets with >=5 HARD anomalies after stage 2:  {len(final)}/{len(surv)}  "
      f"({int((final.truefam=='local').sum())} local / {int((final.truefam=='global').sum())} global)")
print(f"   total anomalies: {int(surv.n_anom.sum())} -> hard {int(surv.n_hard_anom.sum())} "
      f"({100*surv.n_hard_anom.sum()/max(surv.n_anom.sum(),1):.0f}% kept)")
print(f"\nFINAL streamlined benchmark: {len(final)} datasets "
      f"({int((final.truefam=='local').sum())} local / {int((final.truefam=='global').sum())} global)  vs original 162 (106/56)")
print("saved streamline/STREAM_STAGE12.csv")
