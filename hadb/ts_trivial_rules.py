# -*- coding: utf-8 -*-
"""Per-dataset triviality-RULE audit for the TIME-SERIES arms (window-feature space).

The TS triviality filter (Wu-Keogh one-liner) operates on the RAW SIGNAL per channel, but
detectors operate on WINDOW FEATURES. A series can survive the raw one-liner yet be trivially
separable by a simple rule in feature space. Analogous to hadb_trivial_rules.py (tabular): for
each included TS dataset, reconstruct window features, and compute the test AUC of the max|z|
rule and the HBOS-lite histogram rule on anomaly windows vs held-out normal windows.
"""
import os, sys, zipfile, re, io as _io, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
from hadb_ts_mts import mts_window_features, window_labels as mts_wlab, MAX_CH
NBINS = 20


def uni_windows(name, zipname):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", zipname))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]
    if fn.endswith(".csv"):
        df = pd.read_csv(_io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()]); a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    st = np.arange(0, len(x) - W + 1, STRIDE); Xw, _ = _window_features(x, w=W, stride=STRIDE); yw = _window_labels(lab, st, w=W, min_count=1)
    return np.nan_to_num(Xw), yw


def mts_windows(name):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "tsbad", "TSB-AD-M.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith(".csv") and name.split("_")[0] in n]
    df = pd.read_csv(_io.BytesIO(z.read(cand[0]))); lc = [q for q in df.columns if q.lower() in ("label", "is_anomaly", "anomaly")]
    lab = df[lc[0]].to_numpy(int); Xc = np.nan_to_num(df.drop(columns=lc).to_numpy(float))
    if len(Xc) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(Xc) - MAX_LEN)); Xc, lab = Xc[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    if Xc.shape[1] > MAX_CH:
        Xc = Xc[:, np.argsort(-Xc.std(0))[:MAX_CH]]
    Xw, starts = mts_window_features(Xc); yw = mts_wlab(lab, starts)
    return np.nan_to_num(Xw), yw


def rule_aucs(Xw, yw):
    if yw.sum() < 5 or (yw == 0).sum() < 40:
        return np.nan, np.nan
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
    anom = np.where(yw == 1)[0]
    if len(tr) < 30 or len(te) < 10 or len(anom) < 5:
        return np.nan, np.nan
    Xtr = Xw[tr]; Xe = np.vstack([Xw[te], Xw[anom]]); ye = np.r_[np.zeros(len(te)), np.ones(len(anom))]
    mt, st = Xtr.mean(0), Xtr.std(0) + 1e-9
    mz = roc_auc_score(ye, np.abs((Xe - mt) / st).max(1))
    hb = np.zeros(len(Xe))
    for j in range(Xtr.shape[1]):
        cnt, edges = np.histogram(Xtr[:, j], bins=NBINS); dens = cnt / max(cnt.sum(), 1) + 1e-6
        b = np.clip(np.digitize(Xe[:, j], edges) - 1, 0, NBINS - 1); hb += -np.log(dens[b])
    return float(mz), float(roc_auc_score(ye, hb))


M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv")); inc = M[M.include & M.modality.eq("timeseries")]
rows = []
for _, r in inc.iterrows():
    try:
        if r.corpus == "ucr": Xw, yw = uni_windows(r.dataset, "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip")
        elif r.corpus == "tsbad_u": Xw, yw = uni_windows(r.dataset, "tsbad/TSB-AD-U.zip")
        elif r.corpus == "tsbad_m": Xw, yw = mts_windows(r.dataset)
        else: continue
    except Exception:
        continue
    mz, hb = rule_aucs(Xw, yw)
    rows.append(dict(dataset=r.dataset, corpus=r.corpus, mz_rule_auc=mz, hbos_rule_auc=hb))
R = pd.DataFrame(rows).dropna()
R["worst"] = R[["mz_rule_auc", "hbos_rule_auc"]].max(1)
R.to_csv(os.path.join(S, "HADB_TS_TRIVIAL_RULES.csv"), index=False)
print(f"=== TS window-feature triviality-rule audit ({len(R)} included TS datasets) ===")
print(f"  max|z|-rule AUC>0.85: {int((R.mz_rule_auc>0.85).sum())}   HBOS-rule AUC>0.85: {int((R.hbos_rule_auc>0.85).sum())}")
print(f"  EITHER > 0.85 (would be dropped): {int((R.worst>0.85).sum())}/{len(R)}")
print(f"  by corpus:")
for c, g in R.groupby("corpus"):
    print(f"    {c:9s} n={len(g):3d}  median worst-rule AUC {g.worst.median():.3f}  >0.85: {int((g.worst>0.85).sum())}")
print("\n  worst offenders:")
print(R.sort_values("worst", ascending=False).head(8)[["dataset", "corpus", "mz_rule_auc", "hbos_rule_auc"]].round(3).to_string(index=False))
