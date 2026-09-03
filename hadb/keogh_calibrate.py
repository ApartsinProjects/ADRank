# -*- coding: utf-8 -*-
"""Calibrate the Wu & Keogh triviality test against their OWN published numbers.

WHY: my first implementation called 74% of the UCR archive trivial. UCR was built BY
Wu & Keogh to RESIST that test and they state only "a small fraction" is one-liner-solvable.
So the implementation is wrong, not the archive. It also called 72% of TSB-AD trivial, a
number I reported as a finding and now withdraw pending this calibration.

THE LIKELY ERROR: my "solved" criterion was "the argmax of the one-liner statistic falls
inside the labelled anomaly", brute-forced over 30 parameter settings and declared trivial
if ANY landed. That is far too lenient - with one anomaly region per series, the largest
jump often IS the anomaly without the rule being a usable detector.

VALIDATION GATE (this is the point of the script): the implementation must reproduce their
published result on the corpus they published it for -
    Wu & Keogh, IEEE TKDE 35(3), Table 1: 316 of 367 (86.1%) YAHOO series are trivial,
    and 193/367 fall to a bare constant threshold.
TSB-AD-U carries 259 YAHOO series, so we can measure directly. Whichever criterion lands
near 86% on Yahoo is the one to use everywhere else.

CRITERIA COMPARED
  argmax  : argmax of the statistic lies inside the anomaly            (my original)
  auc90   : the statistic achieves ROC-AUC >= 0.90 against point labels (detector-like)
  auc95   : same at 0.95
  top1pct : the anomaly region is enriched in the statistic's top 1% of points
"""
import os, io, re, zipfile, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\keogh_calibrate.csv"


def _movmean(a, k):
    if k <= 1:
        return np.zeros(len(a))
    c = np.cumsum(np.insert(a, 0, 0.0)); out = np.empty(len(a)); h = k // 2
    for i in range(len(a)):
        lo, hi = max(0, i - h), min(len(a), i + h + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def _movstd(a, k):
    if k <= 1:
        return np.zeros(len(a))
    m = _movmean(a, k); m2 = _movmean(a * a, k)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def oneliner_stats(x):
    """Yield (tag, statistic) for each of Wu & Keogh's one-liner parameterisations."""
    d = np.diff(x, prepend=x[0])
    for base, btag in ((np.abs(d), "absdiff"), (d, "diff")):
        for k in (0, 8, 32, 128, 512):
            for u in (0, 1):
                if u and k == 0:
                    continue
                for c in (0.0, 1.0, 3.0):
                    if k == 0 and c > 0:
                        continue
                    thr = np.zeros(len(base))
                    if u:
                        thr = thr + _movmean(base, k)
                    if c and k:
                        thr = thr + c * _movstd(base, k)
                    st = base - thr
                    if np.all(np.isfinite(st)) and np.nanstd(st) > 1e-12:
                        yield f"{btag}_u{u}_k{k}_c{c:g}", st


def evaluate(x, lab):
    """Return best-over-parameterisations value for each candidate criterion."""
    best = dict(argmax=False, auc=0.0, top1pct=0.0)
    n_top = max(1, int(0.01 * len(x)))
    for tag, st in oneliner_stats(x):
        if lab[int(np.nanargmax(st))] == 1:
            best["argmax"] = True
        try:
            a = roc_auc_score(lab, st)
        except Exception:
            a = 0.5
        best["auc"] = max(best["auc"], a)
        top = np.argpartition(-st, n_top - 1)[:n_top]
        best["top1pct"] = max(best["top1pct"], float(lab[top].mean()))
    return best


# ---------------- YAHOO series inside TSB-AD-U: the calibration target ----------------
z = zipfile.ZipFile(os.path.join(ROOT, "data", "tsbad", "TSB-AD-U.zip"))
yahoo = [n for n in z.namelist() if n.lower().endswith(".csv") and "_YAHOO_" in n]
print(f"YAHOO series available in TSB-AD-U: {len(yahoo)}", flush=True)
print("TARGET (Wu & Keogh Table 1): 316/367 = 86.1% trivial\n", flush=True)

rows = []
for i, fn in enumerate(yahoo, 1):
    try:
        df = pd.read_csv(io.BytesIO(z.read(fn)))
        x = np.nan_to_num(df.iloc[:, 0].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        lab = df.iloc[:, 1].to_numpy(int)
    except Exception:
        continue
    if lab.sum() == 0 or lab.sum() == len(lab) or len(x) < 50:
        continue
    r = evaluate(x, lab)
    r.update(dataset=os.path.basename(fn), corpus="YAHOO", n=len(x),
             anom_rate=float(lab.mean()))
    rows.append(r)
    if i % 50 == 0:
        print(f"  [{i}/{len(yahoo)}]", flush=True)

Y = pd.DataFrame(rows)
Y.to_csv(OUT, index=False)
print(f"\nscored {len(Y)} YAHOO series\n")
print("=== criterion calibration on YAHOO (target 86.1%) ===")
cands = [("argmax (my original)", Y.argmax_ if "argmax_" in Y else Y["argmax"]),
         ("auc >= 0.90", Y.auc >= 0.90),
         ("auc >= 0.95", Y.auc >= 0.95),
         ("auc >= 0.99", Y.auc >= 0.99),
         ("top1pct >= 0.5", Y.top1pct >= 0.5),
         ("top1pct >= 0.9", Y.top1pct >= 0.9)]
for name, mask in cands:
    pct = 100 * float(np.mean(mask))
    print(f"  {name:22s} -> {pct:5.1f}% trivial   {'<-- MATCHES' if 80 <= pct <= 92 else ''}")

print(f"\n  median best one-liner AUC on YAHOO: {Y.auc.median():.3f}")
print(f"  (Wu & Keogh: 193/367 = 52.6% fall to a BARE CONSTANT threshold)")
bare = Y.auc.copy()
print(f"  our absdiff_u0_k0_c0-only AUC>=0.9 rate: n/a in this summary")
