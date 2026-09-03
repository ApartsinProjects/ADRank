# -*- coding: utf-8 -*-
"""(a) Fetch TSB-AD-M (multivariate time series) and (b) verify the UCR archive layout.

TSB-AD-M: ~180 multivariate sequences, 540 MB, Apache-2.0. Adds the one data type HADB
lacks entirely. Motivation is not just coverage: Pinet et al. 2026 (arXiv:2606.02670)
measured that across 8 public MTS benchmarks no cross-channel rupture occurs without an
accompanying UNIVARIATE deviation, so most MTS anomalies should fall to a per-channel
one-liner - correctly. What survives is the genuinely multivariate remainder.

UCR: labels live in the FILENAME, not a column
(UCR_Anomaly_<name>_<train_end>_<anom_start>_<anom_end>), so it needs a bespoke loader.
LICENSE: none present in the archive (confirmed) - fetch-only, never redistribute.
"""
import os, re, sys, time, zipfile, urllib.request
import numpy as np

ROOT = r"E:\Projects\Submitted\ADRank"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ---------------- (b) UCR layout check ----------------
ucr_zip = os.path.join(ROOT, "data", "ucr", "UCR_TimeSeriesAnomalyDatasets2021.zip")
print("=== UCR archive layout ===", flush=True)
if os.path.exists(ucr_zip):
    z = zipfile.ZipFile(ucr_zip)
    txt = [n for n in z.namelist() if n.lower().endswith(".txt")]
    inner = [n for n in z.namelist() if n.lower().endswith(".zip")]
    print(f"  .txt series files: {len(txt)}   inner zips: {len(inner)}")
    print(f"  sample names: {[os.path.basename(t) for t in txt[:3]]}")
    pat = re.compile(r"_(\d+)_(\d+)_(\d+)\.txt$", re.I)
    parsed = [t for t in txt if pat.search(t)]
    print(f"  filename-parseable (train_anomStart_anomEnd): {len(parsed)}/{len(txt)}")
    if parsed:
        s = parsed[0]
        m = pat.search(s)
        raw = z.read(s).decode("utf-8", "replace").split()
        x = np.array([float(v) for v in raw if v.strip()], float)
        tr, a0, a1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        print(f"  example {os.path.basename(s)}")
        print(f"    length={len(x)}  train_end={tr}  anomaly=[{a0},{a1}] "
              f"({a1-a0} points, {100*(a1-a0)/len(x):.3f}% of series)")
        print(f"    -> ONE anomaly region per series, as Wu & Keogh specify")
    if inner:
        print(f"  NOTE nested zip present: {inner[:2]} (may hold the same or extra data)")
else:
    print("  UCR zip not found")

# ---------------- (a) TSB-AD-M download ----------------
mts_dir = os.path.join(ROOT, "data", "tsbad")
os.makedirs(mts_dir, exist_ok=True)
dest = os.path.join(mts_dir, "TSB-AD-M.zip")
print("\n=== TSB-AD-M (multivariate) ===", flush=True)
if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
    print(f"  cached {os.path.getsize(dest)/1e6:.1f} MB")
else:
    t = time.time()
    print("  downloading 540 MB ...", flush=True)
    try:
        req = urllib.request.Request("https://www.thedatum.org/datasets/TSB-AD-M.zip", headers=UA)
        with urllib.request.urlopen(req, timeout=1800) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  saved {os.path.getsize(dest)/1e6:.1f} MB in {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(f"  DOWNLOAD FAILED: {type(e).__name__}: {e}")
        sys.exit(1)

try:
    z = zipfile.ZipFile(dest)
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    print(f"  csv files: {len(csvs)}")
    print(f"  sample: {[os.path.basename(c) for c in csvs[:3]]}")
    import pandas as pd, io as _io
    df = pd.read_csv(_io.BytesIO(z.read(csvs[0])))
    print(f"  first file shape={df.shape} cols={list(df.columns)[:8]}")
    lab = [c for c in df.columns if c.lower() in ("label", "is_anomaly", "anomaly")]
    if lab:
        print(f"  label column '{lab[0]}': rate={df[lab[0]].mean():.4f} "
              f"n_anom={int(df[lab[0]].sum())}")
        print(f"  CHANNELS = {df.shape[1]-1}  -> genuinely multivariate")
    import collections
    src = collections.Counter(os.path.basename(c).split("_")[1] if "_" in os.path.basename(c)
                              else "?" for c in csvs)
    print(f"  source datasets: {src.most_common(10)}")
except Exception as e:
    print(f"  inspect failed: {e}")
