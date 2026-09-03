# -*- coding: utf-8 -*-
"""Download and inspect TSB-AD-U (Liu & Paparrizos, NeurIPS 2024 D&B; Apache-2.0).

TSB-AD is re-curated specifically to REMOVE flawed/trivial exemplars, so it is an
externally-prepared HARD benchmark - exactly what we want to test NoMaS against without
building our own.
"""
import os, io, zipfile, urllib.request, time
import numpy as np

ROOT = r"E:\Projects\Submitted\ADRank"
DEST = os.path.join(ROOT, "data", "tsbad")
os.makedirs(DEST, exist_ok=True)
URL = "https://www.thedatum.org/datasets/TSB-AD-U.zip"
ZIP = os.path.join(DEST, "TSB-AD-U.zip")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

if not os.path.exists(ZIP):
    t = time.time()
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(ZIP, "wb") as f:
        f.write(r.read())
    print(f"downloaded {os.path.getsize(ZIP)/1e6:.1f} MB in {time.time()-t:.0f}s", flush=True)
else:
    print(f"cached {os.path.getsize(ZIP)/1e6:.1f} MB", flush=True)

z = zipfile.ZipFile(ZIP)
names = z.namelist()
print(f"entries: {len(names)}")
print("first 8:", names[:8])
exts = {}
for n in names:
    e = os.path.splitext(n)[1].lower()
    exts[e] = exts.get(e, 0) + 1
print("extensions:", exts)

# inspect one data file
data_files = [n for n in names if n.lower().endswith((".csv", ".npy", ".txt", ".out"))]
print(f"data files: {len(data_files)}")
if data_files:
    s = data_files[0]
    raw = z.read(s)
    print(f"\n--- {s} ({len(raw)/1e3:.1f} KB) first 300 bytes ---")
    print(raw[:300].decode("utf-8", errors="replace"))
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(raw))
        print("parsed shape:", df.shape, "cols:", list(df.columns)[:8])
        if "Label" in df.columns or "label" in df.columns:
            lc = "Label" if "Label" in df.columns else "label"
            print(f"anomaly rate: {df[lc].mean():.4f}  n_anom={int(df[lc].sum())}")
    except Exception as e:
        print("csv parse failed:", e)
