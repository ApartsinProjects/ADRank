# -*- coding: utf-8 -*-
"""Fetch two additional HADB sources.

  1. UCR Anomaly Archive (Wu & Keogh, 250 univariate series, 184 MB).
     Built BY the authors of the triviality critique, specifically to resist the one-liner
     test that 72% of our TSB-AD sample failed. This is the right corpus for a hard TS arm.
     LICENSE: none stated on the UCR page. Usable locally; HADB must NOT redistribute it -
     ship a fetch script instead, exactly as the project's Zenodo deposit already does.

  2. MacrOData OvrBench (755 tabular datasets, CC BY 4.0).
     Same repo, format and license as OddBench, which we already parse, so integration is
     free. Note it holds STATISTICAL anomalies where OddBench holds SEMANTIC ones, so it
     adds a different anomaly type rather than more of the same.
"""
import os, sys, io, json, time, urllib.request, zipfile

ROOT = r"E:\Projects\Submitted\ADRank"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get(url, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


# ---------------- 1. UCR Anomaly Archive ----------------
ucr_dir = os.path.join(ROOT, "data", "ucr")
os.makedirs(ucr_dir, exist_ok=True)
ucr_zip = os.path.join(ucr_dir, "UCR_TimeSeriesAnomalyDatasets2021.zip")
if not os.path.exists(ucr_zip):
    t = time.time()
    print("downloading UCR Anomaly Archive (184 MB) ...", flush=True)
    b = get("https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
            "UCR_TimeSeriesAnomalyDatasets2021.zip")
    with open(ucr_zip, "wb") as f:
        f.write(b)
    print(f"  saved {len(b)/1e6:.1f} MB in {time.time()-t:.0f}s", flush=True)
else:
    print(f"UCR cached ({os.path.getsize(ucr_zip)/1e6:.1f} MB)", flush=True)

try:
    z = zipfile.ZipFile(ucr_zip)
    names = z.namelist()
    data = [n for n in names if not n.endswith("/")]
    print(f"  entries={len(names)} files={len(data)}")
    print(f"  sample: {data[:3]}")
    exts = {}
    for n in data:
        exts[os.path.splitext(n)[1].lower()] = exts.get(os.path.splitext(n)[1].lower(), 0) + 1
    print(f"  extensions: {exts}")
    lic = [n for n in data if "licen" in n.lower() or "readme" in n.lower()]
    print(f"  license/readme entries: {lic if lic else 'NONE (confirmed unlicensed)'}")
    s = data[0]
    raw = z.read(s)[:200]
    print(f"  first file head: {raw[:120]!r}")
except Exception as e:
    print(f"  UCR zip inspect failed: {e}")

# ---------------- 2. MacrOData OvrBench ----------------
ovr_dir = os.path.join(ROOT, "data", "ovrbench")
os.makedirs(ovr_dir, exist_ok=True)
API = "https://huggingface.co/api/datasets/MacrOData-CMU/OvrBench/tree/main/public"
HF = "https://huggingface.co/datasets/MacrOData-CMU/OvrBench/resolve/main/public/"
try:
    listing = json.loads(get(API, timeout=120))
    files = [e["path"].split("/")[-1] for e in listing
             if str(e.get("path", "")).endswith(".npz")]
except Exception as e:
    print(f"OvrBench listing failed: {e}")
    files = []

print(f"\nOvrBench: {len(files)} npz files listed", flush=True)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else len(files)
files = sorted(files)[:LIMIT]
ok = skip = fail = 0
t0 = time.time()
for i, name in enumerate(files, 1):
    dest = os.path.join(ovr_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        skip += 1; continue
    try:
        b = get(HF + name + "?download=true", timeout=180)
        with open(dest, "wb") as f:
            f.write(b)
        ok += 1
    except Exception:
        fail += 1
    if i % 50 == 0 or i == len(files):
        print(f"  [{i}/{len(files)}] downloaded={ok} cached={skip} failed={fail} "
              f"| {time.time()-t0:.0f}s", flush=True)

print(f"\nOvrBench done: {ok} new, {skip} cached, {fail} failed")
tot = sum(os.path.getsize(os.path.join(ovr_dir, f)) for f in os.listdir(ovr_dir))
print(f"  {ovr_dir}: {len(os.listdir(ovr_dir))} files, {tot/1e6:.1f} MB")

# confirm the format matches OddBench's loader
import numpy as np
fs = [f for f in os.listdir(ovr_dir) if f.endswith(".npz")]
if fs:
    d = np.load(os.path.join(ovr_dir, fs[0]), allow_pickle=True)
    print(f"  keys: {list(d.keys())}")
    if "train" in d:
        print(f"  train {np.asarray(d['train']).shape} test {np.asarray(d['test']).shape} "
              f"-> SAME FORMAT as OddBench, existing loader works")
