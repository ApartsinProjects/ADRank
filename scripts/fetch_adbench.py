"""Download a curated subset of ADBench Classical .npz datasets.

Source: https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical
Files are public raw .npz with keys 'X' (n,d) and 'y' (n,) with y=1 for anomalies.

Runtime: ~1-2 minutes on a normal connection. All files are small (<10 MB each).
"""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "adbench")

BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"

# Curated: mid-size tabular sets that fit our 200 <= n <= 50k filter and are
# widely used in unsupervised AD literature. Skips images/text.
DATASETS = [
    "6_cardio.npz",
    "8_celeba.npz",
    "10_cover.npz",
    "11_donors.npz",
    "13_fraud.npz",
    "14_glass.npz",
    "16_http.npz",
    "18_Ionosphere.npz",
    "19_landsat.npz",
    "20_letter.npz",
    "21_Lymphography.npz",
    "22_magic.gamma.npz",
    "23_mammography.npz",
    "25_musk.npz",
    "26_optdigits.npz",
    "27_PageBlocks.npz",
    "28_pendigits.npz",
    "29_Pima.npz",
    "30_satellite.npz",
    "31_satimage-2.npz",
    "32_shuttle.npz",
    "33_skin.npz",
    "35_SpamBase.npz",
    "36_speech.npz",
    "37_Stamps.npz",
    "38_thyroid.npz",
    "39_vertebral.npz",
    "40_vowels.npz",
    "41_Waveform.npz",
    "42_WBC.npz",
    "43_WDBC.npz",
    "44_Wilt.npz",
    "45_wine.npz",
    "46_WPBC.npz",
    "47_yeast.npz",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for fn in DATASETS:
        out = os.path.join(OUT_DIR, fn)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            n_skip += 1
            continue
        url = BASE + fn
        try:
            print(f"  fetching {fn} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, out)
            size = os.path.getsize(out)
            print(f"ok ({size/1024:.1f} KB)")
            n_ok += 1
        except Exception as e:
            print(f"FAIL ({e})")
            if os.path.exists(out):
                os.remove(out)
            n_fail += 1
    print(f"\n[fetch] downloaded={n_ok}, cached={n_skip}, failed={n_fail}")


if __name__ == "__main__":
    main()
