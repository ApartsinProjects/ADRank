"""Download and parse the DAMI outlier benchmark (Campos et al. 2016) into npz.

Second benchmark family, distinct from ADBench. Per-dataset tarballs from LMU
Munich, each containing ARFF files with an `id` attribute, numeric features, and
a nominal `outlier` attribute ('yes'/'no'). We take the normalized, duplicate-free
base variant per dataset (the *_withoutdupl_norm_*.arff with the lowest outlier %
and no _vNN replicate suffix) so each dataset appears once.

Output: data/dami/<Dataset>.npz with keys X (n,d) float and y (n,) int (1=anomaly).
"""
from __future__ import annotations
import io, os, re, tarfile, urllib.request

import numpy as np
from scipy.io import arff

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "dami")
BASE = "https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/"

# Curated base datasets (UCI-derived), spanning sizes/dims; distinct from ADBench splits.
DATASETS = [
    "Annthyroid", "Arrhythmia", "Cardiotocography", "HeartDisease", "Hepatitis",
    "PageBlocks", "Parkinson", "Pima", "SpamBase", "Stamps",
    "Waveform", "WBC", "WDBC", "WPBC", "Wilt", "InternetAds",
]


def _pick_arff(names):
    """Prefer *_withoutdupl_norm_*.arff, lowest outlier %, no _vNN replicate."""
    cand = [n for n in names if n.endswith(".arff")]
    def score(n):
        base = os.path.basename(n)
        norm = "_norm" in base
        nodup = "withoutdupl" in base
        no_rep = re.search(r"_v\d+\.arff$", base) is None
        m = re.search(r"_(\d+)\.arff$", base.replace(".arff", "_x.arff"))  # dummy
        pctm = re.findall(r"_(\d{2})(?:_v\d+)?\.arff$", base)
        pct = int(pctm[0]) if pctm else 99
        return (nodup, norm, no_rep, -pct)  # maximize
    cand.sort(key=score, reverse=True)
    return cand[0] if cand else None


def _parse_arff_bytes(b):
    data, meta = arff.loadarff(io.StringIO(b.decode("utf-8", errors="replace")))
    names = list(meta.names())
    # label column
    label_col = next((c for c in names if c.lower() == "outlier"), None)
    id_col = next((c for c in names if c.lower() == "id"), None)
    feat = [c for c in names if c not in (label_col, id_col)]
    X = np.column_stack([np.asarray(data[c], dtype=np.float64) for c in feat])
    yv = data[label_col]
    y = np.array([1 if (str(v).lower().strip("b'\"") == "yes") else 0 for v in yv], dtype=int)
    return X, y


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = fail = 0
    for ds in DATASETS:
        out = os.path.join(OUT, f"{ds}.npz")
        if os.path.exists(out):
            ok += 1; continue
        url = BASE + ds + ".tar.gz"
        try:
            print(f"  {ds} ...", end=" ", flush=True)
            raw = urllib.request.urlopen(url, timeout=60).read()
            tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
            arff_name = _pick_arff(tf.getnames())
            if not arff_name:
                print("no arff"); fail += 1; continue
            b = tf.extractfile(arff_name).read()
            X, y = _parse_arff_bytes(b)
            np.savez_compressed(out, X=X, y=y)
            print(f"ok  {arff_name.split('/')[-1]}  X={X.shape} anom={int(y.sum())} ({100*y.mean():.1f}%)")
            ok += 1
        except Exception as e:
            print(f"FAIL {type(e).__name__}: {e}")
            fail += 1
    print(f"\n[dami] ok={ok} fail={fail} -> {OUT}")


if __name__ == "__main__":
    main()
