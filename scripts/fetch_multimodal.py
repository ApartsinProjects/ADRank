"""Download ADBench CV_by_ResNet18 and NLP_by_BERT subsets.

CV: ResNet18 pretrained embeddings of image datasets (each numeric class as one file).
NLP: BERT pretrained embeddings of text datasets.

Same .npz {X, y} interface as Classical. Drops directly into the existing pipeline.
"""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_DIR = os.path.join(ROOT, "data", "cv")
NLP_DIR = os.path.join(ROOT, "data", "nlp")

BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/"

# Curated: all CIFAR10 and FashionMNIST classes (20 image datasets) + all NLP.
CV_FILES = (
    [f"CIFAR10_{i}.npz" for i in range(10)]
    + [f"FashionMNIST_{i}.npz" for i in range(10)]
)
NLP_FILES = [
    "20news_0.npz", "20news_1.npz", "20news_2.npz",
    "20news_3.npz", "20news_4.npz", "20news_5.npz",
    "agnews_0.npz", "agnews_1.npz", "agnews_2.npz", "agnews_3.npz",
    "amazon.npz", "imdb.npz", "yelp.npz",
]


def _fetch(files, sub, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for fn in files:
        out = os.path.join(out_dir, fn)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            n_skip += 1
            continue
        url = BASE + sub + "/" + fn
        try:
            print(f"  fetching {sub}/{fn} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, out)
            print(f"ok ({os.path.getsize(out)/1024:.1f} KB)")
            n_ok += 1
        except Exception as e:
            print(f"FAIL ({e})")
            if os.path.exists(out):
                os.remove(out)
            n_fail += 1
    return n_ok, n_skip, n_fail


def main():
    o, s, f = _fetch(CV_FILES, "CV_by_ResNet18", CV_DIR)
    print(f"[cv] downloaded={o} cached={s} failed={f}")
    o, s, f = _fetch(NLP_FILES, "NLP_by_BERT", NLP_DIR)
    print(f"[nlp] downloaded={o} cached={s} failed={f}")


if __name__ == "__main__":
    main()
