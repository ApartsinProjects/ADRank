"""Is the VaDE negative real, or a training/param artifact?

Before accepting "VaDE does not help", check that VaDE actually produces GOOD
clusters. Diagnose, on the ceiling datasets, VaDE vs PCA+KMeans:
  - effective K (non-empty clusters); a collapse to few clusters would be a bug
  - silhouette in the clustering's own latent and in original space
  - ARI between VaDE and KMeans labels (are they even similar?)
Then sanity-check the prime suspect: the reconstruction term (summed MSE over d
features) swamping the GMM clustering terms, so VaDE degrades to a plain AE with a
weak mixture. We re-fit VaDE at several alpha (recon weight) and epoch settings and
report both cluster silhouette and, if clusters look better, whether regret moves.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adrank.pipeline import load_npz_dir, embed, cluster
from sklearn.metrics import silhouette_score as _sil, adjusted_rand_score
def silhouette_score(X, labels):
    import numpy as _np
    if len(_np.unique(labels)) < 2: return float('nan')
    n=len(X); ss=min(1000,n)
    return _sil(X, labels, sample_size=ss, random_state=0)
import vade as vade_mod

DATASETS = ["Annthyroid", "Cardiotocography", "WDBC"]


def stats(labels):
    u, c = np.unique(labels, return_counts=True)
    return len(u), int(c.max()), int(c.min())


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dss = {d.name: d for d in load_npz_dir(os.path.join(ROOT, "data", "dami"))}
    for nm in DATASETS:
        if nm not in dss:
            print(f"skip {nm}"); continue
        ds = dss[nm]; X = ds.X; n, d = X.shape
        print(f"\n=== {nm}: n={n} d={d} ===")
        # PCA + KMeans reference
        Zp = embed(X, 16); labp = cluster(Zp, K=30, seed=0)
        kk, kmax, kmin = stats(labp)
        sil_p = silhouette_score(Zp, labp)
        print(f"  PCA16+KMeans: effK={kk}/30 sizes[{kmin},{kmax}] silhouette(latent)={sil_p:.3f}")

        # VaDE default and variants
        for tag, kw in [("default", {}),
                        ("joint120", {"joint_epochs": 120}),
                        ("pre80_joint120", {"pre_epochs": 80, "joint_epochs": 120})]:
            t = time.time()
            Zv, labv = vade_mod.fit_vade(X, K=30, seed=0, **kw)
            vk, vmax, vmin = stats(labv)
            sil_v = silhouette_score(Zv, labv)
            sil_orig = silhouette_score(X, labv)
            ari = adjusted_rand_score(labp, labv)
            print(f"  VaDE[{tag:16s}]: effK={vk}/30 sizes[{vmin},{vmax}] "
                  f"sil(latent)={sil_v:.3f} sil(orig)={sil_orig:.3f} ARI_vs_KM={ari:.3f} ({time.time()-t:.0f}s)")
        # also KMeans silhouette in ORIGINAL space for apples-to-apples with VaDE sil(orig)
        sil_p_orig = silhouette_score(X, labp)
        print(f"  (PCA+KMeans silhouette in ORIG space = {sil_p_orig:.3f})")


if __name__ == "__main__":
    main()
