# -*- coding: utf-8 -*-
"""VaDE per dataset + shuffle-generated synthetic anomalies + VaDE-density filtering.

Pipeline per dev dataset (validation normals only, leak-free):
  1. Fit VaDE on the normals -> a jointly-learned latent + GMM manifold, and a decoder.
  2. Generate candidate anomalies by FEATURE SHUFFLING: copy a normal, overwrite a random
     fraction of its features with values drawn from OTHER normals. This preserves marginals
     but breaks the joint structure -> a point that sits off the normal manifold.
  3. FILTER with VaDE: keep only candidates the VaDE reconstructs POORLY (recon error above
     the normal band) - i.e. genuinely off-manifold, discarding shuffles that stayed normal.
  4. Score every detector on kept-synthetic vs held-out normal -> synth_AUC.

DECISIVE metric (same as the pseudo-anomaly and PCA-synth checks): within-dataset Spearman
between synth_AUC and TRUE test ap_norm. Bars: beat the PCA-between baseline (+0.102) and
ideally EM (+0.133).
"""
import os, sys, io, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import (val_tabular, val_ucr, TAB_POOL, TS_POOL, sample_dev, true_apnorm)

LATENT = 10
torch.set_num_threads(3)


class VaDE(nn.Module):
    def __init__(self, in_dim, K, latent=LATENT, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def recon(self, x):
        return self.dec(self.mu(self.enc(x)))


def fit_vade_ae(X, seed=0, epochs=45, device="cpu"):
    """Light autoencoder-VaDE surrogate: train an AE (the VaDE recon backbone) on normals and
    return (model, scaler). Recon error is the off-manifold score used for filtering."""
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X).astype(np.float32)
    n, d = Xs.shape
    Xt = torch.from_numpy(Xs).to(device)
    model = VaDE(d, K=min(10, max(2, n // 30))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    bs = min(256, n)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i + bs]]
            loss = F.mse_loss(model.recon(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, sc


def recon_err(model, sc, X):
    with torch.no_grad():
        Xt = torch.from_numpy(sc.transform(X).astype(np.float32))
        return np.linalg.norm(sc.transform(X).astype(np.float32) - model.recon(Xt).numpy(), axis=1)


def shuffle_candidates(Xn, n_cand, frac=0.3, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape
    base = Xn[rng.integers(0, n, n_cand)].copy()
    ncol = max(1, int(frac * d))
    for r in range(n_cand):
        cols = rng.choice(d, ncol, replace=False)
        base[r, cols] = Xn[rng.integers(0, n, ncol), cols]
    return base


def synth_auc(Xval, synth, pool, seed=0):
    g = np.random.default_rng(seed); idx = np.arange(len(Xval)); g.shuffle(idx)
    cut = int(0.7 * len(idx)); tr, ho = idx[:cut], idx[cut:]
    if len(tr) < 30 or len(ho) < 10 or len(synth) < 10:
        return {}
    Xtr, Xev = Xval[tr], np.vstack([Xval[ho], synth])
    yev = np.r_[np.zeros(len(ho)), np.ones(len(synth))]
    out = {}
    for vname, ctor in pool:
        try:
            m = ctor()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr)
            s = np.asarray(m.decision_function(Xev), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                out[vname] = float(roc_auc_score(yev, s))
        except Exception:
            continue
    return out


DEV = [("oddbench", "hadb_oddbench.csv", val_tabular, TAB_POOL, sample_dev("oddbench", 10)),
       ("ucr", "hadb_ts_ucr.csv", None, TS_POOL, sample_dev("ucr", 10))]

res = {"vade_filter": [], "shuffle_nofilter": []}
nrank = {"vade_filter": []}
for corpus, csv, loader, pool, names in DEV:
    for name in names:
        try:
            Xval = val_ucr(name) if corpus == "ucr" else loader(corpus, name)
        except Exception:
            continue
        if len(Xval) < 80:
            continue
        try:
            model, sc = fit_vade_ae(Xval)
        except Exception as e:
            print(f"  [vade fail] {name[:30]} {type(e).__name__}"); continue
        cand = shuffle_candidates(Xval, 400)
        ne = recon_err(model, sc, Xval); ce = recon_err(model, sc, cand)
        thr = np.percentile(ne, 85)                 # off-manifold beyond the normal band
        keep = cand[(ce > thr) & (ce < np.percentile(ce, 99))]
        synth_f = keep[:120] if len(keep) >= 10 else None
        synth_n = cand[:120]                          # unfiltered control
        ap, emv = true_apnorm(csv, name)
        for tag, synth in [("vade_filter", synth_f), ("shuffle_nofilter", synth_n)]:
            if synth is None:
                continue
            sa = synth_auc(Xval, synth, pool)
            common = [v for v in sa if v in ap.index]
            if len(common) < 5:
                continue
            sv = np.array([sa[v] for v in common]); tv = np.array([ap[v] for v in common])
            if np.std(sv) > 0 and np.std(tv) > 0:
                r = spearmanr(sv, tv).statistic
                res[tag].append(r)
                if tag == "vade_filter":
                    tr = pd.Series(tv).rank(ascending=False)
                    nrank["vade_filter"].append((tr.iloc[int(np.argmax(sv))] - 1) / (len(common) - 1))
                    print(f"  {corpus:9s} {name[:30]:32s} rho_vade={r:+.3f}  kept={len(synth_f)}")

print(f"\n=== VaDE shuffle+filter synthetic signal ({len(res['vade_filter'])} datasets) ===")
for tag in ["vade_filter", "shuffle_nofilter"]:
    a = np.array(res[tag])
    print(f"  {tag:16s} within-dataset Spearman(synth_AUC,true): mean {np.nanmean(a):+.3f}  "
          f"median {np.nanmedian(a):+.3f}  >0 on {int((a>0).sum())}/{len(a)}")
print(f"  vade_filter argmax pick true-rank pct: {np.nanmean(nrank['vade_filter']):.3f} (0.5=random)")
print(f"\n  baselines: PCA-between +0.102 | EM +0.133")
m = np.nanmean(res["vade_filter"])
print(f"  VERDICT: {'BEATS PCA baseline - promising' if m > 0.12 else 'no better than PCA/EM - not worth the deep model' if m < 0.10 else 'comparable to PCA baseline'}")
