# -*- coding: utf-8 -*-
"""Two-mode synthetic generator on the FRESH holdout (ovrbench + tsbad_u, never used in dev).

  SHUFFLE      breaks joint structure, stays central (dependency-violation anomalies).
  DISPLACE     moves a normal toward/past another cluster centre in PCA latent -> radially
               displaced anomalies, the type shuffling geometrically cannot make.
  TWO-MODE     pool both. A detector must separate BOTH types to score high, matching the mix
               of real hard anomalies (which are displaced AND structure-broken).

Reports within-dataset Spearman AND regret vs shuffle-alone, displace-alone, EM, and the
two-mode+EM ensemble. Bar: two-mode (or its ensemble) regret < EM (0.157 on this holdout).
"""
import os, sys, io, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import val_tabular, val_tsbad_u, TAB_POOL, TS_POOL, sample_holdout, true_apnorm


def gen_shuffle(Xn, ns, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d))
    for r in range(ns):
        b = Xn[rng.integers(n)].copy(); k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        c = rng.choice(d, k, replace=False); out[r] = b; out[r, c] = Xn[rng.integers(n)][c]
    return out


def gen_displace(Xn, ns, seed=1):
    rng = np.random.default_rng(seed); n, d = Xn.shape
    sc = StandardScaler().fit(Xn); Zs = sc.transform(Xn)
    pca = PCA(n_components=min(16, d), random_state=0).fit(Zs); Z = pca.transform(Zs)
    K = min(20, max(3, n // 40))
    km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5).fit(Z); cen = km.cluster_centers_; lab = km.labels_
    out = []
    for _ in range(ns):
        i = rng.integers(n); zi = Z[i]; ci = lab[i]
        cj = rng.integers(K)
        if cj == ci:
            cj = (cj + 1) % K
        t = rng.uniform(0.5, 1.3)                      # 0.5 between, >1 beyond the other centre
        out.append(zi + t * (cen[cj] - zi))
    return sc.inverse_transform(pca.inverse_transform(np.array(out)))


def synth_auc(Xn, syn, pool, seed=0):
    g = np.random.default_rng(seed); idx = np.arange(len(Xn)); g.shuffle(idx); cut = int(0.7 * len(idx)); tr, ho = idx[:cut], idx[cut:]
    if len(tr) < 30 or len(ho) < 10 or len(syn) < 10:
        return {}
    Xe = np.vstack([Xn[ho], syn]); ye = np.r_[np.zeros(len(ho)), np.ones(len(syn))]; o = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xn[tr])
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                o[vn] = float(roc_auc_score(ye, s))
        except Exception:
            pass
    return o


HOLD = [("ovrbench", "hadb_ovrbench.csv", val_tabular, TAB_POOL, sample_holdout("ovrbench", 16)),
        ("tsbad_u", "hadb_ts_tsbad.csv", val_tsbad_u, TS_POOL, sample_holdout("tsbad_u", 14))]

rho = {k: [] for k in ["shuffle", "displace", "twomode"]}
reg = {k: [] for k in ["shuffle", "displace", "twomode", "em", "ensemble", "random"]}
for corpus, csv, loader, pool, names in HOLD:
    for name in names:
        try:
            Xn = loader("ovrbench", name) if corpus == "ovrbench" else loader(name)
        except Exception:
            continue
        if len(Xn) < 80:
            continue
        try:
            ap, emv = true_apnorm(csv, name)
        except Exception:
            continue
        sh = gen_shuffle(Xn, 150); ds = gen_displace(Xn, 150); tm = np.vstack([sh[:75], ds[:75]])
        auc = {"shuffle": synth_auc(Xn, sh, pool), "displace": synth_auc(Xn, ds, pool),
               "twomode": synth_auc(Xn, tm, pool)}
        common = [v for v in auc["twomode"] if v in ap.index and v in emv.index and emv[v] == emv[v]
                  and v in auc["shuffle"] and v in auc["displace"]]
        if len(common) < 5:
            continue
        tv = np.array([ap[v] for v in common]); ev = np.array([emv[v] for v in common]); best = tv.max()
        for k in ["shuffle", "displace", "twomode"]:
            sv = np.array([auc[k][v] for v in common])
            if np.std(sv) > 0 and np.std(tv) > 0:
                rho[k].append(spearmanr(sv, tv).statistic)
                reg[k].append(best - tv[int(np.argmax(sv))])
        tmv = np.array([auc["twomode"][v] for v in common])
        ens = rankdata(tmv) + rankdata(ev)
        reg["em"].append(best - tv[int(np.argmax(ev))])
        reg["ensemble"].append(best - tv[int(np.argmax(ens))])
        reg["random"].append(best - tv.mean())

print(f"=== TWO-MODE on FRESH holdout ({len(reg['em'])} datasets) ===")
print("  within-dataset Spearman(synth,true):")
for k in ["shuffle", "displace", "twomode"]:
    a = np.array(rho[k]); print(f"    {k:10s} mean {np.nanmean(a):+.3f}  median {np.nanmedian(a):+.3f}")
print("  mean regret:")
for k in ["ensemble", "twomode", "em", "shuffle", "displace", "random"]:
    a = np.array(reg[k]); print(f"    {k:10s} {np.nanmean(a):.4f}")
for k in ["twomode", "ensemble"]:
    a = np.array(reg[k]); e = np.array(reg["em"]); m = ~np.isnan(a) & ~np.isnan(e)
    if m.sum() > 5 and (a[m] != e[m]).any():
        print(f"  {k} vs EM paired Wilcoxon p={wilcoxon(a[m], e[m]).pvalue:.4f}  {k} {'<' if a[m].mean() < e[m].mean() else '>='} EM")
