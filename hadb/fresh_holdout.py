# -*- coding: utf-8 -*-
"""FRESH-HOLDOUT confirmation of the simple feature-shuffle synthetic selector.

Everything so far was tuned on 20 oddbench+ucr datasets. This runs the SIMPLE config
(single-donor, hard-replace, no filter - the grid winner, which is also the least-tuned) on
corpora NEVER used in dev iteration: OvrBench (tabular) and TSB-AD-U (time series). Reports
the decisive within-dataset Spearman AND the actual selection regret vs EM and random.

PASS if on fresh data: within-dataset rho stays positive (~>0.10) AND shuffle regret < EM regret.
FAIL (dev-overfit) if rho collapses to ~0 or shuffle loses to EM.
"""
import os, sys, io, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import val_tabular, val_tsbad_u, TAB_POOL, TS_POOL, sample_holdout, true_apnorm


def gen_shuffle(Xn, n_synth, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape
    out = np.empty((n_synth, d))
    for r in range(n_synth):
        base = Xn[rng.integers(n)].copy()
        k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        cols = rng.choice(d, k, replace=False)
        out[r] = base
        out[r, cols] = Xn[rng.integers(n)][cols]     # single donor, hard replace
    return out


def synth_auc(Xval, synth, pool, seed=0):
    g = np.random.default_rng(seed); idx = np.arange(len(Xval)); g.shuffle(idx)
    cut = int(0.7 * len(idx)); tr, ho = idx[:cut], idx[cut:]
    if len(tr) < 30 or len(ho) < 10:
        return {}
    Xtr, Xev = Xval[tr], np.vstack([Xval[ho], synth]); yev = np.r_[np.zeros(len(ho)), np.ones(len(synth))]
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


HOLD = [("ovrbench", "hadb_ovrbench.csv", val_tabular, TAB_POOL, sample_holdout("ovrbench", 16)),
        ("tsbad_u", "hadb_ts_tsbad.csv", val_tsbad_u, TS_POOL, sample_holdout("tsbad_u", 14))]

rho, reg_shuf, reg_em, reg_rand = [], [], [], []
pair = []
for corpus, csv, loader, pool, names in HOLD:
    for name in names:
        try:
            Xval = loader("ovrbench", name) if corpus == "ovrbench" else loader(name)
        except Exception as e:
            print(f"  [skip {name[:26]}] {type(e).__name__}"); continue
        if len(Xval) < 80:
            continue
        try:
            ap, emv = true_apnorm(csv, name)
        except Exception:
            continue
        synth = gen_shuffle(Xval, 150)
        sa = synth_auc(Xval, synth, pool)
        common = [v for v in sa if v in ap.index]
        if len(common) < 5:
            continue
        sv = np.array([sa[v] for v in common]); tv = np.array([ap[v] for v in common])
        if np.std(sv) == 0 or np.std(tv) == 0:
            continue
        r = spearmanr(sv, tv).statistic
        best = ap.max()
        shuf_pick = common[int(np.argmax(sv))]
        em_pick = emv.loc[common].idxmax() if emv.loc[common].notna().any() else None
        rho.append(r)
        reg_shuf.append(best - ap[shuf_pick])
        reg_em.append(best - ap[em_pick] if em_pick else np.nan)
        reg_rand.append(best - ap[common].mean())
        pair.append((best - ap[shuf_pick], best - ap[em_pick] if em_pick else np.nan))
        print(f"  {corpus:9s} {name[:28]:30s} rho={r:+.3f}  regret shuf={reg_shuf[-1]:.3f} em={reg_em[-1]:.3f}")

rho = np.array(rho); rs = np.array(reg_shuf); re = np.array(reg_em); rr = np.array(reg_rand)
print(f"\n=== FRESH HOLDOUT ({len(rho)} datasets, corpora never used in dev) ===")
print(f"  within-dataset Spearman(synth_AUC, true): mean {np.nanmean(rho):+.3f}  "
      f"median {np.nanmedian(rho):+.3f}  >0 on {int((rho>0).sum())}/{len(rho)}")
print(f"  mean regret:  shuffle {np.nanmean(rs):.4f}   EM {np.nanmean(re):.4f}   random {np.nanmean(rr):.4f}")
m = ~np.isnan(rs) & ~np.isnan(re)
if m.sum() > 5 and (rs[m] != re[m]).any():
    p = wilcoxon(rs[m], re[m]).pvalue
    print(f"  shuffle vs EM regret (paired Wilcoxon): p={p:.4f}  "
          f"shuffle {'<' if np.nanmean(rs)<np.nanmean(re) else '>='} EM")
print(f"\n  VERDICT: {'CONFIRMED - shuffle holds on fresh data' if np.nanmean(rho)>0.08 and np.nanmean(rs)<np.nanmean(re) else 'NOT confirmed - dev overfit or ties EM'}")
