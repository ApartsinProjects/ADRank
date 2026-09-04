# -*- coding: utf-8 -*-
"""Is shuffle-synthetic good WITHIN a family (config selection) even though it is biased ACROSS
families (local bias)? If so, a hybrid - EM picks the family, synthetic picks the config within
it - could beat EM alone.

Corrected protocol throughout: detectors fit on TRAIN, scored on VALIDATION (normals) + synth.
"""
import os, sys, io, zipfile, re, contextlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL, sample_holdout, sample_dev, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels

LOCAL = ("LOF", "KNN", "CBLOF"); GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")
fam = lambda v: "local" if str(v).startswith(LOCAL) else ("global" if str(v).startswith(GLOBAL) else "other")


def tab_data(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    ni = np.where(y == 0)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    c1, c2 = int(0.6 * len(idx)), int(0.8 * len(idx))
    return X[ni[idx[:c1]]], X[ni[idx[c1:c2]]]


def ts_data(name, zipname):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", zipname))
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or \
           [n for n in z.namelist() if n.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]
    if fn.endswith(".csv"):
        import io as _io
        df = pd.read_csv(_io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()])
        a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2
        lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    st = np.arange(0, len(x) - W + 1, STRIDE); Xw, _ = _window_features(x, w=W, stride=STRIDE)
    yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
    return Xw[tr], Xw[va]


def gen_shuffle(Xn, ns, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d))
    for r in range(ns):
        b = Xn[rng.integers(n)].copy(); k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        c = rng.choice(d, k, replace=False); out[r] = b; out[r, c] = Xn[rng.integers(n)][c]
    return out


def synth_auc(Xtr, Xval, pool):
    syn = gen_shuffle(Xval, 150); Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; out = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                out[vn] = float(roc_auc_score(ye, s))
        except Exception:
            pass
    return out


DS = [("ovrbench", "hadb_ovrbench.csv", TAB_POOL, lambda n: tab_data("ovrbench", n), sample_holdout("ovrbench", 14)),
      ("oddbench", "hadb_oddbench.csv", TAB_POOL, lambda n: tab_data("oddbench", n), sample_dev("oddbench", 10)),
      ("tsbad_u", "hadb_ts_tsbad.csv", TS_POOL, lambda n: ts_data(n, "tsbad/TSB-AD-U.zip"), sample_holdout("tsbad_u", 12))]

wf_rho = []          # within-family rho (config selection quality)
reg = {"em": [], "synth": [], "hybrid": [], "oracle_fam_synth": []}
for corpus, csv, pool, loader, names in DS:
    for name in names:
        try:
            Xtr, Xval = loader(name)
        except Exception:
            continue
        if len(Xtr) < 40 or len(Xval) < 30:
            continue
        try:
            ap, emv = true_apnorm(csv, name)
        except Exception:
            continue
        sa = synth_auc(Xtr, Xval, pool)
        common = [v for v in sa if v in ap.index and v in emv.index and emv[v] == emv[v]]
        if len(common) < 6:
            continue
        av = pd.Series({v: ap[v] for v in common}); sv = pd.Series({v: sa[v] for v in common}); ev = pd.Series({v: emv[v] for v in common})
        best = av.max()
        # within-family rho
        for f in ["local", "global", "other"]:
            fv = [v for v in common if fam(v) == f]
            if len(fv) >= 4 and sv[fv].std() > 0 and av[fv].std() > 0:
                wf_rho.append(spearmanr(sv[fv], av[fv]).statistic)
        # selectors
        em_pick = ev.idxmax(); synth_pick = sv.idxmax()
        reg["em"].append(best - av[em_pick])
        reg["synth"].append(best - av[synth_pick])
        # hybrid: EM picks the family; synthetic picks the best config in that family
        emf = fam(em_pick); infam = [v for v in common if fam(v) == emf]
        reg["hybrid"].append(best - av[sv[infam].idxmax()] if infam else best - av[em_pick])
        # upper bound: oracle family, synth config
        of = fam(av.idxmax()); of_v = [v for v in common if fam(v) == of]
        reg["oracle_fam_synth"].append(best - av[sv[of_v].idxmax()] if of_v else 0.0)

print(f"=== within-family config selection ({len(reg['em'])} datasets) ===")
a = np.array(wf_rho)
print(f"  within-family Spearman(synth_AUC, true): mean {np.nanmean(a):+.3f} median {np.nanmedian(a):+.3f}  >0 {int((a>0).sum())}/{len(a)}")
print(f"\n  mean regret:")
for k in ["hybrid", "em", "synth", "oracle_fam_synth"]:
    print(f"    {k:18s} {np.nanmean(reg[k]):.4f}")
h, e = np.array(reg["hybrid"]), np.array(reg["em"])
if (h != e).any():
    print(f"  hybrid vs EM paired Wilcoxon p={wilcoxon(h, e).pvalue:.4f}  hybrid {'<' if h.mean() < e.mean() else '>='} EM")
