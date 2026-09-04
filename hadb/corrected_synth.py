# -*- coding: utf-8 -*-
"""BUG HUNT: was synthetic handicapped by fitting detectors on a tiny (14%) validation subset,
while the ground truth uses detectors fit on 60% (train)?

FIX: fit the synthetic-scoring detectors on TRAIN (exactly the ground-truth fit, and the same
detectors EM/MV were computed from), then score held-out VALIDATION normals vs shuffle synth.
Compare the OLD (fit-on-val-subset) and NEW (fit-on-train) synthetic against EM on the fresh
holdout. If NEW jumps up, the negative result was partly this bug.
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
from dev_common import TAB_POOL, TS_POOL, sample_holdout, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels


def tab_trainval(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    nm = np.where(y == 0)[0]; g = np.random.default_rng(0); idx = np.arange(len(nm)); g.shuffle(idx)
    c1, c2 = int(0.6 * len(idx)), int(0.8 * len(idx))
    return X[nm[idx[:c1]]], X[nm[idx[c1:c2]]]     # train, val


def ts_trainval(name, zipname):
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
    uq, first, inv = np.unique(Xw, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
    lo_ = np.full(len(uq), 2, int); hi_ = np.full(len(uq), -1, int); np.minimum.at(lo_, inv, yw); np.maximum.at(hi_, inv, yw)
    keep = np.zeros(len(Xw), bool); keep[first] = True; keep &= ~(lo_ != hi_)[inv]; pos = np.where(keep)[0]; Xw, yw = Xw[keep], yw[keep]
    tr, va, te = block_split3(yw, pos, 0)
    return Xw[tr], Xw[va]


def gen_shuffle(Xn, ns, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d))
    for r in range(ns):
        b = Xn[rng.integers(n)].copy(); k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        c = rng.choice(d, k, replace=False); out[r] = b; out[r, c] = Xn[rng.integers(n)][c]
    return out


def synth_auc_OLD(Xtr, Xval, pool):
    """OLD (buggy): fit on 70% of VAL, score rest+synth."""
    g = np.random.default_rng(0); idx = np.arange(len(Xval)); g.shuffle(idx); cut = int(0.7 * len(idx)); tr, ho = idx[:cut], idx[cut:]
    if len(tr) < 30 or len(ho) < 10:
        return {}
    syn = gen_shuffle(Xval, 150); Xe = np.vstack([Xval[ho], syn]); ye = np.r_[np.zeros(len(ho)), np.ones(len(syn))]
    return _score(Xval[tr], Xe, ye, pool)


def synth_auc_NEW(Xtr, Xval, pool):
    """NEW (fixed): fit on TRAIN (matches ground truth), score held-out VAL normals vs synth."""
    if len(Xtr) < 30 or len(Xval) < 20:
        return {}
    syn = gen_shuffle(Xval, 150); Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]
    return _score(Xtr, Xe, ye, pool)


def _score(Xfit, Xe, ye, pool):
    out = {}
    for vn, ct in pool:
        try:
            m = ct()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xfit)
            s = np.asarray(m.decision_function(Xe), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                out[vn] = float(roc_auc_score(ye, s))
        except Exception:
            pass
    return out


HOLD = [("ovrbench", "hadb_ovrbench.csv", TAB_POOL, lambda n: tab_trainval("ovrbench", n), sample_holdout("ovrbench", 16)),
        ("tsbad_u", "hadb_ts_tsbad.csv", TS_POOL, lambda n: ts_trainval(n, "tsbad/TSB-AD-U.zip"), sample_holdout("tsbad_u", 14))]

rho = {"old": [], "new": []}; reg = {"old": [], "new": [], "em": []}
for corpus, csv, pool, loader, names in HOLD:
    for name in names:
        try:
            Xtr, Xval = loader(name)
        except Exception as e:
            print(f"  [skip {name[:26]}] {type(e).__name__}"); continue
        if len(Xtr) < 40 or len(Xval) < 30:
            continue
        try:
            ap, emv = true_apnorm(csv, name)
        except Exception:
            continue
        for tag, fn in [("old", synth_auc_OLD), ("new", synth_auc_NEW)]:
            sa = fn(Xtr, Xval, pool)
            common = [v for v in sa if v in ap.index]
            if len(common) < 5:
                continue
            sv = np.array([sa[v] for v in common]); tv = np.array([ap[v] for v in common])
            if np.std(sv) > 0 and np.std(tv) > 0:
                rho[tag].append(spearmanr(sv, tv).statistic)
                reg[tag].append(tv.max() - tv[int(np.argmax(sv))])
        try:
            common = [v for v in emv.index if v in ap.index and emv[v] == emv[v]]
            reg["em"].append(ap[common].max() - ap[emv.loc[common].idxmax()])
        except Exception:
            pass

print(f"=== OLD (fit on 14% val) vs NEW (fit on 60% train) synthetic, fresh holdout ===")
for tag in ["old", "new"]:
    a = np.array(rho[tag]); r = np.array(reg[tag])
    print(f"  {tag:4s}  within-dataset rho mean {np.nanmean(a):+.3f} median {np.nanmedian(a):+.3f}  |  regret {np.nanmean(r):.4f}  (n={len(a)})")
print(f"  em    regret {np.nanmean(reg['em']):.4f}")
o, nw = np.array(reg["old"]), np.array(reg["new"])
if len(o) == len(nw) and len(o) > 5 and (o != nw).any():
    print(f"  NEW vs OLD regret paired Wilcoxon p={wilcoxon(nw, o).pvalue:.4f}")
