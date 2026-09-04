# -*- coding: utf-8 -*-
"""When shuffle-synthetic picks a bad detector, does it fail via FALSE NEGATIVES (misses the
real anomalies - scores them like normals) or FALSE POSITIVES (over-flags normals)?

On datasets where synthetic loses to EM: fit the synthetic-picked, EM-picked, and true-best
detectors on TRAIN, score TEST (held-out normals + real hard anomalies), and measure:
  - anomaly rank-pct: median percentile of real anomalies in the detector's test score
    distribution. ~85 = detected; ~50 = missed (look normal); <50 = scored BELOW normals.
  - recall@10%FPR: fraction of real anomalies caught at a 10% false-positive rate.
  - FP-share: at the top-k (k=#anomalies) operating point, how many flags are normals.
Low anomaly-rank / low recall => the failure is FALSE NEGATIVES (the picked detector cannot
see the real anomalies). High FP with anomalies present but normals outranking => false
positives.
"""
import os, sys, io, zipfile, re, contextlib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(S, "scratchpad"))
from dev_common import TAB_POOL, TS_POOL, sample_holdout, sample_dev, true_apnorm
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
POOLD = {v: c for v, c in TAB_POOL}; POOLD.update({v: c for v, c in TS_POOL})


def tab_data(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X)
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    nm = y == 0; anom = np.where(y == 1)[0]
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9; mz = np.abs((X - mu) / sd).max(1)
    hard = anom[mz[anom] <= np.percentile(mz[nm], 99)]
    ni = np.where(nm)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    c1, c2 = int(0.6 * len(idx)), int(0.8 * len(idx))
    return X[ni[idx[:c1]]], X[ni[idx[c1:c2]]], X[ni[idx[c2:]]], X[hard]     # train, val, test-norm, hard


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
    return Xw[tr], Xw[va], Xw[te], Xw[yw == 1]


def gen_shuffle(Xn, ns, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d))
    for r in range(ns):
        b = Xn[rng.integers(n)].copy(); k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        c = rng.choice(d, k, replace=False); out[r] = b; out[r, c] = Xn[rng.integers(n)][c]
    return out


def fit_score(vname, Xtr, Xn, Xa):
    from sklearn.metrics import roc_auc_score
    try:
        m = POOLD[vname]()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(Xtr)
        sn = np.asarray(m.decision_function(Xn), float); sa = np.asarray(m.decision_function(Xa), float)
        if not (np.all(np.isfinite(sn)) and np.all(np.isfinite(sa))):
            return None
        allsc = np.r_[sn, sa]; anom_pct = np.median((np.searchsorted(np.sort(allsc), sa)) / len(allsc) * 100)
        fpr, tpr, thr = roc_curve(np.r_[np.zeros(len(sn)), np.ones(len(sa))], allsc)
        recall10 = float(np.interp(0.10, fpr, tpr))
        return dict(anom_pct=float(anom_pct), recall10=recall10)
    except Exception:
        return None


DS = [("ovrbench", "hadb_ovrbench.csv", TAB_POOL, lambda n: tab_data("ovrbench", n), sample_holdout("ovrbench", 14)),
      ("oddbench", "hadb_oddbench.csv", TAB_POOL, lambda n: tab_data("oddbench", n), sample_dev("oddbench", 10)),
      ("tsbad_u", "hadb_ts_tsbad.csv", TS_POOL, lambda n: ts_data(n, "tsbad/TSB-AD-U.zip"), sample_holdout("tsbad_u", 12))]

rows = []
for corpus, csv, pool, loader, names in DS:
    for name in names:
        try:
            Xtr, Xval, Xten, Xhard = loader(name)
        except Exception:
            continue
        if len(Xtr) < 40 or len(Xval) < 30 or len(Xten) < 10 or len(Xhard) < 5:
            continue
        try:
            ap, emv = true_apnorm(csv, name)
        except Exception:
            continue
        # synthetic AUC per detector (train-fit, corrected protocol)
        syn = gen_shuffle(Xval, 150); Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]
        from sklearn.metrics import roc_auc_score
        sa = {}
        for vn, ct in pool:
            try:
                m = ct()
                with contextlib.redirect_stdout(io.StringIO()):
                    m.fit(Xtr)
                s = np.asarray(m.decision_function(Xe), float)
                if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                    sa[vn] = roc_auc_score(ye, s)
            except Exception:
                pass
        common = [v for v in sa if v in ap.index and v in emv.index and emv[v] == emv[v]]
        if len(common) < 6:
            continue
        av = pd.Series({v: ap[v] for v in common}); sv = pd.Series({v: sa[v] for v in common}); ev = pd.Series({v: emv[v] for v in common})
        if (av.max() - av[sv.idxmax()]) <= (av.max() - av[ev.idxmax()]) + 1e-9:
            continue        # only datasets where synthetic LOSES to EM
        for tag, pick in [("synth_pick", sv.idxmax()), ("em_pick", ev.idxmax()), ("true_best", av.idxmax())]:
            r = fit_score(pick, Xtr, Xten, Xhard)
            if r:
                rows.append(dict(dataset=name, who=tag, **r))
R = pd.DataFrame(rows)
print(f"=== on {R.dataset.nunique()} datasets where synthetic LOSES to EM ===")
print(f"  (anom_pct: percentile of real anomalies in the detector's test scores; recall@10%FPR)\n")
for tag in ["synth_pick", "em_pick", "true_best"]:
    s = R[R.who == tag]
    print(f"  {tag:11s}  anom score-pct median {s.anom_pct.median():5.1f}   recall@10%FPR median {s.recall10.median():.2f}")
print("\n  low anom-pct / low recall for synth_pick vs em_pick => the synthetic-picked detector")
print("  MISSES the real anomalies (false negatives), it does not over-flag normals.")
