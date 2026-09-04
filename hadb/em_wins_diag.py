# -*- coding: utf-8 -*-
"""On datasets where EM beats shuffle-synthetic, WHY does EM's top detector not detect the
synthetic anomalies?

Per dataset (fit-on-TRAIN scoring, the corrected/fair version):
  - EM's pick and shuffle's pick, their FAMILIES (local/global/other), their true ap_norm.
  - EM-pick's synthetic AUC RANK (is EM's good detector scored LOW by the synthetic task?).
  - real hard-anomaly displacement (median radial-pct).
Hypothesis: on EM-wins datasets the real anomalies are DISPLACED, EM picks a GLOBAL/distance
detector (good at displaced), but that detector scores LOW on shuffle synthetics (which are
central/structure-broken, a task global detectors are weak at) -> shuffle mis-ranks it and
picks a local detector that fails on the displaced real anomalies.
"""
import os, sys, io, zipfile, re, contextlib, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
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
    nm = y == 0; anom = np.where(y == 1)[0]
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9; mz = np.abs((X - mu) / sd).max(1)
    hard = anom[mz[anom] <= np.percentile(mz[nm], 99)]
    ni = np.where(nm)[0]; g = np.random.default_rng(0); idx = np.arange(len(ni)); g.shuffle(idx)
    c1, c2 = int(0.6 * len(idx)), int(0.8 * len(idx))
    return X[ni[idx[:c1]]], X[ni[idx[c1:c2]]], X[hard]


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
    return Xw[tr], Xw[va], Xw[yw == 1]


def gen_shuffle(Xn, ns, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d))
    for r in range(ns):
        b = Xn[rng.integers(n)].copy(); k = int(rng.integers(1, max(2, int(0.6 * d) + 1)))
        c = rng.choice(d, k, replace=False); out[r] = b; out[r, c] = Xn[rng.integers(n)][c]
    return out


def synth_auc(Xtr, Xval, syn, pool):
    Xe = np.vstack([Xval, syn]); ye = np.r_[np.zeros(len(Xval)), np.ones(len(syn))]; out = {}
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


def real_radial(Xtr, Xhard):
    Xn = Xtr
    sc = StandardScaler().fit(Xn); Zn = sc.transform(Xn); Za = sc.transform(Xhard)
    if Zn.shape[1] > 16:
        p = PCA(16, random_state=0).fit(Zn); Zn, Za = p.transform(Zn), p.transform(Za)
    cen = MiniBatchKMeans(min(20, max(2, len(Zn) // 30)), random_state=0, n_init=5).fit(Zn).cluster_centers_
    rn = np.linalg.norm(Zn[:, None] - cen[None], axis=2).min(1); ra = np.linalg.norm(Za[:, None] - cen[None], axis=2).min(1)
    return float(np.median(np.searchsorted(np.sort(rn), ra) / len(rn) * 100))


DS = [("ovrbench", "hadb_ovrbench.csv", TAB_POOL, lambda n: tab_data("ovrbench", n), sample_holdout("ovrbench", 14)),
      ("oddbench", "hadb_oddbench.csv", TAB_POOL, lambda n: tab_data("oddbench", n), sample_dev("oddbench", 10)),
      ("tsbad_u", "hadb_ts_tsbad.csv", TS_POOL, lambda n: ts_data(n, "tsbad/TSB-AD-U.zip"), sample_holdout("tsbad_u", 12))]

rows = []
for corpus, csv, pool, loader, names in DS:
    for name in names:
        try:
            Xtr, Xval, Xhard = loader(name)
        except Exception:
            continue
        if len(Xtr) < 40 or len(Xval) < 30 or len(Xhard) < 5:
            continue
        try:
            ap, emv = true_apnorm(csv, name)
        except Exception:
            continue
        sa = synth_auc(Xtr, Xval, gen_shuffle(Xval, 150), pool)
        common = [v for v in sa if v in ap.index and v in emv.index and emv[v] == emv[v]]
        if len(common) < 5:
            continue
        av = pd.Series({v: ap[v] for v in common}); sv = pd.Series({v: sa[v] for v in common}); ev = pd.Series({v: emv[v] for v in common})
        best = av.max()
        em_pick = ev.idxmax(); shuf_pick = sv.idxmax()
        em_reg = best - av[em_pick]; shuf_reg = best - av[shuf_pick]
        # synth-AUC percentile-rank of EM's pick (low => synthetic scores EM's good detector poorly)
        sa_rank = sv.rank(pct=True)[em_pick]
        rows.append(dict(dataset=name, corpus=corpus, em_wins=int(em_reg < shuf_reg - 1e-9),
                         em_pick=em_pick, em_fam=fam(em_pick), em_reg=em_reg,
                         shuf_pick=shuf_pick, shuf_fam=fam(shuf_pick), shuf_reg=shuf_reg,
                         em_pick_synth_rank=sa_rank, real_radial=real_radial(Xtr, Xhard)))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(S, "em_wins_diag.csv"), index=False)
W_ = R[R.em_wins == 1]; L_ = R[R.em_wins == 0]
print(f"=== {len(R)} datasets: EM beats shuffle on {len(W_)}, shuffle >= EM on {len(L_)} ===\n")
print("                          EM-wins   shuffle>=EM")
print(f"  real anomaly radial-pct  {W_.real_radial.median():6.0f}      {L_.real_radial.median():6.0f}   (higher=more displaced)")
print(f"  EM-pick synth-AUC rank   {W_.em_pick_synth_rank.median():6.2f}      {L_.em_pick_synth_rank.median():6.2f}   (low=synthetic scores EM's pick poorly)")
print(f"  EM-pick family global %  {100*(W_.em_fam=='global').mean():6.0f}      {100*(L_.em_fam=='global').mean():6.0f}")
print(f"  EM-pick family local  %  {100*(W_.em_fam=='local').mean():6.0f}      {100*(L_.em_fam=='local').mean():6.0f}")
print(f"  shuffle-pick local    %  {100*(W_.shuf_fam=='local').mean():6.0f}      {100*(L_.shuf_fam=='local').mean():6.0f}")
print(f"\n  on EM-wins datasets: EM picks {dict(W_.em_fam.value_counts())}, shuffle picks {dict(W_.shuf_fam.value_counts())}")
print(f"  corr(real_radial, em_pick_synth_rank) = {R[['real_radial','em_pick_synth_rank']].corr().iloc[0,1]:+.3f}")
