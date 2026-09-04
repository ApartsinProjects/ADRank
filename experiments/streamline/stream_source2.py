# -*- coding: utf-8 -*-
"""SIDE PROJECT v2: 3-stage pipeline + Stage 4 dedup, on the FULL source incl. adbench, dami, MTS.
Detector-free. Writes to streamline/ only.
  Stage 1  drop dataset if OR rule catches >= TRIV1 of anomalies (marginal-trivial)
  Stage 2  drop trivial anomalies (calibrated per-feature rarity above normals' (1-Q) quantile)
  Stage 3  drop dataset if < MIN_HARD hard anomalies survive
  Stage 4  drop DUPLICATE datasets (same-source across corpora) by a data fingerprint (d + sorted
           normal feature-moments); keep the copy with the most hard anomalies.
"""
import os, sys, io, zipfile, re, hashlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as P
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
ZIPS = {"ucr": "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip", "tsbad_u": "tsbad/TSB-AD-U.zip"}
Q = 0.05; TRIV1 = 0.90; MIN_HARD = 10; MAX_MTS = 200
_ZIP = {}
def zf(p):
    if p not in _ZIP: _ZIP[p] = zipfile.ZipFile(os.path.join(ROOT, "data", p))
    return _ZIP[p]


def split_norm_anom(X, y):
    X = np.nan_to_num(np.asarray(X, float)); y = np.asarray(y, int).ravel()
    r = np.random.default_rng(0); ni = np.where(y == 0)[0]; ai = np.where(y == 1)[0]
    if len(ni) > 6000: ni = r.choice(ni, 6000, replace=False)
    idx = np.arange(len(ni)); r.shuffle(idx); ni = ni[idx]
    return X[ni[:int(0.6 * len(ni))]], X[ni[int(0.6 * len(ni)):]], X[ai]


def tab_load(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]); y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    return split_norm_anom(X, y)


def uni_load(name, zipname):
    z = zf(zipname)
    cand = [n for n in z.namelist() if os.path.basename(n) == name] or [n for n in z.namelist() if n.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]
    if fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()]); a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    Xw, _ = _window_features(x, w=W, stride=STRIDE); st = np.arange(0, len(x) - W + 1, STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
    return Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[(yw == 1)]


def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); hist = -np.log(dens[b] + 1e-9)
        sev = np.maximum(sev, np.maximum(tail, hist))
    return sev


def fingerprint(Xtr, Xn):
    Z = np.vstack([Xtr, Xn]); mu = np.round(np.sort(Z.mean(0)), 3); sd = np.round(np.sort(Z.std(0)), 3)
    return (Z.shape[1], hashlib.sha1(mu.tobytes()).hexdigest()[:10], hashlib.sha1(sd.tobytes()).hexdigest()[:10])


# ---- enumerate source ----
def sources():
    for corp, f in CSVMAP.items():
        for name in sorted(pd.read_csv(os.path.join(S, f)).dataset.unique()):
            yield corp, name, ("tab" if corp in ("oddbench", "ovrbench") else "uni")
    for sub in ("adbench", "dami"):
        dd = os.path.join(ROOT, "data", sub)
        if os.path.isdir(dd):
            for ds in P.load_npz_dir(dd): yield sub, ds.name, ("obj", ds)
    for name, src, Xc, lab in load_mts(MAX_MTS): yield "tsbad_m", name, ("mts", Xc, lab)


rows = []
for corp, name, how in sources():
    try:
        if how == "tab": Xtr, Xn, Xa = tab_load(corp, name)
        elif how == "uni": Xtr, Xn, Xa = uni_load(name, ZIPS[corp])
        elif isinstance(how, tuple) and how[0] == "obj": Xtr, Xn, Xa = split_norm_anom(how[1].X, how[1].y)
        elif isinstance(how, tuple) and how[0] == "mts":
            Xc, lab = how[1], how[2]
            if len(Xc) < W + 10: continue
            Xw, starts = mts_window_features(Xc); yw = mts_wlabels(lab, starts); Xw = np.nan_to_num(Xw)
            pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
            Xtr, Xn, Xa = Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[(yw == 1)]
        else: continue
    except Exception:
        continue
    if len(Xtr) < 40 or len(Xn) < 20 or len(Xa) < 3: continue
    sev_n = severity(Xtr, Xn); sev_a = severity(Xtr, Xa)
    thr = np.quantile(sev_n, 1 - Q); triv = sev_a > thr
    # Stage-5 FAIR OR detectors: thresholds from TRAIN normals; measure anomaly recall AND normal FP.
    d = Xtr.shape[1]
    mn, mx = Xtr.min(0), Xtr.max(0); q1, q99 = np.percentile(Xtr, 1, 0), np.percentile(Xtr, 99, 0)
    rec_hard = float(((Xa < mn) | (Xa > mx)).any(1).mean()); fp_hard = float(((Xn < mn) | (Xn > mx)).any(1).mean())
    rec_soft = float(((Xa < q1) | (Xa > q99)).any(1).mean()); fp_soft = float(((Xn < q1) | (Xn > q99)).any(1).mean())
    # Bonferroni-calibrated soft OR: per-feature two-sided tail alpha so total OR normal-FP ~= 5%
    alpha = (1 - 0.95 ** (1.0 / max(d, 1))) / 2; qlo = np.percentile(Xtr, 100 * alpha, 0); qhi = np.percentile(Xtr, 100 * (1 - alpha), 0)
    rec_cal = float(((Xa < qlo) | (Xa > qhi)).any(1).mean()); fp_cal = float(((Xn < qlo) | (Xn > qhi)).any(1).mean())
    rows.append({"corpus": corp, "dataset": str(name)[:40], "n_anom": len(Xa), "n_norm": len(Xn), "d": d,
                 "frac_triv": float(triv.mean()), "n_hard": int((~triv).sum()),
                 "rec_hard": rec_hard, "fp_hard": fp_hard, "rec_soft": rec_soft, "fp_soft": fp_soft,
                 "rec_cal": rec_cal, "fp_cal": fp_cal, "net_soft": rec_soft - fp_soft, "net_cal": rec_cal - fp_cal,
                 "fp": "%d|%s|%s" % fingerprint(Xtr, Xn)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "STREAM_SOURCE2.csv"), index=False)
s1 = df[df.frac_triv < TRIV1]; s3 = s1[s1.n_hard >= MIN_HARD].copy()
s3 = s3.sort_values("n_hard", ascending=False); s4 = s3.drop_duplicates("fp", keep="first")
print(f"=== streamlined pipeline on FULL SOURCE ({len(df)} loaded candidates) ===")
print(f"  STAGE 1 (frac trivial < {TRIV1}):      {len(s1):4d}   (dropped {len(df)-len(s1)} OR-solvable)")
print(f"  STAGE 3 (>= {MIN_HARD} hard anomalies):     {len(s3):4d}   (dropped {len(s1)-len(s3)} too-few-hard)")
print(f"  STAGE 4 (dedup by data fingerprint):  {len(s4):4d}   (dropped {len(s3)-len(s4)} duplicates)")
print(f"\n  per-corpus (loaded -> stage1 -> stage3 -> stage4 dedup):")
for c in ["oddbench", "ovrbench", "ucr", "tsbad_u", "adbench", "dami", "tsbad_m"]:
    print(f"    {c:10s} {int((df.corpus==c).sum()):4d} -> {int((s1.corpus==c).sum()):4d} -> {int((s3.corpus==c).sum()):4d} -> {int((s4.corpus==c).sum()):4d}")
print(f"\n  after stage4: {len(s4)} datasets   median hardened base rate {(s4.n_hard/(s4.n_hard+s4.n_norm)).median():.3f}")
print(f"\n  STAGE 5 FAIR DIAGNOSTIC on the {len(s4)} survivors (recorded, not applied).")
print(f"  Each OR detector: thresholds from TRAIN normals; report anomaly RECALL and normal FP.")
print(f"    detector            mean_recall  mean_FP   (a fair triviality filter uses NET = recall - FP)")
print(f"    strict min/max        {s4.rec_hard.mean():.2f}       {s4.fp_hard.mean():.2f}")
print(f"    soft Q1/Q99           {s4.rec_soft.mean():.2f}       {s4.fp_soft.mean():.2f}   <- raw Q1/Q99 (high FP in high-d)")
print(f"    calibrated (5% FP)    {s4.rec_cal.mean():.2f}       {s4.fp_cal.mean():.2f}   <- Bonferroni per-feature quantile")
print(f"\n  drop counts of the {len(s4)} if we filter datasets a soft OR SEPARABLY solves:")
print(f"    {'tau5':>6s} {'NET soft (rec-FP)':>18s} {'recall_cal @5%FP':>18s} {'raw rec_soft':>14s}")
for t in [0.90, 0.70, 0.50, 0.30]:
    print(f"    {t:6.2f} {int((s4.net_soft>=t).sum()):18d} {int((s4.rec_cal>=t).sum()):18d} {int((s4.rec_soft>=t).sum()):14d}")
s4.to_csv(os.path.join(OUT, "STREAM_SOURCE2_FINAL.csv"), index=False)
print("saved streamline/STREAM_SOURCE2.csv, STREAM_SOURCE2_FINAL.csv")
