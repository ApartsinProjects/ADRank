# -*- coding: utf-8 -*-
"""Fold OddBench's 187 datasets into HADB using the IDENTICAL v3 filter chain.

Same code path as build_hadb_v3.py so the two corpora are directly comparable:
  dedup -> DOUBLE-HARD triviality (max|z| AND LinRes, 99th pct of train-normal)
        -> min hard anomalies -> pool of 41 variants on the PAPER'S target protocol
        -> spread-based inclusion. Stability is RECORDED, never used to filter.

OddBench = MacrOData-CMU (arXiv:2602.09329), CC BY 4.0, cached in data/oddbench/.
Loader matches scripts/modal_oddbench.py: concatenate train+test.

IMPORTANT SCOPE NOTE: OddBench is the paper's FROZEN EXTERNAL TEST SET. Folding it into a
benchmark that will be used for method development destroys that separation. The manifest
therefore tags every row with corpus='oddbench' vs 'dev' so a dev/test split can be
preserved downstream; they must NOT be pooled when reporting a headline.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hadb_round2_common import eval_dataset_3way  # noqa: E402
NPZ_DIR = os.path.join(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d", "scores")
CACHE = os.path.join(ROOT, "data", "ovrbench")
TRIV_PCT, MIN_HARD = 99.0, 20
# The dataset-rate cap was already here; FIX 6 (audit) adds the second one. The test set is
# held-out normals plus ALL hard anomalies, so its rate reaches 0.585 even under the first
# cap, while the UCR arm sits near 0.05. A single spread_ap_norm bar cannot mean the same
# thing across a 10x base-rate gap, since ap_norm divides by exactly that rate.
MAX_DATASET_RATE = 0.25
MAX_TEST_RATE = 0.25
SEEDS = [0, 1, 2]
MAX_N, MAX_D_LINRES = 5000, 120
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\hadb_ovrbench.csv"
if os.environ.get("HADB_NOMAS_ONLY") == "1": OUT = OUT.replace(".csv", "_nomas.csv")

CLASSIC = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
           ("LOF", lambda: LOF(n_neighbors=20)), ("KNN", lambda: KNN(n_neighbors=5)),
           ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
           ("HBOS", lambda: HBOS(n_bins=10)),
           ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
           ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)),
           ("LODA", lambda: LODA(n_bins=10))]

def build_pool():
    p = list(CLASSIC)
    for k in [3, 10, 35, 50, 100, 200]:
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in [5, 20, 50, 100]:
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in [0.3, 0.7, 0.9, 0.99]:
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ncl in [4, 16, 32]:
        p.append((f"CBLOF_c{ncl}", lambda ncl=ncl: CBLOF(n_clusters=ncl, random_state=0)))
    for ne in [50, 300]:
        p.append((f"IF_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=0)))
    for nb in [20, 50]:
        p.append((f"LODA_b{nb}", lambda nb=nb: LODA(n_bins=nb)))
    return p

POOL = build_pool()
CLASSIC_NAMES = {n for n, _ in CLASSIC}


def score(ctor, Xtr, Xte):
    import contextlib, io as _io
    try:
        m = ctor()
        with contextlib.redirect_stdout(_io.StringIO()):
            m.fit(Xtr)
        s = np.asarray(m.decision_function(Xte), float)
        return s if np.all(np.isfinite(s)) else None
    except Exception:
        return None


def linres_scores(X, nm):
    d = X.shape[1]
    if d < 2 or d > MAX_D_LINRES:
        return None
    Xn = X[nm]
    R = np.zeros((len(X), d))
    for k in range(d):
        oth = [i for i in range(d) if i != k]
        try:
            m = Ridge(alpha=1.0).fit(Xn[:, oth], Xn[:, k])
            res = X[:, k] - m.predict(X[:, oth])
            R[:, k] = np.abs(res) / (res[nm].std() + 1e-9)
        except Exception:
            R[:, k] = 0.0
    return R.max(1)


def load(path):
    d = np.load(path, allow_pickle=True)
    tr, trl = np.asarray(d["train"], float), np.asarray(d["train_labels"]).ravel()
    te, tel = np.asarray(d["test"], float), np.asarray(d["test_labels"]).ravel()
    return np.vstack([tr, te]), np.concatenate([trl, tel]).astype(int)


import hashlib
files = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 250
files = sorted(files, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:N_MAX]
print(f"OddBench cached: {len(files)} datasets; pool={len(POOL)} variants", flush=True)

rows, meta, t0 = [], [], time.time()
for i, fn in enumerate(files, 1):
    name = fn[:-4]
    try:
        X, y = load(os.path.join(CACHE, fn))
    except Exception as e:
        meta.append(dict(dataset=name, corpus="ovrbench", status="dropped",
                         reason=f"load_{type(e).__name__}")); continue
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_raw = len(X)
    # FIX 5 (audit): first-occurrence dedup silently kept whichever label came first when
    # identical feature rows carried BOTH. Such anomalies are undetectable in principle, so
    # they pass the triviality filter by construction and enter the benchmark as label noise.
    uniq_rows, first_idx, inv = np.unique(X, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    lo = np.full(len(uniq_rows), 2, int); hi = np.full(len(uniq_rows), -1, int)
    np.minimum.at(lo, inv, y); np.maximum.at(hi, inv, y)
    mixed_grp = lo != hi
    keep_m = np.zeros(len(X), bool); keep_m[first_idx] = True
    keep_m &= ~mixed_grp[inv]
    n_mixed_rows = int(mixed_grp[inv].sum())
    X, y = X[keep_m], y[keep_m]
    dup = 1 - len(X) / max(n_raw, 1)
    if len(X) > MAX_N:
        r = np.random.RandomState(0)
        keep = r.choice(len(X), MAX_N, replace=False); X, y = X[keep], y[keep]

    rec = dict(dataset=name, corpus="ovrbench", n_raw=n_raw, n=len(X), d=X.shape[1],
               dup_frac=round(float(dup), 4), n_mixed_dup_rows=n_mixed_rows,
               n_anom=int(y.sum()), rate=round(float(y.mean()), 5))
    # BASE-RATE CAP: a dataset with a very high "anomaly" fraction is not an anomaly
    # detection dataset (the manifest contained one at 0.867).
    if len(X) < 200 or y.sum() < 10 or y.sum() == len(y) or y.mean() > MAX_DATASET_RATE:
        rec.update(status="dropped",
                   reason="rate_too_high" if y.mean() > MAX_DATASET_RATE
                   else "too_small_or_degenerate")
        meta.append(rec); continue

    nm = y == 0
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9
    maxz = np.abs((X - mu) / sd).max(1)
    lin = linres_scores(X, nm)
    anom = np.where(y == 1)[0]
    keep_z = maxz[anom] <= np.percentile(maxz[nm], TRIV_PCT)
    keep_l = (lin[anom] <= np.percentile(lin[nm], TRIV_PCT)) if lin is not None else True
    hard = anom[keep_z & keep_l]
    rec.update(linres_available=lin is not None, n_hard=int(len(hard)),
               frac_trivial=round(1 - len(hard) / len(anom), 4),
               frac_trivial_maxz_only=round(1 - int(np.sum(keep_z)) / len(anom), 4))
    if len(hard) < MIN_HARD:
        rec.update(status="dropped", reason=f"only_{len(hard)}_double_hard")
        meta.append(rec); continue

    norm_idx = np.where(nm)[0]
    nok = 0
    for seed in SEEDS:
        # THREE-WAY split: train (fit) / val (label-free selection, normals only) / test
        # (ground truth = held-out normals + hard anomalies).
        g = np.random.default_rng(seed)
        idx = np.arange(len(norm_idx)); g.shuffle(idx)
        c1, c2 = int(0.6 * len(idx)), int(0.8 * len(idx))
        tr_i = norm_idx[idx[:c1]]; va_i = norm_idx[idx[c1:c2]]; te_i = norm_idx[idx[c2:]]
        if len(tr_i) < 50 or len(va_i) < 20 or len(te_i) < 20:
            continue
        max_h = int(MAX_TEST_RATE * len(te_i) / (1 - MAX_TEST_RATE))
        if max_h < MIN_HARD:
            continue
        hard_s = (np.sort(np.random.default_rng(1000 + seed).choice(hard, max_h, replace=False))
                  if len(hard) > max_h else hard)
        rec["n_hard_used"] = int(len(hard_s))
        Xte = np.vstack([X[te_i], X[hard_s]])
        yte = np.r_[np.zeros(len(te_i)), np.ones(len(hard_s))]
        r2 = eval_dataset_3way(POOL, X[tr_i], X[va_i], Xte, yte,
                               name, seed, "ovrbench", NPZ_DIR,
                               extra=dict(n_train=len(tr_i), n_val=len(va_i), n_test_norm=len(te_i)))
        rows.extend(r2); nok += len(r2)
    # FIX 7 (audit): status was unconditional, so a dataset whose every variant failed was
    # recorded "scored" with zero result rows.
    rec.update(status="scored" if nok else "dropped",
               reason=None if nok else "no_variant_ran")
    meta.append(rec)
    if i % 5 == 0 or i == len(files):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.DataFrame(meta).to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
        print(f"[{i}/{len(files)}] {name}: hard={len(hard)} "
              f"(triv {100*rec['frac_trivial']:.0f}%) | {time.time()-t0:.0f}s", flush=True)

D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
print(f"\nwrote {OUT} rows={len(D)} datasets={D.dataset.nunique() if len(D) else 0}")

if len(D):
    sc = M0[M0.status == "scored"]
    print(f"\n=== OddBench through the HADB filters ===")
    print(f"  scored {len(sc)}/{len(M0)}; dropped {int((M0.status=='dropped').sum())}")
    print(M0[M0.status == "dropped"].reason.value_counts().head(6).to_string())
    print(f"  mean duplicate fraction: {sc.dup_frac.mean():.3f}")
    print(f"  double-hard removes {sc.frac_trivial.mean():.3f} "
          f"(max|z| alone {sc.frac_trivial_maxz_only.mean():.3f})")

    for metric in ["auc", "pauc10"]:
        gg = D.groupby(["dataset", "seed"])[metric].agg(["max", "mean"])
        M = gg.groupby("dataset").mean()
        M["spread"] = M["max"] - M["mean"]
        zone = lambda b, s: "ceiling" if (b >= 0.95 and s < 0.03) else ("floor" if b < 0.70 else "live")
        M["zone"] = [zone(b, s) for b, s in zip(M["max"], M.spread)]
        print(f"\n=== inclusion on {metric.upper()} ===")
        print("  zones:", M.zone.value_counts().to_dict())
        for thr in [0.05, 0.10, 0.15]:
            k = M[(M.zone == "live") & (M.spread >= thr)]
            print(f"  live AND spread>={thr:.2f}: {len(k):3d}/{len(M)}")
        if metric == "auc":
            M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
