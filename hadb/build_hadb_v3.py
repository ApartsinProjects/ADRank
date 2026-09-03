# -*- coding: utf-8 -*-
"""HADB v3 - Hard Anomaly-detection Benchmark for MODEL SELECTION.

Revised after methodological review. Four changes from v2:

  FIX 1 (S6 REMOVED). v2 excluded "badly-fit" models via label-free bootstrap stability.
     That is exactly the task a label-free selection method is being tested on, so building
     it into the benchmark contaminates the comparison. It also penalises IForest/LODA,
     which are stochastic by design. Stability is now recorded as a COLUMN so it can serve
     as a BASELINE METHOD ("filter by stability, then pick"), never as a construction step.

  FIX 2 (DOUBLE-HARD triviality). v2 used only max_k |z_k|, which catches marginal extremes
     and misses anomalies trivial in other ways. Wu & Keogh's definition is solution-based
     and strictly stronger. We adopt the stricter criterion from Apartsin & Aperstein
     (arXiv:2607.06094): an anomaly is DOUBLE-HARD only if separated by NEITHER the
     univariate max|z| rule NOR a linear cross-channel predictor (LinRes), both thresholded
     at the 99th percentile of ALL NORMALS.
     The threshold uses every normal point, not only the training split, BY DESIGN: the hard
     set is a property of the DATASET and must be identical across seeds, or the benchmark's
     ground truth would shift with the split. An audit measured the alternative (train-normals
     only, seed 0): 167 of 10,146 hard anomalies change status, 1.6%, and no dataset crosses
     the MIN_HARD gate. An earlier version of this docstring said TRAIN-NORMAL; the code was
     always as described here.

  FIX 3 (THRESHOLD SWEEPS). Every cut is reported across a range instead of one hand-picked
     value, so the benchmark is not silently tuned.

  FIX 4 (DEDUPLICATION). Duplicate rows inflate training density and test AUC; 16_http is
     60.9% duplicates and single-handedly carried 64% of one earlier headline. Duplicates
     are removed and the fraction reported.

KNOWN AND DOCUMENTED LIMITATION (not fixable, must be stated wherever HADB is used):
  Datasets are selected using LABEL-DERIVED spread, then a LABEL-FREE method is evaluated on
  the survivors. HADB therefore measures selection skill CONDITIONAL ON THE CHOICE
  MATTERING. Regret reductions on HADB are systematically higher than in deployment and are
  NOT comparable to ADBench-style figures averaged over all datasets.

SECOND DOCUMENTED RISK:
  Filtering out easy anomalies may ENRICH FOR MISLABELLED POINTS (Wu & Keogh's third flaw):
  a point no detector can find may not be an anomaly. We report, per dataset, the fraction
  of surviving hard anomalies that are indistinguishable from normals under every pool
  member (max pool AUC contribution), as a mislabelling proxy.
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
import adrank.pipeline as P  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hadb_round2_common import eval_dataset_3way  # noqa: E402
NPZ_DIR = os.path.join(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d", "scores")

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA

TRIV_PCTS = [95.0, 99.0, 99.5]     # FIX 3: sweep, do not hand-pick
MIN_HARDS = [10, 20, 40]
SPREADS = [0.05, 0.10, 0.15]
PRIMARY_PCT, PRIMARY_MIN_HARD = 99.0, 20
# FIX 6 (audit): the base-rate cap lived only in the OvrBench arm. 25 scored OddBench
# datasets exceed 0.25 and 9 reached the include list with TEST base rates 0.35-0.87, where
# ap_norm = (ap - base)/(1 - base) is near-degenerate. Two caps: one on the dataset's own
# anomaly rate, one on the TEST rate that ap_norm actually divides by, since the test set is
# held-out normals plus ALL hard anomalies and can reach 0.58 even under the first cap.
MAX_DATASET_RATE = 0.25
MAX_TEST_RATE = 0.25
SEEDS = [0, 1, 2]
MAX_N = 5000
MAX_D_LINRES = 120                 # LinRes fits one ridge per feature; cap for runtime
OUT = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d\hadb_v3.csv"
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


def linres_scores(X, norm_mask):
    """FIX 2: max standardized residual predicting each feature from the others.
    Fitted on NORMAL points only. Returns None when d is out of range."""
    d = X.shape[1]
    if d < 2 or d > MAX_D_LINRES:
        return None
    Xn = X[norm_mask]
    R = np.zeros((len(X), d))
    for k in range(d):
        oth = [i for i in range(d) if i != k]
        try:
            m = Ridge(alpha=1.0).fit(Xn[:, oth], Xn[:, k])
            res = X[:, k] - m.predict(X[:, oth])
            R[:, k] = np.abs(res) / (res[norm_mask].std() + 1e-9)
        except Exception:
            R[:, k] = 0.0
    return R.max(1)


rows, meta, t0 = [], [], time.time()
datasets = []
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        datasets += P.load_npz_dir(dd)
print(f"pool={len(POOL)} variants; datasets={len(datasets)}", flush=True)

for ds in datasets:
    X, y = np.asarray(ds.X, float), np.asarray(ds.y, int).ravel()
    n_before = len(X)
    # FIX 4: deduplicate.
    # FIX 5 (audit): keeping the first occurrence silently assigned whichever label came
    # first when identical feature rows carried BOTH. 51 datasets have such groups; in the
    # scored OddBench sets up to 34% of one hard set were anomalies identical to a normal
    # row. Those are undetectable in principle, so they pass the triviality filter by
    # construction and enter the benchmark as pure label noise. Drop the whole group.
    uniq_rows, first_idx, inv = np.unique(X, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    lo = np.full(len(uniq_rows), 2, int); hi = np.full(len(uniq_rows), -1, int)
    np.minimum.at(lo, inv, y); np.maximum.at(hi, inv, y)
    mixed_grp = lo != hi
    keep_m = np.zeros(len(X), bool); keep_m[first_idx] = True
    keep_m &= ~mixed_grp[inv]
    n_mixed_rows = int((mixed_grp[inv]).sum())
    X, y = X[keep_m], y[keep_m]
    dup_frac = 1 - len(X) / n_before
    if len(X) > MAX_N:
        r = np.random.RandomState(0)
        keep = r.choice(len(X), MAX_N, replace=False)
        X, y = X[keep], y[keep]

    rec = dict(dataset=ds.name, n_raw=n_before, n=len(X), d=X.shape[1],
               dup_frac=round(float(dup_frac), 4), n_mixed_dup_rows=n_mixed_rows,
               n_anom=int(y.sum()), rate=round(float(y.mean()), 5))
    if y.sum() < 10 or y.sum() == len(y):
        rec.update(status="dropped", reason="too_few_anomalies"); meta.append(rec); continue
    if y.mean() > MAX_DATASET_RATE:
        rec.update(status="dropped", reason="rate_too_high"); meta.append(rec); continue

    nm = y == 0
    mu, sd = X[nm].mean(0), X[nm].std(0) + 1e-9
    maxz = np.abs((X - mu) / sd).max(1)
    lin = linres_scores(X, nm)
    anom = np.where(y == 1)[0]
    rec["linres_available"] = lin is not None

    # FIX 2 + FIX 3: double-hard at each threshold
    hard_sets = {}
    for pct in TRIV_PCTS:
        keep_z = maxz[anom] <= np.percentile(maxz[nm], pct)
        keep_l = (lin[anom] <= np.percentile(lin[nm], pct)) if lin is not None else True
        h = anom[keep_z & keep_l]
        hard_sets[pct] = h
        rec[f"n_hard_p{pct:g}"] = int(len(h))
        rec[f"frac_trivial_p{pct:g}"] = round(1 - len(h) / len(anom), 4)
        rec[f"frac_trivial_maxz_only_p{pct:g}"] = round(1 - int(keep_z.sum()) / len(anom), 4)

    hard = hard_sets[PRIMARY_PCT]
    if len(hard) < min(MIN_HARDS):
        rec.update(status="dropped", reason=f"only_{len(hard)}_double_hard")
        meta.append(rec); continue

    norm_idx = np.where(nm)[0]
    nok = 0
    for seed in SEEDS:
        # THREE-WAY split of normals: train (fit) / val (label-free selection) / test (ground
        # truth). Hard anomalies all go to test. Val is normals-only, disjoint from test, so a
        # selector never sees the set it is judged on.
        g = np.random.default_rng(seed)
        idx = np.arange(len(norm_idx)); g.shuffle(idx)
        c1, c2 = int(0.6 * len(idx)), int(0.8 * len(idx))
        tr_i = norm_idx[idx[:c1]]; va_i = norm_idx[idx[c1:c2]]; te_i = norm_idx[idx[c2:]]
        if len(tr_i) < 50 or len(va_i) < 20 or len(te_i) < 20:
            continue
        max_h = int(MAX_TEST_RATE * len(te_i) / (1 - MAX_TEST_RATE))
        if max_h < min(MIN_HARDS):
            continue
        hard_s = (np.sort(np.random.default_rng(1000 + seed).choice(hard, max_h, replace=False))
                  if len(hard) > max_h else hard)
        rec["n_hard_used"] = int(len(hard_s))
        Xte = np.vstack([X[te_i], X[hard_s]])
        yte = np.r_[np.zeros(len(te_i)), np.ones(len(hard_s))]
        r2 = eval_dataset_3way(POOL, X[tr_i], X[va_i], Xte, yte,
                               ds.name, seed, "adbench_dami", NPZ_DIR,
                               extra=dict(n_train=len(tr_i), n_val=len(va_i), n_test_norm=len(te_i)))
        rows.extend(r2); nok += len(r2)
    # FIX 7 (audit): status was set unconditionally, so a dataset whose every variant failed
    # was recorded as "scored" with zero result rows (GlobalSharkAttacks did exactly this).
    rec.update(status="scored" if nok else "dropped",
               reason=None if nok else "no_variant_ran")
    meta.append(rec)
    if not nok:
        continue
    print(f"{ds.name}: dup={dup_frac:.2f} hard={len(hard)} "
          f"(triv {100*rec[f'frac_trivial_p{PRIMARY_PCT:g}']:.0f}%, "
          f"maxz-only {100*rec[f'frac_trivial_maxz_only_p{PRIMARY_PCT:g}']:.0f}%) "
          f"| {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(meta).to_csv(OUT.replace(".csv", "_steps.csv"), index=False)

D = pd.DataFrame(rows); D.to_csv(OUT, index=False)
M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
print(f"\nwrote {OUT} rows={len(D)}")

if len(D):
    sc = M0[M0.status == "scored"]
    print(f"\n=== FIX 4: duplicates removed, mean {sc.dup_frac.mean():.3f}, "
          f"max {sc.dup_frac.max():.3f} ({sc.nlargest(1,'dup_frac').dataset.iloc[0]}) ===")
    print(f"\n=== FIX 2: double-hard vs max|z|-only (trivial fraction removed) ===")
    for pct in TRIV_PCTS:
        print(f"  pct={pct:g}: double-hard removes {sc[f'frac_trivial_p{pct:g}'].mean():.3f}, "
              f"max|z| alone removes {sc[f'frac_trivial_maxz_only_p{pct:g}'].mean():.3f}")
    print(f"  LinRes available on {int(sc.linres_available.sum())}/{len(sc)} datasets")

    # inclusion sweep on the arm's own metrics (rough; the consolidator decides final
    # inclusion on ap_norm). Stability and is_classic were dropped in the 3-way round-2
    # refactor; final zones/spread live in hadb_consolidate.py.
    for metric in ["auc", "ap"]:
        gg = D.groupby(["dataset", "seed"])[metric].agg(["max", "mean"])
        M = gg.groupby("dataset").mean().rename(columns={"max": "best", "mean": "mean"})
        M["spread"] = M["best"] - M["mean"]
        M["zone"] = ["floor" if b < 0.70 else "live" for b in M["best"]]
        print(f"\n=== inclusion sweep on {metric.upper()} (arm-local, AUC/AP) ===")
        for thr in SPREADS:
            k = M[(M.zone == "live") & (M.spread >= thr)]
            print(f"  spread>={thr:.2f}: {len(k):3d}/{len(M)} datasets")
        if metric == "auc":
            M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
