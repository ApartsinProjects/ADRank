# -*- coding: utf-8 -*-
"""HADB time-series arm, SCOPE-COMPLIANT: unsupervised detectors TRAINED ONLY ON NORMAL DATA.

WHY THIS REPLACES THE TSB-AD-API VERSION:
TSB-AD's `run_Unsupervise_AD(model, data)` takes a SINGLE array and fits on all of it, so
the detector sees anomalies during fitting - transductive, and structurally incapable of
"trained only on normal data". Its `run_Semisupervise_AD` has the right signature but its
CPU-friendly entries (MCD, OCSVM, Sub_MCD, Sub_OCSVM, AutoEncoder) raise
"Model function 'run_X' is not defined", leaving only deep GPU models.

So the correct construction for this scope is: window the series, fit PyOD detectors on
NORMAL WINDOWS ONLY, score held-out normal windows + anomaly windows. Identical protocol to
the tabular arm, CPU-only, and the full PyOD pool with hyperparameter sweeps is available.

SOURCES (both handled here; formats differ):
  TSB-AD-U : CSV, columns [value, Label], per-point labels.
  UCR      : .txt, ONE value per line, labels encoded in the FILENAME
             (UCR_Anomaly_<name>_<trainEnd>_<anomStart>_<anomEnd>). Built by Wu & Keogh to
             RESIST the one-liner test, with realistic anomaly density (~0.2%).
             LICENSE: none in the archive - fetch-only, never redistribute.

TRIVIALITY: Wu & Keogh Definition 1, at the SERIES level (their construction).
"""
import os, re, sys, io, zipfile, hashlib, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
sys.path.insert(0, os.path.join(ROOT, "src"))
from adrank.ts import _window_features, _window_labels  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hadb_round2_common import eval_dataset_3way  # noqa: E402

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "tsbad"      # tsbad | ucr
N_SERIES = int(sys.argv[2]) if len(sys.argv) > 2 else 120
SEEDS = [0, 1, 2]
W, STRIDE, MAX_LEN = 64, 16, 40000
MIN_WIN, MIN_NORM_WIN = 200, 150
# Consecutive windows share W - STRIDE = 48 of 64 raw points, so a RANDOM window-level split
# leaks: an audit measured 100% of test windows sharing raw data with some training window
# (mean 1.0000, min 1.0000 over 40 UCR series). The leak is not uniform either - it favours
# memorisation-style detectors (KNN, small-k LOF) over global ones (PCA, HBOS), so it biases
# the ground-truth RANKING that is this benchmark's actual product. Split by CONTIGUOUS
# BLOCKS instead, then purge training windows overlapping any evaluation window.
N_BLOCKS = 10
OVERLAP_R = W // STRIDE - 1          # windows within this many positions share raw points
# UCR carries EXACTLY ONE anomaly region per series (Wu & Keogh's fix for the
# unrealistic-density flaw), smeared over overlapping windows: median 7 anomaly windows,
# effective EVENT count 1. A 20-window floor discarded 75/250 usable series for a property
# that is by design. Power comes from many datasets, not many anomalies per dataset.
MIN_HARD = int(os.environ.get("HADB_MIN_HARD", "20"))
MAX_TEST_RATE = 0.25               # cap the eval base rate; ap_norm divides by it
OUT = os.path.join(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d",
                   f"hadb_ts_{SOURCE}.csv")
if os.environ.get("HADB_NOMAS_ONLY") == "1": OUT = OUT.replace(".csv", "_nomas.csv")
NPZ_DIR = os.path.join(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d",
                       "scores")


def POOL():
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
         ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()),
         ("HBOS", lambda: HBOS(n_bins=10)),
         ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0))]
    for k in (3, 10, 20, 50, 100):
        p.append((f"LOF_k{k}", lambda k=k: LOF(n_neighbors=k)))
        p.append((f"KNN_k{k}", lambda k=k: KNN(n_neighbors=k)))
    for nb in (5, 20, 50):
        p.append((f"HBOS_b{nb}", lambda nb=nb: HBOS(n_bins=nb)))
    for nc in (0.3, 0.9):
        p.append((f"PCA_c{nc}", lambda nc=nc: PPCA(n_components=nc, random_state=0)))
    for ne in (50, 300):
        p.append((f"IF_n{ne}", lambda ne=ne: IForest(n_estimators=ne, random_state=0)))
    return p


def _movmean(a, k):
    c = np.cumsum(np.insert(a, 0, 0.0)); out = np.empty(len(a)); h = k // 2
    for i in range(len(a)):
        lo, hi = max(0, i - h), min(len(a), i + h + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def keogh_trivial(x, lab):
    """Wu & Keogh Definition 1: solvable by a single line of primitive operations."""
    if lab.sum() == 0 or len(x) < 50:
        return True, "degenerate"
    d = np.diff(x, prepend=x[0])
    for base, tag in ((np.abs(d), "absdiff"), (d, "diff")):
        for k in (0, 8, 32, 128, 512):
            for u in (0, 1):
                if u and k == 0:
                    continue
                for c in (0.0, 1.0, 3.0):
                    if k == 0 and c > 0:
                        continue
                    thr = np.zeros(len(base))
                    if u:
                        thr = thr + _movmean(base, k)
                    if c and k:
                        m = _movmean(base, k); m2 = _movmean(base * base, k)
                        thr = thr + c * np.sqrt(np.maximum(m2 - m * m, 0.0))
                    st = base - thr
                    if np.all(np.isfinite(st)) and lab[int(np.nanargmax(st))] == 1:
                        return True, f"{tag}_u{u}_k{k}_c{c:g}"
    return False, "survived"


def block_split(yw, pos, seed, train_frac=0.8):
    """Contiguous-block train/test split over windows, with an overlap purge.

    Anomaly windows are ALWAYS evaluation-side: the protocol never fits on them. Normal
    windows are assigned by contiguous block, then every training window sharing raw points
    with an evaluation window is dropped, so train and test are disjoint in the RAW SERIES
    rather than merely in window indices.

    `pos` carries each surviving window's ORIGINAL window index, because the duplicate-row
    filter above leaves gaps: array-adjacency no longer implies time-adjacency, so overlap
    is measured in true position space. Returns (train_normal_idx, test_normal_idx).
    """
    n = len(yw)
    g = np.random.default_rng(seed)
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    order = g.permutation(N_BLOCKS)
    train_blocks = set(order[:max(1, int(round(train_frac * N_BLOCKS)))].tolist())

    in_train_block = np.zeros(n, bool)
    for b in train_blocks:
        in_train_block[edges[b]:edges[b + 1]] = True

    is_eval = (yw == 1) | (~in_train_block)
    # a window overlaps an evaluation window iff some eval position lies within OVERLAP_R;
    # pos is sorted, so searchsorted answers this exactly and in O(n log n)
    ev_pos = pos[is_eval]
    if len(ev_pos):
        lo = np.searchsorted(ev_pos, pos - OVERLAP_R, side="left")
        hi = np.searchsorted(ev_pos, pos + OVERLAP_R, side="right")
        near_eval = hi > lo
    else:
        near_eval = np.zeros(n, bool)
    tr_i = np.where((yw == 0) & in_train_block & ~near_eval)[0]
    te_i = np.where((yw == 0) & ~in_train_block)[0]
    return tr_i, te_i


def _near(pos, ref_pos):
    """Boolean mask: window whose raw span (within OVERLAP_R positions) touches any ref window."""
    if not len(ref_pos):
        return np.zeros(len(pos), bool)
    r = np.sort(ref_pos)
    lo = np.searchsorted(r, pos - OVERLAP_R, side="left")
    hi = np.searchsorted(r, pos + OVERLAP_R, side="right")
    return hi > lo


def block_split3(yw, pos, seed, fracs=(0.6, 0.2, 0.2)):
    """Three-way contiguous-block split of NORMAL windows into (train, val, test), mutually
    disjoint in the raw series. Anomaly windows are handled by the caller (all test-side).

    Disjointness by raw points, enforced with the same overlap purge as block_split:
      test  is the reference; kept as-is.
      val   drops any window overlapping a test window (selector must not see test).
      train drops any window overlapping a val OR test window (detectors fit unseen data).
    Returns (train_normal_idx, val_normal_idx, test_normal_idx)."""
    n = len(yw)
    g = np.random.default_rng(seed)
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    order = list(g.permutation(N_BLOCKS))
    n_tr = max(1, int(round(fracs[0] * N_BLOCKS)))
    n_va = max(1, int(round(fracs[1] * N_BLOCKS)))
    tr_b, va_b = set(order[:n_tr]), set(order[n_tr:n_tr + n_va])
    te_b = set(order[n_tr + n_va:]) or {order[-1]}

    def in_blocks(bs):
        m = np.zeros(n, bool)
        for b in bs:
            m[edges[b]:edges[b + 1]] = True
        return m
    in_tr, in_va, in_te = in_blocks(tr_b), in_blocks(va_b), in_blocks(te_b)
    norm = yw == 0
    # anomaly windows count as test-side references for the purge
    test_ref = pos[(yw == 1) | (norm & in_te)]
    val_ref = pos[norm & in_va]
    te_i = np.where(norm & in_te)[0]
    va_i = np.where(norm & in_va & ~_near(pos, test_ref))[0]
    tr_i = np.where(norm & in_tr & ~_near(pos, np.r_[test_ref, val_ref]))[0]
    return tr_i, va_i, te_i


def load_series():
    """Yield (name, source_tag, x, labels)."""
    if SOURCE == "tsbad":
        z = zipfile.ZipFile(os.path.join(ROOT, "data", "tsbad", "TSB-AD-U.zip"))
        fs = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
        fs = sorted(fs, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:N_SERIES]
        for fn in fs:
            try:
                df = pd.read_csv(io.BytesIO(z.read(fn)))
                x = np.nan_to_num(df.iloc[:, 0].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
                yield os.path.basename(fn), os.path.basename(fn).split("_")[1], x, \
                    df.iloc[:, 1].to_numpy(int)
            except Exception:
                continue
    else:
        z = zipfile.ZipFile(os.path.join(ROOT, "data", "ucr",
                                         "UCR_TimeSeriesAnomalyDatasets2021.zip"))
        pat = re.compile(r"_(\d+)_(\d+)_(\d+)\.txt$", re.I)
        fs = [n for n in z.namelist() if pat.search(n)]
        fs = sorted(fs, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:N_SERIES]
        for fn in fs:
            m = pat.search(fn)
            try:
                raw = z.read(fn).decode("utf-8", "replace").split()
                x = np.array([float(v) for v in raw if v.strip()], float)
            except Exception:
                continue
            a0, a1 = int(m.group(2)), int(m.group(3))
            lab = np.zeros(len(x), int)
            lab[a0:min(a1 + 1, len(x))] = 1
            yield os.path.basename(fn), "UCR", x, lab


def score(ctor, Xtr, Xte):
    import contextlib
    try:
        m = ctor()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(Xtr)                      # <-- NORMAL WINDOWS ONLY
        s = np.asarray(m.decision_function(Xte), float)
        return s if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12 else None
    except Exception:
        return None


def main():
    """Run one arm. Guarded so the module can be imported for its helpers
    (block_split, keogh_trivial, the windowing constants) WITHOUT executing a full
    scoring run - importing it unguarded silently launched a 3-minute UCR arm."""
    rows, meta, t0 = [], [], time.time()
    pool = POOL()
    print(f"source={SOURCE} pool={len(pool)} variants (all fit on NORMAL windows only)", flush=True)

    for i, (name, src, x, lab) in enumerate(load_series(), 1):
        if len(x) > MAX_LEN:
            # BUG FIX: truncating from the START deleted the anomaly on 33% of UCR series
            # (75th pct of anomaly-start is 54,600; MAX_LEN was 40,000), which then registered
            # as "degenerate" and was miscounted as trivial. Keep a window that CONTAINS the
            # anomaly: take the training head plus a span centred on the labelled region.
            a = np.where(lab == 1)[0]
            if len(a):
                c = int((a[0] + a[-1]) // 2)
                half = MAX_LEN // 2
                lo = max(0, min(c - half, len(x) - MAX_LEN))
                x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
            else:
                x, lab = x[:MAX_LEN], lab[:MAX_LEN]
        rec = dict(dataset=name, corpus=SOURCE, source=src, n=len(x), n_anom=int(lab.sum()),
                   anom_rate=round(float(lab.mean()), 5))
        triv, how = keogh_trivial(x, lab)
        rec.update(keogh_trivial=bool(triv), keogh_rule=how)
        if triv:
            rec.update(status="dropped", reason="keogh_trivial"); meta.append(rec)
        else:
            try:
                # BUG FIX: _window_features returns starts + w//2 (window CENTERS), while
                # _window_labels slices labels[s:s+w] expecting a START. Passing centers shifted
                # every label by w/2: 12.1% of "anomaly" windows held zero anomalous points, and
                # 171 windows that DID contain the anomaly were labelled normal and fitted on,
                # violating normals-only training on 96 of 96 series. Recompute real starts, the
                # way the library's own _make_ts_dataset does.
                starts = np.arange(0, len(x) - W + 1, STRIDE)
                Xw, _ = _window_features(x, w=W, stride=STRIDE)
                yw = _window_labels(lab, starts, w=W, min_count=1)
                if len(Xw) != len(yw):
                    rec.update(status="dropped", reason="window_align"); meta.append(rec); continue
            except Exception:
                rec.update(status="dropped", reason="window"); meta.append(rec); continue
            Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
            # BUG FIX: keeping the first occurrence of a duplicate feature row silently assigns
            # whichever label came first when identical rows carry both. Drop those groups: they
            # are undetectable in principle, and being indistinguishable from a normal row they
            # pass the triviality filter by construction, inflating the hard set with noise.
            uniq, first_idx, inv = np.unique(Xw, axis=0, return_index=True, return_inverse=True)
            inv = np.asarray(inv).ravel()
            lab_min = np.full(len(uniq), 2, int); lab_max = np.full(len(uniq), -1, int)
            np.minimum.at(lab_min, inv, yw); np.maximum.at(lab_max, inv, yw)
            mixed_grp = lab_min != lab_max
            keep = np.zeros(len(Xw), bool); keep[first_idx] = True
            keep &= ~mixed_grp[inv]
            pos = np.where(keep)[0]                 # ORIGINAL window index of each survivor
            Xw, yw = Xw[keep], yw[keep]
            n_norm = int((yw == 0).sum())
            rec.update(n_win=len(Xw), n_anom_win=int(yw.sum()), n_norm_win=n_norm,
                       n_mixed_dup_groups=int(mixed_grp.sum()))
            if len(Xw) < MIN_WIN or yw.sum() < MIN_HARD or n_norm < MIN_NORM_WIN:
                rec.update(status="dropped", reason="too_short_or_dense"); meta.append(rec); continue
            norm_i, anom_i = np.where(yw == 0)[0], np.where(yw == 1)[0]
            nok = 0
            for seed in SEEDS:
                # THREE-WAY split: train fits detectors, val is where selectors compute their
                # label-free criteria (normals only), test carries held-out normals + hard
                # anomalies for the ground truth. All three disjoint in the raw series.
                tr_i, va_i, te_i = block_split3(yw, pos, seed)
                if len(tr_i) < 50 or len(va_i) < 20 or len(te_i) < 20:
                    continue
                max_a = int(MAX_TEST_RATE * len(te_i) / (1 - MAX_TEST_RATE))
                if max_a < MIN_HARD:
                    continue
                anom_s = (np.sort(np.random.default_rng(1000 + seed).choice(anom_i, max_a, replace=False))
                          if len(anom_i) > max_a else anom_i)
                ev = np.r_[te_i, anom_s]
                yte = np.r_[np.zeros(len(te_i)), np.ones(len(anom_s))]
                r2 = eval_dataset_3way(pool, Xw[tr_i], Xw[va_i], Xw[ev], yte,
                                       name, seed, SOURCE, NPZ_DIR,
                                       extra=dict(source=src, n_train=len(tr_i),
                                                  n_val=len(va_i), n_test_norm=len(te_i)))
                rows.extend(r2); nok += len(r2)
            rec.update(status="scored" if nok else "dropped",
                       reason=None if nok else "no_variant_ran")
            meta.append(rec)
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            pd.DataFrame(meta).to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
            sc = sum(1 for m in meta if m.get("status") == "scored")
            tv = sum(1 for m in meta if m.get("keogh_trivial"))
            print(f"[{i}] scored={sc} keogh_trivial={tv} | {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
    D = pd.DataFrame(rows)
    print(f"\nwrote {OUT} rows={len(D)} series={D.dataset.nunique() if len(D) else 0}")
    if len(M0):
        tv = M0.keogh_trivial.fillna(False)
        print(f"  Wu-Keogh TRIVIAL: {int(tv.sum())}/{len(M0)} ({100*tv.mean():.0f}%)")
        print(f"  drops: {M0[M0.status=='dropped'].reason.value_counts().to_dict()}")
        print(f"  median anomaly rate: {M0.anom_rate.median():.5f}")
    if len(D):
        D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
        M = D.groupby("dataset").ap_norm.agg(["max", "mean"])
        M["spread"] = M["max"] - M["mean"]
        au = D.groupby("dataset").auc.max()
        zone = [("ceiling" if (au[i] >= 0.95 and s < 0.03) else "floor" if au[i] < 0.70 else "live")
                for i, s in zip(M.index, M.spread)]
        M["zone"] = zone
        print(f"  zones: {pd.Series(zone).value_counts().to_dict()}")
        for thr in (0.05, 0.10, 0.15):
            print(f"  live AND spread>={thr:.2f}: {len(M[(M.zone=='live')&(M.spread>=thr)])}/{len(M)}")
        M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)


if __name__ == "__main__":
    main()
