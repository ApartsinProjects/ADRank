# -*- coding: utf-8 -*-
"""HADB multivariate time-series arm (TSB-AD-M, 200 series, Apache-2.0).

Same protocol as the univariate arm: unsupervised detectors fitted on NORMAL WINDOWS ONLY,
contiguous-block split with an overlap purge, Wu & Keogh triviality applied before ranking.
It imports those helpers from hadb_ts_final rather than restating them, so the label-shift
and window-overlap fixes cannot drift apart between the two arms.

WHAT IS DIFFERENT, AND WHY

  1. TRIVIALITY IS PER CHANNEL. Pinet et al. 2026 (arXiv:2606.02670) measured across eight
     public MTS benchmarks that no cross-channel rupture occurs without an accompanying
     UNIVARIATE deviation. So a series is trivial if the one-liner test fires on ANY single
     channel: the anomaly did not need the multivariate structure. What survives is the
     genuinely multivariate remainder, which is the only part that justifies an MTS arm.

  2. COMPACT PER-CHANNEL FEATURES. The univariate arm's 28 features per window would give
     55 x 28 = 1540 dimensions on the widest series, where nearest-neighbour distances
     concentrate and LOF/KNN stop measuring anything. Eight features per channel, and the
     widest MAX_CH channels by variance, keeps the joint space at most 160-dimensional.
     Cross-channel structure stays available to the detectors: the channels share one
     feature vector, so PCA/LOF/KNN see their covariation directly. No explicit pairwise
     correlation feature is computed, which would be O(C^2) per window and dominate runtime.

  3. ANOMALY-CENTRED TRUNCATION at MAX_LEN, as in the univariate arm: these series reach
     230k points, and truncating from the start deletes the labelled region outright.
"""
import os, sys, io, time, zipfile, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from hadb_ts_final import (W, STRIDE, MAX_LEN, MIN_WIN, MIN_NORM_WIN,  # noqa: E402
                           MIN_HARD, block_split3, score, POOL)
from ts_triviality import perm_trivial  # noqa: E402
from hadb_round2_common import eval_dataset_3way  # noqa: E402
NPZ_DIR = os.path.join(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d", "scores")

N_SERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MAX_CH = 20
TRIV_MAXLEN = 12000                 # anomaly-centred cap for the triviality test only
MAX_TEST_RATE = 0.25                # match the tabular arms; ap_norm divides by this
SEEDS = [0, 1, 2]
OUT = os.path.join(r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d",
                   "hadb_ts_mts.csv")
if os.environ.get("HADB_NOMAS_ONLY") == "1": OUT = OUT.replace(".csv", "_nomas.csv")
ZIP = os.path.join(ROOT, "data", "tsbad", "TSB-AD-M.zip")


def mts_window_features(Xc, w=W, stride=STRIDE):
    """Eight features per channel per window, concatenated. Fully vectorised."""
    n, C = Xc.shape
    starts = np.arange(0, n - w + 1, stride)
    blocks = []
    for c in range(C):
        sw = np.lib.stride_tricks.sliding_window_view(Xc[:, c], w)[::stride]
        mu, sd = sw.mean(1), sw.std(1)
        mn, mx = sw.min(1), sw.max(1)
        rms = np.sqrt((sw ** 2).mean(1))
        ad = np.abs(np.diff(sw, axis=1)).mean(1)
        a = sw - mu[:, None]
        ac1 = (a[:, :-1] * a[:, 1:]).mean(1) / (sd ** 2 + 1e-12)
        blocks.append(np.column_stack([mu, sd, mn, mx, mx - mn, rms, ad, ac1]))
    return np.hstack(blocks), starts


def window_labels(lab, starts, w=W):
    return np.array([int(lab[s:s + w].sum() >= 1) for s in starts], int)


def load_mts(n_max):
    z = zipfile.ZipFile(ZIP)
    fs = sorted(q for q in z.namelist() if q.lower().endswith(".csv"))
    fs = sorted(fs, key=lambda s: hashlib.sha1(s.encode()).hexdigest())[:n_max]
    for fn in fs:
        try:
            df = pd.read_csv(io.BytesIO(z.read(fn)))
        except Exception:
            continue
        lc = [q for q in df.columns if q.lower() in ("label", "is_anomaly", "anomaly")]
        if not lc:
            continue
        lab = df[lc[0]].to_numpy(int)
        Xc = df.drop(columns=lc).to_numpy(float)
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        src = os.path.basename(fn).split("_")[1] if "_" in os.path.basename(fn) else "?"
        yield os.path.basename(fn), src, Xc, lab


def main():
    rows, meta, t0 = [], [], time.time()
    pool = POOL()
    print(f"MTS arm: pool={len(pool)} variants, fit on NORMAL windows only, "
          f"max {MAX_CH} channels", flush=True)

    for i, (name, src, Xc, lab) in enumerate(load_mts(N_SERIES), 1):
        if len(Xc) > MAX_LEN:
            a = np.where(lab == 1)[0]
            if len(a):
                c = int((a[0] + a[-1]) // 2); half = MAX_LEN // 2
                lo = max(0, min(c - half, len(Xc) - MAX_LEN))
                Xc, lab = Xc[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
            else:
                Xc, lab = Xc[:MAX_LEN], lab[:MAX_LEN]
        n_ch_raw = Xc.shape[1]
        if n_ch_raw > MAX_CH:                       # widest channels by variance
            Xc = Xc[:, np.argsort(-Xc.std(0))[:MAX_CH]]

        rec = dict(dataset=name, corpus="tsbadm", source=src, n=len(Xc),
                   n_ch_raw=n_ch_raw, n_ch=Xc.shape[1], n_anom=int(lab.sum()),
                   anom_rate=round(float(lab.mean()), 5))

        # CALIBRATED triviality across all channels at once. perm_trivial takes the max AUC
        # over (channel x parameterisation) and compares it to a circular-shift null of the
        # SAME max statistic, so the multi-channel multiplicity is calibrated away and the
        # false-positive rate is alpha=5% by construction. The old "argmax lands in region on
        # any channel" test fired on 56% of RANDOM label placements and is withdrawn.
        Xt, lt = Xc, lab
        if len(Xc) > TRIV_MAXLEN:                       # anomaly-centred cap, test only
            a = np.where(lab == 1)[0]
            c0 = int((a[0] + a[-1]) // 2) if len(a) else TRIV_MAXLEN // 2
            lo = max(0, min(c0 - TRIV_MAXLEN // 2, len(Xc) - TRIV_MAXLEN))
            Xt, lt = Xc[lo:lo + TRIV_MAXLEN], lab[lo:lo + TRIV_MAXLEN]
        triv, how, stat, thr = perm_trivial(Xt, lt, K=99, seed=0)
        rec.update(keogh_trivial=bool(triv), keogh_rule=how,
                   triv_stat=round(float(stat), 4) if stat == stat else None,
                   triv_thr=round(float(thr), 4) if thr == thr else None)
        if triv:
            rec.update(status="dropped", reason="trivial_calibrated")
            meta.append(rec); continue

        try:
            Xw, starts = mts_window_features(Xc)
            yw = window_labels(lab, starts)
        except Exception:
            rec.update(status="dropped", reason="window"); meta.append(rec); continue
        Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)

        uniq, first_idx, inv = np.unique(Xw, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).ravel()
        lo_, hi_ = np.full(len(uniq), 2, int), np.full(len(uniq), -1, int)
        np.minimum.at(lo_, inv, yw); np.maximum.at(hi_, inv, yw)
        mixed = lo_ != hi_
        keep = np.zeros(len(Xw), bool); keep[first_idx] = True
        keep &= ~mixed[inv]
        pos = np.where(keep)[0]
        Xw, yw = Xw[keep], yw[keep]

        n_norm = int((yw == 0).sum())
        rec.update(n_win=len(Xw), n_feat=Xw.shape[1], n_anom_win=int(yw.sum()),
                   n_norm_win=n_norm, n_mixed_dup_groups=int(mixed.sum()))
        if len(Xw) < MIN_WIN or yw.sum() < MIN_HARD or n_norm < MIN_NORM_WIN:
            rec.update(status="dropped", reason="too_short_or_dense")
            meta.append(rec); continue

        anom_i = np.where(yw == 1)[0]
        nok = 0
        for seed in SEEDS:
            # THREE-WAY split: train (fit) / val (label-free selection, normal windows only) /
            # test (ground truth = held-out normal windows + anomaly windows), all disjoint in
            # the raw series via the overlap purge.
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
                                   name, seed, "tsbadm", NPZ_DIR,
                                   extra=dict(source=src, n_train=len(tr_i),
                                              n_val=len(va_i), n_test_norm=len(te_i)))
            rows.extend(r2); nok += len(r2)
        rec.update(status="scored" if nok else "dropped",
                   reason=None if nok else "no_variant_ran")
        meta.append(rec)
        if i % 5 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            pd.DataFrame(meta).to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
            sc = sum(1 for m in meta if m.get("status") == "scored")
            tv = sum(1 for m in meta if m.get("keogh_trivial"))
            print(f"[{i}] scored={sc} univariate_trivial={tv} | {time.time()-t0:.0f}s",
                  flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    M0 = pd.DataFrame(meta); M0.to_csv(OUT.replace(".csv", "_steps.csv"), index=False)
    D = pd.DataFrame(rows)
    print(f"\nwrote {OUT} rows={len(D)} series={D.dataset.nunique() if len(D) else 0}")
    if len(M0):
        tv = M0.keogh_trivial.fillna(False)
        print(f"  solvable by a SINGLE-CHANNEL one-liner: {int(tv.sum())}/{len(M0)} "
              f"({100*tv.mean():.0f}%)  <- not genuinely multivariate")
        print(f"  drops: {M0[M0.status=='dropped'].reason.value_counts().to_dict()}")
    if len(D):
        D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
        M = D.groupby("dataset").ap_norm.agg(["max", "mean"])
        M["spread"] = M["max"] - M["mean"]
        M["best_auc"] = D.groupby("dataset").auc.max()
        M.reset_index().to_csv(OUT.replace(".csv", "_manifest.csv"), index=False)
        for thr in (0.05, 0.10, 0.15):
            print(f"  spread>={thr:.2f}: {int((M.spread >= thr).sum())}/{len(M)}")


if __name__ == "__main__":
    main()
