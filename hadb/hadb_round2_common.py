# -*- coding: utf-8 -*-
"""Three-way evaluation of one (dataset, seed), shared by every arm so the partitions are
identical to the ones the metrics come from.

  TRAIN       normals only -> fit each detector.
  VALIDATION  normals only -> the label-free selectors compute their criteria here. No labels,
              no anomalies: strictly 'unsupervised, trained on normal data'. This is disjoint
              from both train and test, so a selector never sees the set it is judged on.
  TEST        held-out normals + all hard anomalies -> ground-truth AUC / AP / ap_norm.

Per (dataset, seed) it fits every candidate once, scores validation + test + uniform samples
over the validation box, computes the label-free criteria on VALIDATION, the ground-truth
metrics on TEST, saves the per-point score matrices (validation and test) to npz, and returns
one row per variant.

Row schema: dataset, corpus, seed, variant, auc, pauc10, ap, base_rate,   <- TEST (ground truth)
            em, mv, consensus, model_centrality, hits                       <- VALIDATION (selection)
"""
import os, io, sys, contextlib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from label_free_criteria import (consensus_scores, model_centrality_scores,
                                  hits_authority_scores, _em_auc, _mv_auc)

sys.path.insert(0, os.path.join(r"E:\Projects\Submitted\ADRank", "src"))
from adrank.pipeline import (embed as _embed, cluster as _cluster,  # noqa: E402
                             cluster_composite_scores as _ccs, sample_subsets as _subsets)

N_GEN = int(os.environ.get("HADB_R2_NGEN", "10000"))   # uniform samples for EM/MV on val box
NOMAS_ONLY = os.environ.get("HADB_NOMAS_ONLY", "0") == "1"
NOMAS_M = int(os.environ.get("HADB_NOMAS_M", "15"))    # pseudo-anomaly subsets to average


def nomas_scores(pool, Xval, seed, M=None, K=50):
    """NoMaS (Normal Manifold Separability): cluster the validation normals, hold out clusters
    as pseudo-anomalies, and rank each detector by mean pseudo-AUC separating held-out clusters
    from the rest. Returns {variant: mean pseudo-AUC}. Uses the SAME embed/cluster/subset
    machinery as the paper's pipeline, but over the HADB candidate pool passed in `pool`.
    Leak-free: operates only on validation normals; test is never touched."""
    M = M or NOMAS_M
    n = len(Xval)
    if n < 60:
        return {}
    try:
        Z = _embed(Xval, dim=16)
        labels = _cluster(Z, K=K, seed=seed)
        cs = _ccs(Z, labels)
        subs = _subsets(labels, cs, n_points=n, M=M, seed=seed)
    except Exception:
        return {}
    if not subs:
        return {}
    out = {}
    for vname, ctor in pool:
        aucs = []
        for sub in subs:
            if len(sub.pseudo_anom_idx) < 3 or len(sub.pseudo_norm_idx) < 3 or len(sub.train_idx) < 20:
                continue
            try:
                m = ctor()
                with contextlib.redirect_stdout(io.StringIO()):
                    m.fit(Xval[sub.train_idx])
                idx = np.concatenate([sub.pseudo_norm_idx, sub.pseudo_anom_idx])
                s = np.asarray(m.decision_function(Xval[idx]), float)
                y = np.r_[np.zeros(len(sub.pseudo_norm_idx)), np.ones(len(sub.pseudo_anom_idx))]
                if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                    aucs.append(roc_auc_score(y, s))
            except Exception:
                continue
        if aucs:
            out[vname] = float(np.mean(aucs))
    return out


def _fit_score(ctor, Xtr, Xval, Xte, U):
    """Fit on train; return (val_scores, test_scores, unif_scores) or Nones on failure."""
    try:
        m = ctor()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(Xtr)
        sv = np.asarray(m.decision_function(Xval), float)
        st = np.asarray(m.decision_function(Xte), float)
        su = np.asarray(m.decision_function(U), float)
        if not (np.all(np.isfinite(sv)) and np.all(np.isfinite(st))):
            return None, None, None
        if np.nanstd(st) < 1e-12 or np.nanstd(sv) < 1e-12:
            return None, None, None
        su = np.nan_to_num(su, nan=float(np.nanmedian(su[np.isfinite(su)])) if np.isfinite(su).any() else 0.0)
        return sv, st, su
    except Exception:
        return None, None, None


def eval_dataset_3way(pool, Xtr, Xval, Xte, yte, dataset, seed, corpus, npz_dir,
                      extra=None, n_gen=N_GEN, rng_seed=0):
    """Returns per-variant rows; saves validation & test score matrices to npz_dir."""
    # NoMaS-only fast path: compute just the NoMaS pick (needs Xval + pool), skip the
    # expensive EM/MV uniform sampling and the ground-truth scoring (already on disk). Emits
    # a minimal row per variant, merged into the full arm CSV by (dataset, seed, variant).
    if NOMAS_ONLY:
        nm = nomas_scores(pool, Xval, seed)
        rows = []
        for vname, val in nm.items():
            r = dict(dataset=dataset, corpus=corpus, seed=seed, variant=vname, nomas=val)
            if extra:
                r.update({k: v for k, v in extra.items() if k == "source"})
            rows.append(r)
        return rows

    rng = np.random.default_rng(rng_seed + seed)
    lo, hi = Xval.min(0), Xval.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    U = lo + rng.random((n_gen, Xval.shape[1])) * span      # uniform over the VALIDATION box

    names, Vcols, Tcols, Ucols, aucs, paucs, aps = [], [], [], [], [], [], []
    br = float(np.mean(yte))
    for vname, ctor in pool:
        sv, st, su = _fit_score(ctor, Xtr, Xval, Xte, U)
        if sv is None:
            continue
        names.append(vname); Vcols.append(sv); Tcols.append(st); Ucols.append(su)
        aucs.append(float(roc_auc_score(yte, st)))
        paucs.append(float(roc_auc_score(yte, st, max_fpr=0.10)))
        aps.append(float(average_precision_score(yte, st)))
    if len(names) < 3:
        return []

    V = np.column_stack(Vcols)          # [n_val, m]  validation scores -> selection
    T = np.column_stack(Tcols)          # [n_test, m] test scores -> ground truth
    cons = consensus_scores(V)
    mc = model_centrality_scores(V)
    hits = hits_authority_scores(V)
    t = np.linspace(0, 100, 1000); alpha = np.linspace(0.9, 0.999, 1000)
    em = np.empty(len(names)); mv = np.empty(len(names))
    for j in range(len(names)):
        sx = -V[:, j]; su = -Ucols[j]           # normality score on validation
        em[j] = _em_auc(t, 1.0, su, sx, n_gen)
        mv[j] = _mv_auc(alpha, 1.0, su, sx, n_gen)

    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(os.path.join(npz_dir, f"{corpus}__{dataset}__s{seed}.npz"),
                        V=V.astype(np.float32), T=T.astype(np.float32),
                        yte=yte.astype(np.int8), variants=np.array(names), base_rate=br)

    rows = []
    for j, vname in enumerate(names):
        r = dict(dataset=dataset, corpus=corpus, seed=seed, variant=vname,
                 auc=aucs[j], pauc10=paucs[j], ap=aps[j], base_rate=br,
                 em=float(em[j]), mv=float(mv[j]), consensus=float(cons[j]),
                 model_centrality=float(mc[j]), hits=float(hits[j]))
        if extra:
            r.update(extra)
        rows.append(r)
    return rows
