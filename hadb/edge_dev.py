# -*- coding: utf-8 -*-
"""DEV harness: iterate the edge/tail pseudo-anomaly selector on a tiny fixed set (2 tabular +
2 time-series) until it beats random and approaches EM, BEFORE any full run.

Per dev dataset it reconstructs the validation-normal features exactly as the arm does (seed 0),
looks up the frozen TEST ap_norm per detector, and evaluates several selectors live:
  edge_*   the user's idea in configurable variants; nomas = whole-cluster holdout (current);
  em / random from the frozen CSVs. Lower regret is better; 0 = picked the true best.
"""
import os, sys, io, contextlib, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(S, "scratchpad"))
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3, POOL as TS_POOL  # noqa: E402
from adrank.ts import _window_features, _window_labels  # noqa: E402
from hadb_round2_common import nomas_scores  # whole-cluster NoMaS  # noqa: E402

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PPCA
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA


def build_tab_pool():
    p = [("IForest", lambda: IForest(n_estimators=100, random_state=0)),
         ("LOF", lambda: LOF(n_neighbors=20)), ("KNN", lambda: KNN(n_neighbors=5)),
         ("ECOD", lambda: ECOD()), ("COPOD", lambda: COPOD()), ("HBOS", lambda: HBOS(n_bins=10)),
         ("PCA", lambda: PPCA(n_components=0.5, random_state=0)),
         ("CBLOF", lambda: CBLOF(n_clusters=8, random_state=0)), ("LODA", lambda: LODA(n_bins=10))]
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


TAB_POOL = build_tab_pool()


# ------------- reconstruct validation features (seed 0), exactly as the arms do -------------
def val_tabular(corpus, name):
    d = np.load(os.path.join(ROOT, "data", corpus, name + ".npz"), allow_pickle=True)
    X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(),
                        np.asarray(d["test_labels"]).ravel()]).astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    uq, first, inv = np.unique(X, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel(); lo = np.full(len(uq), 2, int); hi = np.full(len(uq), -1, int)
    np.minimum.at(lo, inv, y); np.maximum.at(hi, inv, y); mixed = lo != hi
    keep = np.zeros(len(X), bool); keep[first] = True; keep &= ~mixed[inv]
    X, y = X[keep], y[keep]
    if len(X) > 5000:
        r = np.random.RandomState(0); k = r.choice(len(X), 5000, replace=False); X, y = X[k], y[k]
    nm = np.where(y == 0)[0]
    g = np.random.default_rng(0); idx = np.arange(len(nm)); g.shuffle(idx)
    va = nm[idx[int(0.6 * len(idx)):int(0.8 * len(idx))]]
    return X[va]


def val_ucr(name):
    import zipfile, re
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "ucr", "UCR_TimeSeriesAnomalyDatasets2021.zip"))
    cand = [n for n in z.namelist() if os.path.basename(n) == name]
    if not cand:
        cand = [n for n in z.namelist() if n.lower().endswith(".txt") and name.split("_")[0] + "_" in os.path.basename(n)]
    fn = cand[0]
    m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn, re.I)
    x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()], float)
    a0, a1 = int(m.group(2)), int(m.group(3))
    lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2
        loo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[loo:loo + MAX_LEN], lab[loo:loo + MAX_LEN]
    starts = np.arange(0, len(x) - W + 1, STRIDE)
    Xw, _ = _window_features(x, w=W, stride=STRIDE)
    yw = _window_labels(lab, starts, w=W, min_count=1)
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)
    uniq, first, inv = np.unique(Xw, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel(); lo = np.full(len(uniq), 2, int); hi = np.full(len(uniq), -1, int)
    np.minimum.at(lo, inv, yw); np.maximum.at(hi, inv, yw); mixed = lo != hi
    keep = np.zeros(len(Xw), bool); keep[first] = True; keep &= ~mixed[inv]
    pos = np.where(keep)[0]; Xw, yw = Xw[keep], yw[keep]
    tr_i, va_i, te_i = block_split3(yw, pos, 0)
    return Xw[va_i]


# ------------------------------- the edge selector -------------------------------
def edge_scores(Xval, pool, q=0.01, k_clusters=1, use_pca=True, seed=0):
    """Cluster (k may be 1); take the top-q farthest-from-centre points as pseudo-anomalies;
    fit each detector on the core, rank by AUC separating the edge from held-out core."""
    Xs = StandardScaler().fit_transform(Xval)
    Z = (PCA(n_components=16, random_state=0).fit_transform(Xs)
         if use_pca and Xs.shape[1] > 16 else Xs)
    if k_clusters <= 1:
        lab = np.zeros(len(Z), int); cen = Z.mean(0, keepdims=True)
    else:
        km = MiniBatchKMeans(n_clusters=min(k_clusters, max(2, len(Z) // 20)),
                             random_state=seed, n_init=5).fit(Z)
        lab, cen = km.labels_, km.cluster_centers_
    dist = np.linalg.norm(Z - cen[lab], axis=1)
    n_edge = max(10, int(q * len(Z)))
    edge = np.argsort(-dist)[:n_edge]
    core = np.setdiff1d(np.arange(len(Z)), edge)
    g = np.random.default_rng(seed); c = core.copy(); g.shuffle(c)
    cut = int(0.7 * len(c)); tr, ho = c[:cut], c[cut:]
    if len(tr) < 30 or len(ho) < 10:
        return {}
    Xtr = Xval[tr]; Xev = np.vstack([Xval[ho], Xval[edge]])
    yev = np.r_[np.zeros(len(ho)), np.ones(len(edge))]
    out = {}
    for vname, ctor in pool:
        try:
            m = ctor()
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(Xtr)
            s = np.asarray(m.decision_function(Xev), float)
            if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                out[vname] = float(roc_auc_score(yev, s))
        except Exception:
            continue
    return out


def overcluster_scores(Xval, pool, K=60, size_q=0.33, embedded=False, M=15, seed=0):
    """OVERCLUSTER into many fine clusters, hold out SMALL ones as pseudo-anomalies (no
    far-from-others bias, so they can sit INSIDE/BETWEEN the manifold where the EDA found the
    real hard anomalies). `embedded`=True further restricts to small clusters whose centre is
    CLOSE to other centres (locally embedded, not peripheral)."""
    Xs = StandardScaler().fit_transform(Xval)
    Z = PCA(n_components=16, random_state=0).fit_transform(Xs) if Xs.shape[1] > 16 else Xs
    Keff = min(K, max(4, len(Z) // 15))
    km = MiniBatchKMeans(n_clusters=Keff, random_state=seed, n_init=5).fit(Z)
    lab, cen = km.labels_, km.cluster_centers_
    sizes = np.array([(lab == c).sum() for c in range(Keff)])
    small = np.where(sizes <= max(3, np.quantile(sizes, size_q)))[0]
    if embedded and len(small) > 2:
        nn = NearestNeighbors(n_neighbors=min(3, Keff - 1)).fit(cen)
        far = nn.kneighbors(cen)[0][:, 1:].mean(1)          # dist to nearest other centres
        emb = small[far[small] <= np.median(far)]           # keep the locally-embedded ones
        if len(emb) >= 1:                                    # fallback: keep all small if none embedded
            small = emb
    if len(small) < 1:
        return {}
    g = np.random.default_rng(seed); cand = list(small); g.shuffle(cand)
    all_idx = np.arange(len(Z))
    out = {v: [] for v, _ in pool}
    for c in cand[:M]:
        pa = all_idx[lab == c]
        if len(pa) < 3:
            continue
        rest = all_idx[lab != c]; g.shuffle(rest)
        ho = rest[:max(20, len(rest) // 5)]; tr = rest[len(ho):]
        if len(tr) < 30 or len(ho) < 10:
            continue
        Xtr = Xval[tr]; Xev = np.vstack([Xval[ho], Xval[pa]])
        yev = np.r_[np.zeros(len(ho)), np.ones(len(pa))]
        for vname, ctor in pool:
            try:
                m = ctor()
                with contextlib.redirect_stdout(io.StringIO()):
                    m.fit(Xtr)
                s = np.asarray(m.decision_function(Xev), float)
                if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                    out[vname].append(roc_auc_score(yev, s))
            except Exception:
                continue
    return {v: float(np.mean(a)) for v, a in out.items() if a}


def two_channel_scores(Xval, pool, K=60, size_q=0.33, M=12, seed=0):
    """Two pseudo-anomaly channels from the same overclustering:
      EDGE     small clusters that are PERIPHERAL (far from other centres) -> rewards global
               detectors (HBOS/COPOD/ECOD/IForest).
      EMBEDDED small clusters AMONG other clusters (close to other centres) -> rewards local
               detectors (LOF/KNN/CBLOF), matching where most real hard anomalies sit.
    Returns {variant: (edge_auc, embedded_auc)}; either may be nan if a channel is empty."""
    Xs = StandardScaler().fit_transform(Xval)
    Z = PCA(n_components=16, random_state=0).fit_transform(Xs) if Xs.shape[1] > 16 else Xs
    Keff = min(K, max(4, len(Z) // 15))
    km = MiniBatchKMeans(n_clusters=Keff, random_state=seed, n_init=5).fit(Z)
    lab, cen = km.labels_, km.cluster_centers_
    sizes = np.array([(lab == c).sum() for c in range(Keff)])
    small = np.where(sizes <= max(3, np.quantile(sizes, size_q)))[0]
    if len(small) < 2:
        return {}
    nn = NearestNeighbors(n_neighbors=min(3, Keff - 1)).fit(cen)
    far = nn.kneighbors(cen)[0][:, 1:].mean(1)
    med = np.median(far[small])
    edge_cl = small[far[small] > med]
    emb_cl = small[far[small] <= med]
    all_idx = np.arange(len(Z))
    g = np.random.default_rng(seed)

    def channel(clusters):
        out = {v: [] for v, _ in pool}
        cl = list(clusters); g.shuffle(cl)
        for c in cl[:M]:
            pa = all_idx[lab == c]
            if len(pa) < 3:
                continue
            rest = all_idx[lab != c]; gg = rest.copy(); g.shuffle(gg)
            ho = gg[:max(20, len(gg) // 5)]; tr = gg[len(ho):]
            if len(tr) < 30 or len(ho) < 10:
                continue
            Xtr = Xval[tr]; Xev = np.vstack([Xval[ho], Xval[pa]]); yev = np.r_[np.zeros(len(ho)), np.ones(len(pa))]
            for vname, ctor in pool:
                try:
                    m = ctor()
                    with contextlib.redirect_stdout(io.StringIO()):
                        m.fit(Xtr)
                    s = np.asarray(m.decision_function(Xev), float)
                    if np.all(np.isfinite(s)) and np.nanstd(s) > 1e-12:
                        out[vname].append(roc_auc_score(yev, s))
                except Exception:
                    continue
        return {v: float(np.mean(a)) if a else np.nan for v, a in out.items()}

    e = channel(edge_cl); m = channel(emb_cl)
    return {v: (e.get(v, np.nan), m.get(v, np.nan)) for v, _ in pool}


# ------------------------------- dev evaluation -------------------------------
# auto-sample N per corpus spanning the spread range (representative, not extreme)
def sample_dev(corpus, n=10):
    sub = _INC[_INC.corpus == corpus].sort_values("spread_ap_norm")
    idx = np.linspace(0, len(sub) - 1, n).astype(int)
    return list(sub.dataset.iloc[idx])

_INC = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
_INC = _INC[_INC.include]
DEV = [
    ("oddbench", "hadb_oddbench.csv", val_tabular, TAB_POOL, sample_dev("oddbench", 10)),
    ("ucr", "hadb_ts_ucr.csv", None, TS_POOL(), sample_dev("ucr", 10)),
]
CONFIGS = []  # edge configs dropped (shown to lose); overcluster only below


def regret_of(pick, ap, best):
    return best - ap.get(pick, np.nan)


# TWO-CHANNEL sweep: cache (edge_auc, embedded_auc) per detector per dataset, then sweep the
# mix weight w for final = w*embedded + (1-w)*edge, pick argmax. Compare to nomas/em/random.
WSWEEP = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]
cache = []          # list of (ap, best, tc_dict, em_pick_regret, random_regret)
for corpus, csv, loader, pool, names in DEV:
    D = pd.read_csv(os.path.join(S, csv)); D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
    apn = D.groupby(["dataset", "variant"]).ap_norm.mean()
    emv = D.groupby(["dataset", "variant"]).em.mean()
    for name in names:
        try:
            Xval = val_ucr(name) if corpus == "ucr" else loader(corpus, name)
        except Exception as e:
            print(f"  [skip {name[:28]}] {type(e).__name__}"); continue
        if len(Xval) < 60:
            continue
        ap = apn.loc[name]; best = ap.max()
        tc = two_channel_scores(Xval, pool)
        nm = nomas_scores(pool, Xval, 0)
        cache.append(dict(name=name, ap=ap, best=best, tc=tc,
                          em=regret_of(emv.loc[name].idxmax(), ap, best),
                          nomas=regret_of(max(nm, key=nm.get), ap, best) if nm else np.nan,
                          random=best - ap.mean()))
        e_ok = sum(1 for v in tc.values() if v[0] == v[0]); m_ok = sum(1 for v in tc.values() if v[1] == v[1])
        print(f"  {corpus:9s} {name[:30]:32s} best={best:.3f}  channels edge={e_ok} emb={m_ok}")
        cache[-1]["modality"] = "tabular" if corpus in ("oddbench", "ovrbench", "adbench_dami") else "timeseries"

# persist per-(dataset, model) two-channel scores for the scatter analysis
_pr = []
for r in cache:
    for v, (e, m) in r["tc"].items():
        _pr.append(dict(dataset=r["name"], modality=r.get("modality"), variant=v,
                        edge_auc=e, embedded_auc=m, true_ap_norm=float(r["ap"].get(v, np.nan))))
pd.DataFrame(_pr).to_csv(os.path.join(S, "HADB_TWO_CHANNEL.csv"), index=False)
print(f"  wrote HADB_TWO_CHANNEL.csv ({len(_pr)} model-dataset rows)")


def mix_regret(w):
    regs = []
    for r in cache:
        tc = r["tc"]
        scored = {v: w * m + (1 - w) * e for v, (e, m) in tc.items()
                  if (e == e or w == 1) and (m == m or w == 0)}
        # allow single-channel fallback when the other channel is nan
        scored = {v: (w * (m if m == m else 0) + (1 - w) * (e if e == e else 0))
                  for v, (e, m) in tc.items() if (e == e or m == m)}
        if not scored:
            continue
        pick = max(scored, key=scored.get)
        regs.append(r["best"] - r["ap"].get(pick, np.nan))
    return np.nanmean(regs)

print(f"\n=== TWO-CHANNEL mix sweep ({len(cache)} datasets; final=w*embedded+(1-w)*edge) ===")
em_reg = np.nanmean([r["em"] for r in cache])
for w in WSWEEP:
    mr = mix_regret(w)
    tag = {0.0: " (edge only)", 1.0: " (embedded only)"}.get(w, "")
    flag = "  <- BEATS EM" if mr < em_reg else ""
    print(f"  w={w:.2f}{tag:18s} {mr:.4f}{flag}")
print(f"  --- baselines ---")
print(f"  em     {em_reg:.4f}  (bar)")
print(f"  nomas  {np.nanmean([r['nomas'] for r in cache]):.4f}")
print(f"  random {np.nanmean([r['random'] for r in cache]):.4f}")
