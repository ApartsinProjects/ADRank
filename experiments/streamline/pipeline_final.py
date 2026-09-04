# -*- coding: utf-8 -*-
"""DEFINITIVE streamlined pipeline (two-stage linear hardening locked in). From source, detector-free.
  Stage 1  drop dataset if hardening rule catches >= TRIV1 of anomalies (OR-solvable)
  Stage 2a harden: original-feature OR (tail + interior-gap histogram) at 5% FP
  Stage 2b harden: PCA-whitened OR (tail + interior-gap histogram) at 5% FP  [union with 2a]
  Stage 3  keep >= MIN_HARD distinct hard anomalies (n_eff)
  Stage 4  dedup by data fingerprint
  Stage 5  keep >= MIN_NORM normals (held-out) to model
Writes streamline/ only."""
import os, sys, io, zipfile, re, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"; S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "scratchpad", "streamline")
sys.path.insert(0, os.path.join(S, "scratchpad")); sys.path.insert(0, os.path.join(ROOT, "src"))
import adrank.pipeline as PP
from hadb_ts_final import W, STRIDE, MAX_LEN, block_split3
from adrank.ts import _window_features, _window_labels
from hadb_ts_mts import mts_window_features, window_labels as mts_wlabels, load_mts
CSVMAP = {"oddbench": "hadb_oddbench.csv", "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv", "tsbad_u": "hadb_ts_tsbad.csv"}
ZIPS = {"ucr": "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip", "tsbad_u": "tsbad/TSB-AD-U.zip"}
Q = 0.05; TRIV1 = 0.90; MIN_HARD = 100; MIN_NORM = 800; MAX_MTS = 200
_Z = {}
def zf(p):
    if p not in _Z: _Z[p] = zipfile.ZipFile(os.path.join(ROOT, "data", p))
    return _Z[p]
OBJ = {}
for sub in ("adbench", "dami"):
    dd = os.path.join(ROOT, "data", sub)
    if os.path.isdir(dd):
        for ds in PP.load_npz_dir(dd): OBJ[(sub, str(ds.name)[:40])] = (np.nan_to_num(np.asarray(ds.X, float)), np.asarray(ds.y, int).ravel())
MTS = {}
for name, src, Xc, lab in load_mts(MAX_MTS): MTS[str(name)[:40]] = (Xc, lab)
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b] + 1e-9)))
    return sev
def two_stage_trivial(Xtr, Xhold, Xq):
    """return boolean: is each row of Xq trivial under stage2a (orig) OR stage2b (PC), each @5% FP."""
    so_h, so_q = severity(Xtr, Xhold), severity(Xtr, Xq); tho = np.quantile(so_h, 1 - Q)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ph, Pq, Ptr = pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xq)), pca.transform(sc.transform(Xtr))
        sp_h, sp_q = severity(Ptr, Ph), severity(Ptr, Pq); thp = np.quantile(sp_h, 1 - Q)
    except Exception:
        sp_q = np.zeros(len(Xq)); thp = np.inf
    return (so_q > tho) | (sp_q > thp)
def n_eff(Xtr, Xh):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd; Zh = (Xh - mu) / sd
    rng = np.random.default_rng(0); sub = Zt[rng.choice(len(Zt), min(len(Zt), 1500), replace=False)]
    r = np.median(NearestNeighbors(n_neighbors=2).fit(sub).kneighbors(sub)[0][:, 1]) + 1e-9
    if len(Zh) > 600: Zh = Zh[rng.choice(len(Zh), 600, replace=False)]
    nn = NearestNeighbors(radius=r).fit(Zh); cov = np.zeros(len(Zh), bool); c = 0
    for i in range(len(Zh)):
        if cov[i]: continue
        c += 1; cov[nn.radius_neighbors(Zh[i:i + 1], return_distance=False)[0]] = True
    return c
def fingerprint(Xtr, Xh_norm):
    Z = np.vstack([Xtr, Xh_norm]); return "%d|%s|%s" % (Z.shape[1], hashlib.sha1(np.round(np.sort(Z.mean(0)), 3).tobytes()).hexdigest()[:10], hashlib.sha1(np.round(np.sort(Z.std(0)), 3).tobytes()).hexdigest()[:10])
def sources():
    for corp, f in CSVMAP.items():
        for name in sorted(pd.read_csv(os.path.join(S, f)).dataset.unique()): yield corp, name, "tab" if corp in ("oddbench", "ovrbench") else "uni"
    for k in OBJ: yield k[0], k[1], "obj"
    for name in MTS: yield "tsbad_m", name, "mts"
def get_split(corp, name, how):
    if how == "tab":
        d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
        X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
        y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
        return _tab_split(X[y == 0], X[y == 1])
    if how == "obj":
        X, y = OBJ[(corp, name)]; X = np.nan_to_num(X); return _tab_split(X[y == 0], X[y == 1])
    if how == "mts":
        Xc, lab = MTS[name]
        if len(Xc) < W + 10: return None
        Xw, st = mts_window_features(Xc); yw = mts_wlabels(lab, st); Xw = np.nan_to_num(Xw)
        pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
        return Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[yw == 1]
    z = zf(ZIPS[corp]); cand = [q for q in z.namelist() if os.path.basename(q) == name] or [q for q in z.namelist() if q.lower().endswith((".txt", ".csv")) and name.split("_")[0] + "_" in os.path.basename(q)]
    fn = cand[0]
    if fn.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(z.read(fn))); x = np.nan_to_num(df.iloc[:, 0].to_numpy(float)); lab = df.iloc[:, 1].to_numpy(int)
    else:
        m = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", fn); x = np.array([float(v) for v in z.read(fn).decode("utf-8", "replace").split() if v.strip()]); a0, a1 = int(m.group(2)), int(m.group(3)); lab = np.zeros(len(x), int); lab[a0:min(a1 + 1, len(x))] = 1
    if len(x) > MAX_LEN:
        a = np.where(lab == 1)[0]; c = int((a[0] + a[-1]) // 2) if len(a) else MAX_LEN // 2; lo = max(0, min(c - MAX_LEN // 2, len(x) - MAX_LEN)); x, lab = x[lo:lo + MAX_LEN], lab[lo:lo + MAX_LEN]
    Xw, _ = _window_features(x, w=W, stride=STRIDE); st = np.arange(0, len(x) - W + 1, STRIDE); yw = _window_labels(lab, st, w=W, min_count=1); Xw = np.nan_to_num(Xw)
    pos = np.arange(len(Xw)); tr, va, te = block_split3(yw, pos, 0)
    return Xw[tr][yw[tr] == 0], Xw[te][yw[te] == 0], Xw[yw == 1]
def _tab_split(Xn, Xa):
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); k = int(0.6 * len(idx)); return Xn[idx[:k]], Xn[idx[k:]], Xa
rows = []
for corp, name, how in sources():
    try:
        sp = get_split(corp, name, how)
        if sp is None: continue
        Xtr, Xhold, Xa = sp
    except Exception: continue
    if len(Xtr) < 40 or len(Xhold) < 20 or len(Xa) < 5: continue
    triv = two_stage_trivial(Xtr, Xhold, Xa); hard = Xa[~triv]
    ne = n_eff(Xtr, hard) if len(hard) >= 5 else len(hard)
    rows.append({"corpus": corp, "dataset": str(name)[:40], "n_norm": len(Xhold), "n_anom": len(Xa),
                 "n_hard": int((~triv).sum()), "frac_triv": float(triv.mean()), "n_eff": int(ne),
                 "fp": fingerprint(Xtr, Xhold)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "STREAM_FINAL2_ALL.csv"), index=False)
s1 = df[df.frac_triv < TRIV1]; s5 = s1[s1.n_norm >= MIN_NORM]; s3 = s5[s5.n_eff >= MIN_HARD]
s4 = s3.sort_values("n_eff", ascending=False).drop_duplicates("fp"); s4.to_csv(os.path.join(OUT, "STREAM_FINAL2_SET.csv"), index=False)
print(f"=== DEFINITIVE pipeline (two-stage linear hardening) - {len(df)} loaded candidates ===")
print(f"  S1 keep frac_triv<{TRIV1}:   {len(s1)}")
print(f"  S5 keep n_norm>={MIN_NORM}:      {len(s5)}")
print(f"  S3 keep n_eff>={MIN_HARD}:       {len(s3)}")
print(f"  S4 dedup:                {len(s4)}   <- FINAL")
print(f"  FINAL by corpus: {s4.corpus.value_counts().to_dict()}")
print(f"  (previous single-stage-hardening final was 184; two-stage is stricter)")
print("saved streamline/STREAM_FINAL2_ALL.csv, STREAM_FINAL2_SET.csv")
