"""B4 reconciliation: does the marginal-tail regime help IN-DISTRIBUTION (DAMI +
ADBench tabular), or does it contradict the DAMI global-tail negative in Sec 5.7?

Runs the SAME marginal-tail regime as scripts/modal_marginal.py on the DAMI (10)
and ADBench Classical tabular (26) datasets, reporting cluster-only vs +marginal
vs gated regret. If it helps or is safely gated off in-distribution too, B4 can
go in the paper; if it hurts the easy datasets, it stays diagnostics.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
)
vol = modal.Volume.from_name("adrank-reconcile-results", create_if_missing=True)
app = modal.App("adrank-reconcile")
KS, SELS, M, MTAIL_FRAC = [30, 50], ["smallest", "random", "hard"], 20, 0.05
DAMI = ["Annthyroid", "Cardiotocography", "PageBlocks", "Pima", "Stamps", "WBC",
        "WDBC", "Waveform", "Wilt", "InternetAds"]
ADB = ["6_cardio", "14_glass", "18_Ionosphere", "19_landsat", "20_letter", "22_magic.gamma",
       "23_mammography", "25_musk", "26_optdigits", "27_PageBlocks", "28_pendigits", "29_Pima",
       "30_satellite", "31_satimage-2", "35_SpamBase", "36_speech", "37_Stamps", "38_thyroid",
       "39_vertebral", "40_vowels", "41_Waveform", "42_WBC", "43_WDBC", "44_Wilt", "47_yeast"]


def _load_adb(name):
    import io, urllib.request, numpy as np
    url = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/" + name + ".npz"
    a = np.load(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    return np.asarray(a["X"], float), np.asarray(a["y"], int).ravel()


def _load_dami(name):
    import io, tarfile, urllib.request, re, numpy as np
    from scipy.io import arff
    raw = urllib.request.urlopen("https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/" + name + ".tar.gz", timeout=120).read()
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    cand = [n for n in tf.getnames() if n.endswith(".arff")]
    def sc(n):
        b = n.split("/")[-1]; pct = re.findall(r"_(\d{2})(?:_v\d+)?\.arff$", b)
        return ("withoutdupl" in b, "_norm" in b, re.search(r"_v\d+\.arff$", b) is None, -(int(pct[0]) if pct else 99))
    cand.sort(key=sc, reverse=True)
    data, meta = arff.loadarff(io.StringIO(tf.extractfile(cand[0]).read().decode("utf-8", "replace")))
    names = list(meta.names()); lab = next(c for c in names if c.lower() == "outlier")
    idc = next((c for c in names if c.lower() == "id"), None)
    feat = [c for c in names if c not in (lab, idc)]
    X = np.column_stack([np.asarray(data[c], float) for c in feat])
    y = np.array([1 if "yes" in str(v).lower() else 0 for v in data[lab]], int)
    return X, y


@app.function(image=image, volumes={"/results": vol}, cpu=2.0, memory=8192, timeout=60 * 60)
def run_cell(spec):
    import sys, os; sys.path.insert(0, "/root")
    import numpy as np, pandas as pd
    from sklearn.metrics import roc_auc_score
    bench, name = spec
    outp = f"/results/{bench}_{name}.parquet"
    if os.path.exists(outp):
        return pd.read_parquet(outp).to_dict("records")
    from adrank.pipeline import (Dataset, pseudo_auc_for_dataset, true_rank_from_labels,
                                 detector_names, fit_and_score)
    try:
        X, y = (_load_dami(name) if bench == "dami" else _load_adb(name))
    except Exception as e:
        return [{"dataset": name, "bench": bench, "status": f"fail:{type(e).__name__}"}]
    X = np.nan_to_num(X); n, d = X.shape
    if not (200 <= n <= 50000):
        return [{"dataset": name, "bench": bench, "status": "filtered"}]
    ds = Dataset(name=name, X=X, y=y); dets = detector_names(d); allidx = np.arange(n)

    def mtail(seed):
        rng = np.random.default_rng(seed); rows = []
        for j in range(M):
            f = int(rng.integers(d)); col = X[:, f]
            if rng.random() < 0.5:
                anom = allidx[col >= np.quantile(col, 1 - MTAIL_FRAC)]
            else:
                anom = allidx[col <= np.quantile(col, MTAIL_FRAC)]
            if len(anom) < 5 or len(anom) > n // 2:
                continue
            comp = allidx[~np.isin(allidx, anom)]; rng.shuffle(comp)
            nh = min(max(50, len(comp) // 5), len(comp) // 3)
            Xsc = X[np.concatenate([np.sort(comp[:nh]), anom])]
            yp = np.concatenate([np.zeros(nh), np.ones(len(anom))])
            for det in dets:
                s = fit_and_score(det, X[np.sort(comp[nh:])], Xsc)
                auc = np.nan if (s is None or len(np.unique(yp)) < 2) else roc_auc_score(yp, s)
                rows.append({"detector": det, "regime": "mtail", "pseudo_auc": auc})
        return pd.DataFrame(rows)

    def agg_disc(pp):
        ms = pp.dropna(subset=["pseudo_auc"]).groupby(["regime", "detector"]).pseudo_auc.mean()
        w = {}
        for r in pp.regime.unique():
            v = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
            w[r] = float(v.var())
        sden = sum(w.values()) or 1.0; w = {r: w[r] / sden for r in w}
        out = pd.Series(0.0, index=ms.index.get_level_values(1).unique()); ws = 0.0
        for r in ms.index.get_level_values(0).unique():
            out = out.add(ms.loc[r].rank(ascending=True) * w.get(r, 0.0), fill_value=0.0); ws += w.get(r, 0.0)
        return out / (ws or 1.0), w

    rows = []
    for seed in [0, 1, 2]:
        tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc.dropna()
        if len(tr) < 3:
            continue
        parts = []
        for K in KS:
            for sel in SELS:
                p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=seed)
                p["regime"] = f"{sel}_K{K}"; parts.append(p)
        clbank = pd.concat(parts, ignore_index=True)
        augbank = pd.concat([clbank, mtail(seed)], ignore_index=True)

        def reg(bank):
            sc, w = agg_disc(bank); sc = sc[sc.index.isin(tr.index)]
            if sc.empty:
                return np.nan, np.nan
            mrv = max(v for r, v in w.items() if r != "mtail")
            return float(tr.max() - tr[sc.sort_values(ascending=False).index[0]]), mrv
        rc, mrv = reg(clbank); ra, _ = reg(augbank)
        rows.append({"dataset": name, "bench": bench, "seed": seed, "status": "ok",
                     "regret_cluster": rc, "regret_aug": ra, "max_regime_var": mrv,
                     "random_regret": float(tr.max() - tr.mean())})
    if not rows:
        rows = [{"dataset": name, "bench": bench, "status": "no_seed"}]
    pd.DataFrame(rows).to_parquet(outp); vol.commit()
    return rows


@app.local_entrypoint()
def main():
    import pandas as pd, os
    specs = [("dami", n) for n in DAMI] + [("adb", n) for n in ADB]
    print(f"[reconcile] {len(specs)} datasets (DAMI + ADBench tabular)")
    allrows = []
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("  FAIL", res); continue
        allrows.extend(res)
    df = pd.DataFrame(allrows); os.makedirs("results", exist_ok=True)
    df.to_csv("results/reconcile.csv", index=False)
    print("[reconcile] DONE -> results/reconcile.csv")
