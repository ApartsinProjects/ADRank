"""B2 scale-up: effK-gated hybrid clustering on the FULL DAMI (10) + ADBench
Classical tabular (26) = 36 datasets, for a paired significance test.

Per dataset, over 3 seeds, computes k-means regret and balanced-VaDE (protectclust)
regret + effective cluster count (effK). Offline, the gated hybrid is:
  gated = VaDE if effK >= 3 else k-means
and we paired-Wilcoxon the gated hybrid against k-means across the 36 datasets.
GPU (T4). Reuses scripts/vade.py.
"""
import modal

APP_NAME = "adrank-b2-scaleup"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6", "tqdm")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
    .add_local_file("scripts/vade.py", "/root/vade.py")
)
vol = modal.Volume.from_name("adrank-b2-results", create_if_missing=True)
app = modal.App(APP_NAME)
KS, SELS, M, SEEDS = [30, 50], ["smallest", "random", "hard"], 10, [0, 1, 2]
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
    raw = urllib.request.urlopen("https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/" + name + ".tar.gz", timeout=180).read()
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


@app.function(image=image, gpu="T4", cpu=4.0, memory=16384, timeout=60 * 60, volumes={"/results": vol})
def run_cell(spec):
    import sys, os; sys.path.insert(0, "/root")
    import numpy as np, pandas as pd, torch, torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from adrank.pipeline import Dataset, pseudo_auc_for_dataset, true_rank_from_labels
    import vade as V
    bench, name = spec
    outp = f"/results/{bench}_{name}.parquet"
    if os.path.exists(outp):
        return pd.read_parquet(outp).to_dict("records")
    try:
        X, y = (_load_dami(name) if bench == "dami" else _load_adb(name))
    except Exception as e:
        return [{"dataset": name, "bench": bench, "status": f"fail:{type(e).__name__}"}]
    X = np.nan_to_num(X); n = X.shape[0]
    if not (200 <= n <= 50000):
        return [{"dataset": name, "bench": bench, "status": "filtered"}]
    ds = Dataset(name=name, X=X, y=y)

    def vade_fit(X, K, seed, alpha_mult=0.25, pre=40, joint=60):
        torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
        Xs = StandardScaler().fit_transform(X).astype(np.float32); nn, d = Xs.shape
        K = min(K, max(2, nn // 20)); alpha = (1.0 / max(1, d // 16)) * alpha_mult
        Xt = torch.from_numpy(Xs).to(dev); model = V.VaDE(d, K, latent=V.LATENT).to(dev)
        opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.mu.parameters()) + list(model.dec.parameters()), lr=2e-3)
        bs = min(256, nn)
        for _ in range(pre):
            perm = torch.randperm(nn, device=dev)
            for i in range(0, nn, bs):
                xb = Xt[perm[i:i+bs]]; z = model.mu(model.enc(xb)); loss = F.mse_loss(model.dec(z), xb)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Zpre = model.mu(model.enc(Xt)).cpu().numpy()
        gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed, reg_covar=1e-4, n_init=3).fit(Zpre)
        model.pi.data = torch.log(torch.from_numpy(gmm.weights_).float() + 1e-9).to(dev)
        model.mu_c.data = torch.from_numpy(gmm.means_).float().to(dev)
        model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float() + 1e-6).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(joint):
            perm = torch.randperm(nn, device=dev)
            for i in range(0, nn, bs):
                xb = Xt[perm[i:i+bs]]; xh, mu, logvar, z = model(xb)
                loss = V.vade_loss(model, xb, xh, mu, logvar, z, alpha)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        with torch.no_grad():
            mu, _ = model.encode(Xt); g = model.gamma(mu)
            return mu.cpu().numpy(), g.argmax(1).cpu().numpy()

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
        return out / (ws or 1.0)

    def regret(seed, precomp):
        parts = []; effk = []
        for K in KS:
            pre = vade_fit(ds.X, K, seed) if precomp else None
            if pre is not None:
                effk.append(len(np.unique(pre[1])))
            for sel in SELS:
                p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=seed, precomputed=pre)
                p["regime"] = f"{sel}_K{K}"; parts.append(p)
        bank = pd.concat(parts, ignore_index=True)
        tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc.dropna()
        sc = agg_disc(bank); sc = sc[sc.index.isin(tr.index)]
        if sc.empty or len(tr) < 3:
            return np.nan, np.nan
        return float(tr.max() - tr[sc.sort_values(ascending=False).index[0]]), (float(np.mean(effk)) if effk else np.nan)

    rows = []
    for s in SEEDS:
        rk, _ = regret(s, False)
        rv, ek = regret(s, True)
        rows.append({"dataset": name, "bench": bench, "seed": s, "status": "ok",
                     "regret_kmeans": rk, "regret_vade": rv, "effK": ek})
    pd.DataFrame(rows).to_parquet(outp); vol.commit()
    return rows


@app.local_entrypoint()
def main():
    import pandas as pd, os
    specs = [("dami", n) for n in DAMI] + [("adb", n) for n in ADB]
    print(f"[b2] {len(specs)} datasets on T4 (k-means vs guarded-VaDE)")
    allrows = []
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("  FAIL", res); continue
        allrows.extend(res)
    df = pd.DataFrame(allrows); os.makedirs("results", exist_ok=True)
    df.to_csv("results/b2_scaleup.csv", index=False)
    print("[b2] DONE -> results/b2_scaleup.csv")
