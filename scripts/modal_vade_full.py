"""Modal (llmcourse workspace) VaDE-tuning audit for the ADRank ceiling.

Fans out over (dataset, config): KMeans vs 6 VaDE configs on the 4 DAMI ceiling
datasets, 3 seeds each, on a T4 GPU. Each container fetches its DAMI dataset from
LMU (ARFF). Answers whether the VaDE negative survives tuning.
"""
import modal

APP_NAME = "adrank-vade-full"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6", "tqdm")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
    .add_local_file("scripts/vade.py", "/root/vade.py")
)
app = modal.App(APP_NAME)

CEILING = ["Annthyroid","WDBC","Cardiotocography","PageBlocks","Pima","Stamps","Waveform","WBC","Wilt"]
CONFIGS = ["kmeans", "vade_protectclust"]
SEEDS = [0, 1, 2]
KS = [30, 50]
SELS = ["smallest", "random", "hard"]
M = 10


@app.function(image=image, gpu="T4", cpu=4.0, memory=16384, timeout=60 * 60)
def run_cell(spec):
    import sys, io, tarfile, urllib.request, re, time
    sys.path.insert(0, "/root")
    import numpy as np, pandas as pd, torch, torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from adrank.pipeline import Dataset, pseudo_auc_for_dataset, true_rank_from_labels
    import vade as V

    dataset, config = spec

    # --- fetch DAMI dataset (ARFF) ---
    from scipy.io import arff
    BASE = "https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/"
    raw = urllib.request.urlopen(BASE + dataset + ".tar.gz", timeout=120).read()
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    cand = [n for n in tf.getnames() if n.endswith(".arff")]
    def sc(n):
        b = n.split("/")[-1]; pct = re.findall(r"_(\d{2})(?:_v\d+)?\.arff$", b)
        return ("withoutdupl" in b, "_norm" in b, re.search(r"_v\d+\.arff$", b) is None,
                -(int(pct[0]) if pct else 99))
    cand.sort(key=sc, reverse=True)
    data, meta = arff.loadarff(io.StringIO(tf.extractfile(cand[0]).read().decode("utf-8", "replace")))
    names = list(meta.names())
    lab = next(c for c in names if c.lower() == "outlier")
    idc = next((c for c in names if c.lower() == "id"), None)
    feat = [c for c in names if c not in (lab, idc)]
    X = np.column_stack([np.asarray(data[c], dtype=np.float64) for c in feat])
    y = np.array([1 if str(v).lower().find("yes") >= 0 else 0 for v in data[lab]], dtype=int)
    ds = Dataset(name=dataset, X=X, y=y)

    def vade_fit(X, K, seed, alpha_mult=1.0, latent=None, pre=40, joint=60):
        torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
        Xs = StandardScaler().fit_transform(X).astype(np.float32); n, d = Xs.shape
        K = min(K, max(2, n // 20)); lat = latent or V.LATENT
        alpha = (1.0 / max(1, d // 16)) * alpha_mult
        Xt = torch.from_numpy(Xs).to(dev); model = V.VaDE(d, K, latent=lat).to(dev)
        opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.mu.parameters())
                               + list(model.dec.parameters()), lr=2e-3)
        bs = min(256, n)
        for _ in range(pre):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                xb = Xt[perm[i:i+bs]]; z = model.mu(model.enc(xb)); loss = F.mse_loss(model.dec(z), xb)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Zpre = model.mu(model.enc(Xt)).cpu().numpy()
        gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed,
                              reg_covar=1e-4, n_init=3).fit(Zpre)
        model.pi.data = torch.log(torch.from_numpy(gmm.weights_).float() + 1e-9).to(dev)
        model.mu_c.data = torch.from_numpy(gmm.means_).float().to(dev)
        model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float() + 1e-6).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(joint):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                xb = Xt[perm[i:i+bs]]; xh, mu, logvar, z = model(xb)
                loss = V.vade_loss(model, xb, xh, mu, logvar, z, alpha)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        with torch.no_grad():
            mu, _ = model.encode(Xt); g = model.gamma(mu)
            return mu.cpu().numpy(), g.argmax(1).cpu().numpy()

    cfg_fns = {
        "kmeans": None,
        "vade_default":      lambda X, K, s: vade_fit(X, K, s),
        "vade_joint150":     lambda X, K, s: vade_fit(X, K, s, joint=150),
        "vade_pre100j150":   lambda X, K, s: vade_fit(X, K, s, pre=100, joint=150),
        "vade_protectclust": lambda X, K, s: vade_fit(X, K, s, alpha_mult=0.25),
        "vade_boostclust":   lambda X, K, s: vade_fit(X, K, s, alpha_mult=0.1, joint=150),
        "vade_latent16":     lambda X, K, s: vade_fit(X, K, s, latent=16),
    }
    fn = cfg_fns[config]

    def agg_disc(pp):
        ms = pp.dropna(subset=["pseudo_auc"]).groupby(["regime", "detector"]).pseudo_auc.mean()
        regs = pp.regime.unique(); w = {}
        for r in regs:
            v = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
            w[r] = float(v.var())
        sden = sum(w.values()) or 1.0; w = {r: w[r]/sden for r in regs}
        dets = ms.index.get_level_values(1).unique()
        out = pd.Series(0.0, index=dets); wsum = 0.0
        for r in ms.index.get_level_values(0).unique():
            out = out.add(ms.loc[r].rank(ascending=True) * w.get(r, 0.0), fill_value=0.0); wsum += w.get(r, 0.0)
        return out / (wsum or 1.0)

    reg, ek = [], []
    for s in SEEDS:
        parts = []; effK = []
        for K in KS:
            pre = fn(ds.X, K, s) if fn else None
            if pre is not None:
                effK.append(len(np.unique(pre[1])))
            for sel in SELS:
                p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=s, precomputed=pre)
                p["regime"] = f"{sel}_K{K}"; parts.append(p)
        bank = pd.concat(parts, ignore_index=True)
        tr = true_rank_from_labels(ds, seed=s).set_index("detector").true_auc.dropna()
        score = agg_disc(bank); score = score[score.index.isin(tr.index)]
        if not score.empty:
            pick = score.sort_values(ascending=False).index[0]
            reg.append(float(tr.max() - tr[pick]))
        if effK:
            ek.append(float(np.mean(effK)))
    import numpy as np
    return {"dataset": dataset, "config": config,
            "regret": float(np.mean(reg)) if reg else None,
            "effK": float(np.mean(ek)) if ek else None}


@app.local_entrypoint()
def main():
    import pandas as pd
    specs = [(d, c) for d in CEILING for c in CONFIGS]
    print(f"[vade] launching {len(specs)} cells on T4 ...")
    rows = []
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  FAILED: {res}"); continue
        rows.append(res)
        print(f"  {res['dataset']}/{res['config']}: regret={res['regret']} effK={res['effK']}")
    import os; os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/vade_full_dami.csv", index=False)
    print("\n[vade] === regret by config (kmeans=baseline; lower better) ===")
    print(df.pivot_table(index="dataset", columns="config", values="regret").round(3).to_string())
    print("[vade] DONE")
