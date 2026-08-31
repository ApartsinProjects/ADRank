"""Discriminating probe: is the balanced-VaDE collapse on Wilt/Waveform genuine
low-cluster geometry, or a reconstruction-weighting pathology (Fable memo)?

Runs VaDE on Wilt + Waveform across the alpha axis {1.0, 0.5, 0.25} plus a
KL-warmup config (ramp the mixture-prior terms 0->1 over the first 20 joint epochs
at alpha 0.25), 3 seeds each, reporting effK and regret.

Predicted: if WEIGHTING pathology, effK rises well above 3 at alpha=1.0 (and warmup
helps at low alpha). If GENUINE, effK stays <=3-4 even at alpha=1.0.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pyod==3.6.1", "scikit-learn==1.8.0", "scipy==1.17.1",
                 "pandas==2.3.3", "pyarrow", "numpy==2.2.6", "tqdm")
    .env({"ADRANK_EXCLUDE_DETECTORS": "OCSVM"})
    .add_local_dir("src/adrank", "/root/adrank")
    .add_local_file("scripts/vade.py", "/root/vade.py")
)
vol = modal.Volume.from_name("adrank-vprobe-results", create_if_missing=True)
app = modal.App("adrank-vprobe")
KS, SELS, M, SEEDS = [30, 50], ["smallest", "random", "hard"], 10, [0, 1, 2]
DATASETS = ["Wilt", "Waveform"]
CONFIGS = ["alpha1.0", "alpha0.5", "alpha0.25", "alpha0.25_warmup"]


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
    name, config = spec
    outp = f"/results/{name}_{config}.parquet"
    if os.path.exists(outp):
        return pd.read_parquet(outp).to_dict("records")
    X, y = _load_dami(name); X = np.nan_to_num(X)
    ds = Dataset(name=name, X=X, y=y)
    amap = {"alpha1.0": 1.0, "alpha0.5": 0.5, "alpha0.25": 0.25, "alpha0.25_warmup": 0.25}
    alpha_mult = amap[config]; warmup = config.endswith("warmup")

    def vloss(model, x, xh, mu, logvar, z, alpha, beta):
        lvc = model.logvar_c.clamp(min=-10.0); g = model.gamma(z)
        rec = alpha * V._mse_recon(xh, x)
        h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu_c).pow(2)
        h = (lvc + h / lvc.exp()).sum(dim=2)
        logp_z_c = 0.5 * torch.sum(g * h)
        logpi = torch.log_softmax(model.pi, dim=0)
        logp_c = torch.sum(g * logpi.unsqueeze(0)); logq_c = torch.sum(g * torch.log(g + 1e-9))
        logq_z = 0.5 * torch.sum(1 + logvar)
        return (rec + beta * (logp_z_c - logp_c + logq_c) - logq_z) / x.size(0)

    def vade_fit(K, seed, pre=40, joint=60):
        torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
        Xs = StandardScaler().fit_transform(ds.X).astype(np.float32); nn_, d = Xs.shape
        K = min(K, max(2, nn_ // 20)); alpha = (1.0 / max(1, d // 16)) * alpha_mult
        Xt = torch.from_numpy(Xs).to(dev); model = V.VaDE(d, K, latent=V.LATENT).to(dev)
        opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.mu.parameters()) + list(model.dec.parameters()), lr=2e-3)
        bs = min(256, nn_)
        for _ in range(pre):
            perm = torch.randperm(nn_, device=dev)
            for i in range(0, nn_, bs):
                xb = Xt[perm[i:i+bs]]; z = model.mu(model.enc(xb)); loss = F.mse_loss(model.dec(z), xb)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Zpre = model.mu(model.enc(Xt)).cpu().numpy()
        gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed, reg_covar=1e-4, n_init=3).fit(Zpre)
        model.pi.data = torch.log(torch.from_numpy(gmm.weights_).float() + 1e-9).to(dev)
        model.mu_c.data = torch.from_numpy(gmm.means_).float().to(dev)
        model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float() + 1e-6).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for ep in range(joint):
            beta = min(1.0, (ep + 1) / 20.0) if warmup else 1.0
            perm = torch.randperm(nn_, device=dev)
            for i in range(0, nn_, bs):
                xb = Xt[perm[i:i+bs]]; xh, mu, logvar, z = model(xb)
                loss = vloss(model, xb, xh, mu, logvar, z, alpha, beta)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        with torch.no_grad():
            mu, _ = model.encode(Xt); g = model.gamma(mu)
            return mu.cpu().numpy(), g.argmax(1).cpu().numpy()

    def agg_disc(pp):
        ms = pp.dropna(subset=["pseudo_auc"]).groupby(["regime", "detector"]).pseudo_auc.mean()
        w = {}
        for r in pp.regime.unique():
            v = pp[pp.regime == r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean(); w[r] = float(v.var())
        sden = sum(w.values()) or 1.0; w = {r: w[r] / sden for r in w}
        out = pd.Series(0.0, index=ms.index.get_level_values(1).unique()); ws = 0.0
        for r in ms.index.get_level_values(0).unique():
            out = out.add(ms.loc[r].rank(ascending=True) * w.get(r, 0.0), fill_value=0.0); ws += w.get(r, 0.0)
        return out / (ws or 1.0)

    rows = []
    for s in SEEDS:
        parts = []; effk = []
        for K in KS:
            pre = vade_fit(K, s); effk.append(len(np.unique(pre[1])))
            for sel in SELS:
                p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=s, precomputed=pre)
                p["regime"] = f"{sel}_K{K}"; parts.append(p)
        bank = pd.concat(parts, ignore_index=True)
        tr = true_rank_from_labels(ds, seed=s).set_index("detector").true_auc.dropna()
        sc = agg_disc(bank); sc = sc[sc.index.isin(tr.index)]
        reg = float(tr.max() - tr[sc.sort_values(ascending=False).index[0]]) if not sc.empty else np.nan
        rows.append({"dataset": name, "config": config, "seed": s, "regret": reg, "effK": float(np.mean(effk))})
    pd.DataFrame(rows).to_parquet(outp); vol.commit()
    return rows


@app.local_entrypoint()
def main():
    import pandas as pd, os
    specs = [(d, c) for d in DATASETS for c in CONFIGS]
    print(f"[vprobe] {len(specs)} cells on T4")
    allrows = []
    for res in run_cell.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("  FAIL", res); continue
        allrows.extend(res)
    df = pd.DataFrame(allrows); os.makedirs("results", exist_ok=True)
    df.to_csv("results/vade_probe.csv", index=False)
    print(df.groupby(["dataset", "config"]).agg(effK=("effK", "mean"), regret=("regret", "mean")).round(3).to_string())
    print("[vprobe] DONE")
