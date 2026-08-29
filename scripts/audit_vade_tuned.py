"""Is the VaDE negative robust to tuning? Lean diagnostic (no silhouette).

For the ceiling datasets, report VaDE effective-K (a collapse to few clusters would
be a bug), and measure ADRank regret under: PCA+KMeans (baseline), default VaDE, and
tuned VaDE variants (more epochs; clustering-protecting recon weight; latent 16 to
match PCA). If NO VaDE config lowers regret, the negative is not a tuning artifact.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from adrank.pipeline import load_npz_dir, pseudo_auc_for_dataset, true_rank_from_labels
from analyze_regimes import _aggregate_rank
import vade as vade_mod

SEEDS = [0, 1]
KS = [30, 50]
SELS = ["smallest", "random", "hard"]
M = 10
DATASETS = ["Annthyroid"]


def regret_for_precomp(ds, seed, precomp_fn):
    parts = []
    effK = []
    for K in KS:
        pre = precomp_fn(ds.X, K, seed) if precomp_fn else None
        if pre is not None:
            effK.append(len(np.unique(pre[1])))
        for sel in SELS:
            p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=seed, precomputed=pre)
            p["regime"] = f"{sel}_K{K}"; parts.append(p)
    bank = pd.concat(parts, ignore_index=True)
    tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc.dropna()
    score = _aggregate_rank(bank, "discriminative"); score = score[score.index.isin(tr.index)]
    if score.empty:
        return np.nan, (np.mean(effK) if effK else np.nan)
    pick = score.sort_values(ascending=False).index[0]
    return tr.max() - tr[pick], (np.mean(effK) if effK else np.nan)


CONFIGS = {
    "kmeans": None,
    "vade_default": lambda X, K, s: vade_mod.fit_vade(X, K, seed=s),
    "vade_joint150": lambda X, K, s: vade_mod.fit_vade(X, K, seed=s, joint_epochs=150),
    "vade_protectclust": lambda X, K, s: _vade_alpha(X, K, s, alpha_scale=0.25),
}


def _vade_alpha(X, K, s, alpha_scale):
    # monkeypatch: lower reconstruction weight so GMM clustering terms are not swamped
    import vade as v
    orig = v.fit_vade
    # replicate fit_vade but scale alpha; simplest: temporarily patch the alpha formula
    return _fit_vade_custom(X, K, s, alpha_mult=alpha_scale)


def _vade_latent(X, K, s, latent):
    return _fit_vade_custom(X, K, s, latent=latent)


def _fit_vade_custom(X, K, seed, alpha_mult=1.0, latent=None):
    """Thin re-impl calling the same internals but overriding alpha / latent."""
    import torch, torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    v = vade_mod
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xs = StandardScaler().fit_transform(X).astype(np.float32); n, d = Xs.shape
    K = min(K, max(2, n // 20)); lat = latent or v.LATENT
    alpha = (1.0 / max(1, d // 16)) * alpha_mult
    Xt = torch.from_numpy(Xs).to(dev)
    model = v.VaDE(d, K, latent=lat).to(dev)
    opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.mu.parameters()) + list(model.dec.parameters()), lr=2e-3)
    bs = min(256, n)
    for _ in range(40):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i+bs]]; z = model.mu(model.enc(xb)); loss = F.mse_loss(model.dec(z), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Zpre = model.mu(model.enc(Xt)).cpu().numpy()
    gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed, reg_covar=1e-4, n_init=3).fit(Zpre)
    model.pi.data = torch.log(torch.from_numpy(gmm.weights_).float()+1e-9).to(dev)
    model.mu_c.data = torch.from_numpy(gmm.means_).float().to(dev)
    model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float()+1e-6).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(60):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i+bs]]; xh, mu, logvar, z = model(xb)
            loss = v.vade_loss(model, xb, xh, mu, logvar, z, alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
    with torch.no_grad():
        mu, _ = model.encode(Xt); g = model.gamma(mu)
        return mu.cpu().numpy(), g.argmax(1).cpu().numpy()


def main():
    dss = {d.name: d for d in load_npz_dir(os.path.join(ROOT, "data", "dami"))}
    rows = []
    for nm in DATASETS:
        if nm not in dss: continue
        ds = dss[nm]
        for cfg, fn in CONFIGS.items():
            reg, effk = [], []
            for s in SEEDS:
                try:
                    r, k = regret_for_precomp(ds, s, fn)
                except Exception as e:
                    r, k = np.nan, np.nan
                    print(f"  {nm}/{cfg}/seed{s}: FAILED {type(e).__name__}: {e}", flush=True)
                if not np.isnan(r): reg.append(r)
                if not np.isnan(k): effk.append(k)
            row = {"dataset": nm, "config": cfg,
                   "regret": np.mean(reg) if reg else np.nan,
                   "effK": np.mean(effk) if effk else np.nan}
            rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "audit_vade_tuned.csv"), index=False)
            print(f"  {nm:16s} {cfg:18s} regret={row['regret']!s:8.8} effK={row['effK']!s:6.6}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== regret by config (lower=better); kmeans is the baseline to beat ===")
    print(df.pivot_table(index="dataset", columns="config", values="regret").round(3).to_string())


if __name__ == "__main__":
    main()
