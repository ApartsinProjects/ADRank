"""VaDE (Variational Deep Embedding, Jiang et al. IJCAI 2017) for ADRank.

Joint latent embedding + GMM clustering, to replace PCA-16 + MiniBatchKMeans and
test whether a jointly-learned clustering dents the local-detector ceiling.

Adapted for standardized CONTINUOUS features: Gaussian decoder / MSE reconstruction
(not the MNIST Bernoulli/BCE). log-var parameterization throughout. Recipe: pretrain
a plain autoencoder (MSE), init the GMM prior from a diagonal GMM on the pretrained
embeddings, then joint VaDE training. gamma (cluster posterior) computed in log-space
with logsumexp for stability. Loss verbatim-equivalent to mperezcarrasco/Pytorch-VaDE
with the reconstruction term swapped to MSE.

fit_vade(X, K, seed) -> (Z, labels): Z = encoder mean (latent), labels = argmax gamma.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

LATENT = 10


class VaDE(nn.Module):
    def __init__(self, in_dim, K, latent=LATENT, hidden=256):
        super().__init__()
        self.K, self.latent = K, latent
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, in_dim))
        self.pi = nn.Parameter(torch.ones(K) / K)
        self.mu_c = nn.Parameter(torch.zeros(K, latent))
        self.logvar_c = nn.Parameter(torch.randn(K, latent) * 0.1)

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.dec(z), mu, logvar, z

    def gamma(self, z):
        # log p(z|c) per cluster, diagonal Gaussian; log-space + logsumexp
        lvc = self.logvar_c.clamp(min=-10.0)
        h = (z.unsqueeze(1) - self.mu_c).pow(2) / lvc.exp() + lvc  # (B,K,J)
        logpzc = -0.5 * (h + np.log(2 * np.pi)).sum(dim=2)         # (B,K)
        logpi = torch.log_softmax(self.pi, dim=0)                  # keep pi normalized
        log_num = logpi.unsqueeze(0) + logpzc
        return torch.softmax(log_num, dim=1)                       # (B,K) gamma


def _mse_recon(xh, x):
    return 0.5 * F.mse_loss(xh, x, reduction="sum")


def vade_loss(model, x, xh, mu, logvar, z, alpha):
    lvc = model.logvar_c.clamp(min=-10.0)
    g = model.gamma(z)                                            # (B,K)
    rec = alpha * _mse_recon(xh, x)
    # E_q[log p(z|c)] cross-term, weighted by gamma
    h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu_c).pow(2)  # (B,K,J)
    h = (lvc + h / lvc.exp()).sum(dim=2)                          # (B,K)
    logp_z_c = 0.5 * torch.sum(g * h)
    logpi = torch.log_softmax(model.pi, dim=0)
    logp_c = torch.sum(g * logpi.unsqueeze(0))
    logq_c = torch.sum(g * torch.log(g + 1e-9))
    logq_z = 0.5 * torch.sum(1 + logvar)
    return (rec + logp_z_c - logp_c + logq_c - logq_z) / x.size(0)


def fit_vade(X, K, seed=0, pre_epochs=40, joint_epochs=60, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    n, d = Xs.shape
    K = min(K, max(2, n // 20))
    alpha = 1.0 / max(1, d // 16)          # keep recon from swamping clustering at high d
    Xt = torch.from_numpy(Xs).to(dev)
    model = VaDE(d, K).to(dev)

    # 1) pretrain plain autoencoder (MSE)
    opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.mu.parameters())
                           + list(model.dec.parameters()), lr=2e-3)
    bs = min(256, n)
    for ep in range(pre_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb = Xt[idx]
            h = model.enc(xb); z = model.mu(h); xh = model.dec(z)
            loss = F.mse_loss(xh, xb)
            opt.zero_grad(); loss.backward(); opt.step()

    # 2) GMM init on pretrained embeddings
    with torch.no_grad():
        Zpre = model.mu(model.enc(Xt)).cpu().numpy()
    gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed,
                          reg_covar=1e-4, n_init=3).fit(Zpre)
    model.pi.data = torch.log(torch.from_numpy(gmm.weights_).float() + 1e-9).to(dev)  # store as logits
    model.mu_c.data = torch.from_numpy(gmm.means_).float().to(dev)
    model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float() + 1e-6).to(dev)

    # 3) joint VaDE training
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    for ep in range(joint_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb = Xt[idx]
            xh, mu, logvar, z = model(xb)
            loss = vade_loss(model, xb, xh, mu, logvar, z, alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()

    # 4) outputs: latent = encoder mean; labels = argmax gamma
    with torch.no_grad():
        mu, logvar = model.encode(Xt)
        g = model.gamma(mu)
        labels = g.argmax(dim=1).cpu().numpy()
        Z = mu.cpu().numpy()
    return Z, labels
