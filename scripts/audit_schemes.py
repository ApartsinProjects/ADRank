"""Test improved label-free regime-weighting schemes against the discriminative
baseline, on ADBench (tabular/cv/nlp/ts) plus DAMI, 5 seeds each.

Every scheme sees only pseudo-AUCs (label-free) except those prefixed ORACLE_.

Schemes:
  baselines : ens_smallest_rand, uniform_bank, discriminative, stability
  new       : disc_x_stab      variance * (1/rank-instability) weights
              margin           weight = pseudo-AUC gap top1-top2 per regime
              softmax_var_T*   softmax(variance/T) weights
              agreement        weight = mean Spearman corr of a regime's ranking
                               with the other regimes' rankings
              disc_agree       variance * agreement
              best_regime_var  single regime with max cross-detector variance
              best_regime_agr  single regime with max agreement
              zscore_disc      variance-weighted average of within-regime
                               z-scored pseudo-AUC (scores, not ranks)
              top1_vote        detector ranked #1 by most regimes (Borda tiebreak)
  oracle    : ORACLE_best_regime (per dataset, uses labels; upper bound)

Outputs results/audit_schemes.csv (per modality x scheme, mean+-std over seeds)
and results/audit_schemes_cells.csv (per cell regret for the key schemes).
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")


def load():
    p = pd.concat([pd.read_parquet(os.path.join(RAW, "modal_pseudo_regimes.parquet")),
                   pd.read_parquet(os.path.join(RAW, "modal_pseudo_dami.parquet"))],
                  ignore_index=True)
    t = pd.concat([pd.read_parquet(os.path.join(RAW, "modal_true_regimes.parquet")),
                   pd.read_parquet(os.path.join(RAW, "modal_true_dami.parquet"))],
                  ignore_index=True)
    return p, t


def detector_scores(cell):
    """(regime, detector) -> mean pseudo-AUC for one (modality,dataset,seed)."""
    return (cell.dropna(subset=["pseudo_auc"])
            .groupby(["regime", "detector"]).pseudo_auc.mean())


def regime_tables(cell):
    ms = detector_scores(cell)
    regimes = list(ms.index.get_level_values(0).unique())
    tabs = {r: ms.loc[r] for r in regimes}
    return tabs


def w_variance(tabs):
    return {r: float(v.var()) for r, v in tabs.items()}


def w_stability(cell):
    w = {}
    for r, sub in cell.dropna(subset=["pseudo_auc"]).groupby("regime"):
        sub = sub.copy()
        sub["rk"] = sub.groupby("subset").pseudo_auc.rank(ascending=False)
        inst = sub.groupby("detector").rk.std().mean()
        w[r] = 1.0 / (inst + 1e-6)
    return w


def w_margin(tabs):
    w = {}
    for r, v in tabs.items():
        s = v.sort_values(ascending=False)
        w[r] = float(s.iloc[0] - s.iloc[1]) if len(s) > 1 else 0.0
    return w


def w_agreement(tabs):
    regimes = list(tabs)
    ranks = {r: tabs[r].rank() for r in regimes}
    w = {}
    for r in regimes:
        cs = []
        for r2 in regimes:
            if r2 == r:
                continue
            common = ranks[r].index.intersection(ranks[r2].index)
            if len(common) >= 3:
                cs.append(spearmanr(ranks[r].loc[common], ranks[r2].loc[common]).statistic)
        w[r] = max(0.0, float(np.nanmean(cs))) if cs else 0.0
    return w


def norm(w):
    s = sum(w.values())
    return {r: (v / s if s > 0 else 1.0 / len(w)) for r, v in w.items()}


def rank_average(tabs, w):
    w = norm(w)
    agg = None
    for r, v in tabs.items():
        rk = v.rank(ascending=True) * w.get(r, 0.0)
        agg = rk if agg is None else agg.add(rk, fill_value=0.0)
    return agg


def zscore_average(tabs, w):
    w = norm(w)
    agg = None
    for r, v in tabs.items():
        sd = v.std()
        z = (v - v.mean()) / (sd if sd > 0 else 1.0)
        z = z * w.get(r, 0.0)
        agg = z if agg is None else agg.add(z, fill_value=0.0)
    return agg


def top1_vote(tabs):
    votes = {}
    borda = None
    for r, v in tabs.items():
        votes[v.idxmax()] = votes.get(v.idxmax(), 0) + 1
        rk = v.rank(ascending=True)
        borda = rk if borda is None else borda.add(rk, fill_value=0.0)
    vs = pd.Series(0.0, index=borda.index)
    for d, c in votes.items():
        vs[d] = c
    return vs * 1000 + borda  # votes dominate, borda breaks ties


def scheme_score(cell, scheme):
    tabs = regime_tables(cell)
    if scheme == "ens_smallest_rand":
        keep = {r: v for r, v in tabs.items() if r in ("smallest_K30", "random_K30")}
        return pd.concat(keep.values()).groupby(level=0).mean()
    if scheme == "uniform_bank":
        return rank_average(tabs, {r: 1.0 for r in tabs})
    if scheme == "discriminative":
        return rank_average(tabs, w_variance(tabs))
    if scheme == "stability":
        return rank_average(tabs, w_stability(cell))
    if scheme == "disc_x_stab":
        wv, ws = w_variance(tabs), w_stability(cell)
        return rank_average(tabs, {r: wv[r] * ws.get(r, 0.0) for r in tabs})
    if scheme == "margin":
        return rank_average(tabs, w_margin(tabs))
    if scheme.startswith("softmax_var_T"):
        T = float(scheme.split("T")[-1])
        wv = w_variance(tabs)
        mx = max(wv.values())
        e = {r: np.exp((v - mx) / T) for r, v in wv.items()}
        return rank_average(tabs, e)
    if scheme == "agreement":
        return rank_average(tabs, w_agreement(tabs))
    if scheme == "disc_agree":
        wv, wa = w_variance(tabs), w_agreement(tabs)
        return rank_average(tabs, {r: wv[r] * wa[r] for r in tabs})
    if scheme == "best_regime_var":
        wv = w_variance(tabs)
        r = max(wv, key=wv.get)
        return tabs[r]
    if scheme == "best_regime_agr":
        wa = w_agreement(tabs)
        r = max(wa, key=wa.get)
        return tabs[r]
    if scheme == "zscore_disc":
        return zscore_average(tabs, w_variance(tabs))
    if scheme == "zscore_uniform":
        return zscore_average(tabs, {r: 1.0 for r in tabs})
    if scheme == "top1_vote":
        return top1_vote(tabs)
    raise ValueError(scheme)


SCHEMES = ["ens_smallest_rand", "uniform_bank", "discriminative", "stability",
           "disc_x_stab", "margin", "softmax_var_T0.001", "softmax_var_T0.0003",
           "agreement", "disc_agree", "best_regime_var", "best_regime_agr",
           "zscore_disc", "zscore_uniform", "top1_vote",
           "ORACLE_best_regime"]


def main():
    p, t = load()
    cell_rows = []
    for m in ["tabular", "cv", "nlp", "ts", "dami"]:
        pm, tm = p[p.modality == m], t[t.modality == m]
        for seed in sorted(pm.seed.unique()):
            ps = pm[pm.seed == seed]
            tavg = (tm[tm.seed == seed].groupby(["dataset", "detector"])
                    .true_auc.mean().reset_index())
            for ds, cell in ps.groupby("dataset"):
                tds = tavg[tavg.dataset == ds].set_index("detector").true_auc
                if tds.empty:
                    continue
                best = tds.max()
                rec = dict(modality=m, seed=seed, dataset=ds)
                for scheme in SCHEMES:
                    if scheme == "ORACLE_best_regime":
                        tabs = regime_tables(cell)
                        r1 = min(best - tds.get(v.idxmax(), np.nan) for v in tabs.values())
                        r3 = r1  # not meaningful; keep r1
                        rec["r1_" + scheme] = r1
                        rec["r3_" + scheme] = r3
                        continue
                    score = scheme_score(cell, scheme)
                    score = score[score.index.isin(tds.index)]
                    order = score.sort_values(ascending=False)
                    rec["r1_" + scheme] = best - tds.get(order.index[0], np.nan)
                    rec["r3_" + scheme] = best - tds.loc[tds.index.intersection(order.index[:3])].max()
                cell_rows.append(rec)

    cells = pd.DataFrame(cell_rows)
    cells.to_csv(os.path.join(ROOT, "results", "audit_schemes_cells.csv"), index=False)

    # aggregate: mean over datasets per seed, then mean+-std over seeds
    out = []
    for m, g in cells.groupby("modality"):
        seed_means = g.groupby("seed").mean(numeric_only=True)
        for scheme in SCHEMES:
            out.append(dict(modality=m, scheme=scheme,
                            r1_mean=seed_means["r1_" + scheme].mean(),
                            r1_std=seed_means["r1_" + scheme].std(ddof=0),
                            r3_mean=seed_means["r3_" + scheme].mean(),
                            r3_std=seed_means["r3_" + scheme].std(ddof=0)))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(ROOT, "results", "audit_schemes.csv"), index=False)

    pd.set_option("display.width", 300)
    piv = res.pivot(index="scheme", columns="modality", values="r1_mean").round(4)
    piv = piv[["tabular", "cv", "nlp", "ts", "dami"]]
    piv["mean_all"] = piv.mean(axis=1)
    print("regret@1 mean over seeds (rows=schemes):")
    print(piv.sort_values("mean_all").to_string())
    print("\nstd over seeds:")
    piv2 = res.pivot(index="scheme", columns="modality", values="r1_std").round(4)
    print(piv2[["tabular", "cv", "nlp", "ts", "dami"]].to_string())
    print("\nregret@3 mean over seeds:")
    piv3 = res.pivot(index="scheme", columns="modality", values="r3_mean").round(4)
    print(piv3[["tabular", "cv", "nlp", "ts", "dami"]].to_string())


if __name__ == "__main__":
    main()
