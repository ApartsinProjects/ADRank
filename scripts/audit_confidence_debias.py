"""Two follow-up audits.

A) CONFIDENCE: label-free per-cell confidence signals for the discriminative
   pick, correlated with realized regret:
     - top1_agree : fraction of regimes whose top-1 equals the aggregate pick
     - rank_margin: aggregated-rank gap between pick and runner-up (normalized)
     - inter_regime_rho: mean pairwise Spearman between regime rankings
     - true-side  : rand_regret (needs labels; shown for reference only)

B) DEBIAS: cluster-holdout pseudo-anomalies systematically favor local
   detectors (LOF/KNN/CBLOF). Label-free panel-level correction: within a
   modality+seed, z-score each detector's aggregated rank across datasets and
   pick by alpha*rank + (1-alpha)*surprise. Uses only pseudo-AUCs. Recompute
   regret for alpha grid.

C) MISSING REGIME: oracle-in-bank regret split by whether the true-best
   detector is local vs global. If the bank cannot express "global wins",
   oracle-in-bank stays high on global-best cells -> a global-pseudo-anomaly
   regime is the missing ingredient.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from audit_schemes import load, regime_tables, w_variance, rank_average

LOCAL = {"LOF", "KNN", "CBLOF"}


def disc_score(cell):
    tabs = regime_tables(cell)
    return rank_average(tabs, w_variance(tabs)), tabs


def main():
    p, t = load()
    conf_rows = []
    # per (modality, seed): collect aggregated scores for all datasets first (for debias)
    debias_cells = []
    for m in ["tabular", "cv", "nlp", "ts", "dami"]:
        pm, tm = p[p.modality == m], t[t.modality == m]
        for seed in sorted(pm.seed.unique()):
            ps = pm[pm.seed == seed]
            tavg = (tm[tm.seed == seed].groupby(["dataset", "detector"])
                    .true_auc.mean().reset_index())
            scores = {}
            truths = {}
            for ds, cell in ps.groupby("dataset"):
                tds = tavg[tavg.dataset == ds].set_index("detector").true_auc
                if tds.empty:
                    continue
                sc, tabs = disc_score(cell)
                sc = sc[sc.index.isin(tds.index)]
                scores[ds] = sc
                truths[ds] = tds
                pick = sc.idxmax()
                best = tds.max()
                regret = best - tds.get(pick, np.nan)
                # confidence signals
                tops = [v.idxmax() for v in tabs.values()]
                top1_agree = np.mean([tt == pick for tt in tops])
                so = sc.sort_values(ascending=False)
                rank_margin = (so.iloc[0] - so.iloc[1]) / (len(sc) - 1)
                rhos = []
                keys = list(tabs)
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        a, b = tabs[keys[i]], tabs[keys[j]]
                        c = a.index.intersection(b.index)
                        if len(c) >= 3:
                            rhos.append(spearmanr(a.loc[c], b.loc[c]).statistic)
                conf_rows.append(dict(modality=m, seed=seed, dataset=ds,
                                      regret=regret, top1_agree=top1_agree,
                                      rank_margin=rank_margin,
                                      inter_regime_rho=np.nanmean(rhos),
                                      rand_regret=best - tds.mean(),
                                      best_is_local=tds.idxmax() in LOCAL))
            # ---- debias within (modality, seed) panel ----
            S = pd.DataFrame(scores).T  # datasets x detectors, aggregated rank scores
            mu, sd = S.mean(axis=0), S.std(axis=0).replace(0, 1.0)
            Z = (S - mu) / sd  # surprise: how unusually high is this detector here
            for alpha in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]:
                # rank scores normalized 0..1 per dataset for comparability
                Sn = S.sub(S.min(axis=1), axis=0)
                Sn = Sn.div(Sn.max(axis=1).replace(0, 1.0), axis=0)
                Zn = Z.sub(Z.min(axis=1), axis=0)
                Zn = Zn.div(Zn.max(axis=1).replace(0, 1.0), axis=0)
                C = alpha * Sn + (1 - alpha) * Zn
                for ds in C.index:
                    tds = truths[ds]
                    pick = C.loc[ds].dropna().idxmax()
                    debias_cells.append(dict(modality=m, seed=seed, dataset=ds,
                                             alpha=alpha,
                                             regret=tds.max() - tds.get(pick, np.nan)))

    conf = pd.DataFrame(conf_rows)
    conf.to_csv(os.path.join(ROOT, "results", "audit_confidence.csv"), index=False)
    deb = pd.DataFrame(debias_cells)

    pd.set_option("display.width", 250)
    print("== A) confidence vs realized regret (Spearman, per modality) ==")
    for m, g in conf.groupby("modality"):
        r1 = spearmanr(g.top1_agree, g.regret).statistic
        r2 = spearmanr(g.rank_margin, g.regret).statistic
        r3 = spearmanr(g.inter_regime_rho, g.regret).statistic
        print(f"{m:8s} top1_agree {r1:+.2f}  rank_margin {r2:+.2f}  inter_regime_rho {r3:+.2f}  (n={len(g)})")
    g = conf
    print(f"{'ALL':8s} top1_agree {spearmanr(g.top1_agree, g.regret).statistic:+.2f}  "
          f"rank_margin {spearmanr(g.rank_margin, g.regret).statistic:+.2f}  "
          f"inter_regime_rho {spearmanr(g.inter_regime_rho, g.regret).statistic:+.2f}")
    # binned view: high vs low confidence
    med = conf.groupby("modality").top1_agree.transform("median")
    conf["conf_hi"] = conf.top1_agree >= med
    print("\nmean regret by top1_agree above/below modality median:")
    print(conf.groupby(["modality", "conf_hi"]).regret.mean().unstack().round(4))

    print("\n== B) debias alpha sweep: mean regret@1 (seed-mean +- std) ==")
    sm = deb.groupby(["modality", "alpha", "seed"]).regret.mean().reset_index()
    tab = sm.groupby(["modality", "alpha"]).regret.agg(["mean", "std"]).round(4)
    print(tab.unstack(0)["mean"].to_string())

    print("\n== C) oracle-in-bank regret by true-best locality ==")
    aud = pd.read_csv(os.path.join(ROOT, "results", "audit_per_dataset.csv"))
    aud["best_is_local"] = aud.true_best.isin(LOCAL)
    print(aud.groupby(["modality", "best_is_local"])
          .agg(n=("best_regime_regret", "size"),
               oracle_in_bank=("best_regime_regret", "mean"),
               disc=("regret", "mean")).round(4))


if __name__ == "__main__":
    main()
