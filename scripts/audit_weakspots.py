"""Audit ADRank weak spots: per-dataset regret diagnosis on the regime bank.

For every (modality incl. DAMI, seed, dataset):
  - discriminative-scheme pick and its regret@1
  - true best detector and true AUC spread (best-minus-mean = random regret)
  - per-regime top-1 pick regret (is the right answer present in the bank?)
  - per-regime Spearman rho between mean pseudo-AUC and true AUC
  - category: ill_posed (best-minus-mean < 0.02), flip (no regime in the bank
    picks within eps of best), weighting_failure (some regime picks well but
    the discriminative aggregate does not), ok (regret < eps)

Outputs results/audit_per_dataset.csv and prints summaries.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("ADRANK_EXCLUDE_DETECTORS", "OCSVM")

from analyze_regimes import _aggregate_rank, _detector_scores_per_regime

EPS = 0.01  # "picked well" tolerance on true AUC


def load():
    p1 = pd.read_parquet(os.path.join(RAW, "modal_pseudo_regimes.parquet"))
    t1 = pd.read_parquet(os.path.join(RAW, "modal_true_regimes.parquet"))
    p2 = pd.read_parquet(os.path.join(RAW, "modal_pseudo_dami.parquet"))
    t2 = pd.read_parquet(os.path.join(RAW, "modal_true_dami.parquet"))
    p = pd.concat([p1, p2], ignore_index=True)
    t = pd.concat([t1, t2], ignore_index=True)
    return p, t


def main():
    p, t = load()
    rows = []
    for m in ["tabular", "cv", "nlp", "ts", "dami"]:
        pm = p[p.modality == m]
        tm = t[t.modality == m]
        for seed in sorted(pm.seed.unique()):
            ps = pm[pm.seed == seed]
            tavg = (tm[tm.seed == seed]
                    .groupby(["dataset", "detector"]).true_auc.mean().reset_index())
            for ds, cell in ps.groupby("dataset"):
                tds = tavg[tavg.dataset == ds].set_index("detector").true_auc
                if tds.empty:
                    continue
                best = tds.max()
                best_det = tds.idxmax()
                rand_regret = best - tds.mean()

                score = _aggregate_rank(cell, "discriminative")
                score = score[score.index.isin(tds.index)]
                pick = score.sort_values(ascending=False).index[0]
                regret = best - tds.get(pick, np.nan)

                # per-regime diagnostics
                ms = _detector_scores_per_regime(cell)
                regime_regret = {}
                regime_rho = {}
                for r in ms.index.get_level_values(0).unique():
                    rr = ms.loc[r]
                    rr = rr[rr.index.isin(tds.index)]
                    rpick = rr.sort_values(ascending=False).index[0]
                    regime_regret[r] = best - tds.get(rpick, np.nan)
                    common = rr.index.intersection(tds.index)
                    if len(common) >= 3:
                        rho = spearmanr(rr.loc[common], tds.loc[common]).statistic
                    else:
                        rho = np.nan
                    regime_rho[r] = rho
                best_regime_regret = min(regime_regret.values())
                best_regime = min(regime_regret, key=regime_regret.get)

                if regret <= EPS:
                    cat = "ok"
                elif rand_regret < 0.02:
                    cat = "ill_posed"
                elif best_regime_regret <= EPS:
                    cat = "weighting_failure"
                else:
                    cat = "flip"

                row = dict(modality=m, seed=seed, dataset=ds,
                           regret=regret, pick=pick, true_best=best_det,
                           best_auc=best, pick_auc=tds.get(pick, np.nan),
                           rand_regret=rand_regret,
                           best_regime=best_regime,
                           best_regime_regret=best_regime_regret,
                           category=cat)
                for r, v in regime_regret.items():
                    row[f"regret_{r}"] = v
                for r, v in regime_rho.items():
                    row[f"rho_{r}"] = v
                rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "audit_per_dataset.csv")
    df.to_csv(out, index=False)
    print("saved", out, df.shape)

    # ---- summaries ----
    pd.set_option("display.width", 250)
    print("\n== mean regret per modality (discriminative), sanity vs known ==")
    print(df.groupby(["modality", "seed"]).regret.mean().groupby("modality").agg(["mean", "std"]))

    print("\n== category counts (dataset-seed cells) ==")
    print(df.groupby(["modality", "category"]).size().unstack(fill_value=0))

    # per-dataset average over seeds, worst first
    dd = df.groupby(["modality", "dataset"]).agg(
        regret=("regret", "mean"), rand_regret=("rand_regret", "mean"),
        best_regime_regret=("best_regime_regret", "mean"),
        pick=("pick", lambda s: s.mode()[0]),
        true_best=("true_best", lambda s: s.mode()[0]),
        cat=("category", lambda s: s.mode()[0])).reset_index()
    worst = dd.sort_values("regret", ascending=False).head(25)
    print("\n== worst 25 datasets by mean regret ==")
    print(worst.to_string(index=False))

    print("\n== DAMI detail (per dataset, mean over seeds) ==")
    print(dd[dd.modality == "dami"].sort_values("regret", ascending=False).to_string(index=False))

    # DAMI: per-regime regret means
    dcols = [c for c in df.columns if c.startswith("regret_")]
    print("\n== DAMI mean per-regime top-1 regret ==")
    print(df[df.modality == "dami"].groupby("dataset")[dcols + ["regret"]].mean().round(3).to_string())

    rcols = [c for c in df.columns if c.startswith("rho_")]
    print("\n== DAMI mean per-regime Spearman rho (pseudo vs true) ==")
    print(df[df.modality == "dami"].groupby("dataset")[rcols].mean().round(2).to_string())

    print("\n== tabular datasets with regret > 0.01 (mean over seeds) ==")
    tb = dd[(dd.modality == "tabular") & (dd.regret > 0.01)].sort_values("regret", ascending=False)
    print(tb.to_string(index=False))


if __name__ == "__main__":
    main()
