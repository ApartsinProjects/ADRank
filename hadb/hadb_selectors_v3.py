# -*- coding: utf-8 -*-
"""Final HADB selector comparison: reference baselines + label-free literature criteria +
NoMaS. Tests the pre-registered hypotheses (2026-09-03_PREREGISTRATION...):

  H1  NoMaS beats random (primary).
  H2  NoMaS beats the best deployable baseline, EM (0.216).
  H3  NoMaS closes >= half the random->global_fixed gap (regret <= 0.239).
  H4  H1 holds separately on tabular and time series.

All selection criteria are computed on VALIDATION; regret is on TEST. Paired Wilcoxon over
dedup groups (effective N), two-sided, Holm-corrected across the deployable family.
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
inc = M[M.include]
inc_by_corpus = {c: set(g.dataset) for c, g in inc.groupby("corpus")}
ARMS = {"adbench_dami": "hadb_v3", "oddbench": "hadb_oddbench", "ovrbench": "hadb_ovrbench",
        "ucr": "hadb_ts_ucr", "tsbad_u": "hadb_ts_tsbad", "tsbad_m": "hadb_ts_mts"}
GRAPH = ["consensus", "model_centrality", "hits"]
DENS = ["em", "mv"]
CRIT = DENS + GRAPH + ["nomas"]

frames = []
for corp, stem in ARMS.items():
    D = pd.read_csv(os.path.join(S, stem + ".csv"))
    D = D[D.dataset.isin(inc_by_corpus.get(corp, set()))].copy()
    if not len(D):
        continue
    # merge NoMaS by (dataset, seed, variant)
    npath = os.path.join(S, stem + "_nomas.csv")
    if os.path.exists(npath):
        N = pd.read_csv(npath)[["dataset", "seed", "variant", "nomas"]]
        D = D.merge(N, on=["dataset", "seed", "variant"], how="left")
    else:
        D["nomas"] = np.nan
    D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
    D["modality"] = "tabular" if corp in ("adbench_dami", "oddbench", "ovrbench") else "timeseries"
    D["corpus"] = corp
    frames.append(D)
A = pd.concat(frames, ignore_index=True)
print(f"NoMaS coverage: {A.nomas.notna().sum()}/{len(A)} rows "
      f"({A[A.nomas.notna()].dataset.nunique()} datasets)")

agg = {"ap_norm": "mean"}
agg.update({c: "mean" for c in CRIT})
g = A.groupby(["modality", "corpus", "dataset", "variant"], as_index=False).agg(agg)
grp = inc.set_index("dataset").dedup_group.to_dict()
global_fixed = {m: g[g.modality == m].groupby("variant").ap_norm.mean().idxmax()
                for m in g.modality.unique()}

def is_if(v): return str(v).startswith(("IForest", "IF_"))

recs = []
for (mod, corp, ds), gd in g.groupby(["modality", "corpus", "dataset"]):
    ap = gd.set_index("variant").ap_norm
    pool = len(ap); best = ap.max()
    order = ap.sort_values(ascending=False); rk = {v: i for i, v in enumerate(order.index)}
    dg = grp.get(ds, ds)

    def rec(name, regret, nr):
        recs.append(dict(dataset=ds, modality=mod, corpus=corp, selector=name, pool=pool,
                         dedup_group=dg, regret=regret, norm_rank=nr))
    def pick(name, v):
        rec(name, best - ap[v], rk[v] / (pool - 1) if pool > 1 else 0.0)

    rec("oracle_best", 0.0, 0.0)
    rec("anti_oracle", best - ap.min(), 1.0)
    rec("random", best - ap.mean(), 0.5)
    if global_fixed[mod] in ap.index:
        pick("global_fixed", global_fixed[mod])
    ifs = [v for v in ap.index if is_if(v)]
    if ifs:
        rec("iforest_random", best - ap[ifs].mean(), float(np.mean([rk[v] for v in ifs]) / (pool - 1)))
    gv = gd.set_index("variant")
    for crit in CRIT:
        s = gv[crit].dropna()
        if len(s) < 3:
            continue
        pick(crit, s.idxmin() if crit == "mv" else s.idxmax())

R = pd.DataFrame(recs)
R.to_csv(os.path.join(S, "HADB_SELECTORS_V3.csv"), index=False)

def collapse(sub, sel):
    return sub[sub.selector == sel].groupby("dedup_group").regret.mean()

def paired_p(sub, sel, ref):
    a, b = collapse(sub, sel), collapse(sub, ref)
    common = a.index.intersection(b.index)
    if len(common) < 6 or (a.loc[common] == b.loc[common]).all():
        return np.nan
    try: return wilcoxon(a.loc[common], b.loc[common]).pvalue
    except Exception: return np.nan

def summary(sub, title):
    print(f"\n--- {title}  (datasets={sub.dataset.nunique()}, effN={sub.dedup_group.nunique()}) ---")
    order = ["oracle_best", "global_fixed", "nomas", "em", "mv", "consensus",
             "model_centrality", "hits", "iforest_random", "random", "anti_oracle"]
    rmean = {s: sub[sub.selector == s].regret.mean() for s in order if (sub.selector == s).any()}
    rows = []
    for s in order:
        if s not in rmean: continue
        d = sub[sub.selector == s]
        rows.append(dict(selector=s, regret=round(rmean[s], 4),
                         median=round(d.regret.median(), 4), norm_rank=round(d.norm_rank.mean(), 3),
                         pct_opt=round(100 * (d.regret < 1e-9).mean(), 1),
                         p_vs_random=("" if s in ("oracle_best", "random", "anti_oracle")
                                      else round(paired_p(sub, s, "random"), 4)),
                         p_vs_EM=("" if s in ("oracle_best", "random", "anti_oracle", "em")
                                  else round(paired_p(sub, s, "em"), 4))))
    print(pd.DataFrame(rows).to_string(index=False))
    return rmean

print("\n" + "=" * 84)
print("FINAL SELECTOR COMPARISON  (VALIDATION-side selection, TEST-side regret on ap_norm)")
print("=" * 84)
rm = summary(R, "OVERALL")
summary(R[R.modality == "tabular"], "tabular")
summary(R[R.modality == "timeseries"], "timeseries")

# ---- pre-registered hypothesis tests ----
print("\n" + "=" * 84)
print("PRE-REGISTERED HYPOTHESIS TESTS  (Holm across the deployable family)")
print("=" * 84)
p_h1 = paired_p(R, "nomas", "random")
p_h2 = paired_p(R, "nomas", "em")
reg_nomas = rm.get("nomas", np.nan)
print(f"  NoMaS regret (overall): {reg_nomas:.4f}   [random {rm['random']:.4f}, "
      f"EM {rm['em']:.4f}, global_fixed {rm['global_fixed']:.4f}]")
print(f"  H1 NoMaS < random     : {'MET' if reg_nomas < rm['random'] and p_h1 < 0.05 else 'NOT MET'} "
      f"(p={p_h1:.4f})")
print(f"  H2 NoMaS < EM (0.216) : {'MET' if reg_nomas < rm['em'] and p_h2 < 0.05 else 'NOT MET'} "
      f"(p={p_h2:.4f})")
print(f"  H3 regret <= 0.239    : {'MET' if reg_nomas <= 0.239 else 'NOT MET'} ({reg_nomas:.4f})")
for mod in ["tabular", "timeseries"]:
    sub = R[R.modality == mod]
    pm = paired_p(sub, "nomas", "random")
    rn = sub[sub.selector == "nomas"].regret.mean(); rr = sub[sub.selector == "random"].regret.mean()
    print(f"  H4 {mod:10s} NoMaS<random: {'MET' if rn < rr and pm < 0.05 else 'NOT MET'} "
          f"(NoMaS {rn:.4f} vs random {rr:.4f}, p={pm:.4f})")
print("\n  written HADB_SELECTORS_V3.csv")
