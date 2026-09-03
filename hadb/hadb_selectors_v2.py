# -*- coding: utf-8 -*-
"""HADB selector comparison round 2: reference baselines + the label-free literature criteria.

Each label-free selector computes its criterion on the VALIDATION scores (normals only, no
leak) and picks one detector per dataset; the pick's quality is the regret of its TEST ap_norm
against the label-derived best. Criterion columns (em, mv, consensus, model_centrality, hits)
are validation-side; auc/ap/ap_norm are test-side.

  selector        rule
  oracle_best     the label-derived best (regret 0; sanity anchor)
  anti_oracle     the worst (upper bound; sanity anchor)
  random          expected uniform pick (regret = best - mean; norm_rank 0.5)
  global_fixed    single best-mean detector per modality, applied everywhere (label-cheating
                  reference bar, not deployable)
  iforest_random  random IForest config
  em / mv         Goix Excess-Mass (argmax) / Mass-Volume (argmin) on validation
  consensus       UDR: agreement with the pool consensus on validation (argmax)
  model_centrality  centrality in the validation agreement graph (argmax)
  hits            HITS authority on the validation agreement graph (argmax)

INVARIANTS (stated up front): oracle==0; anti_oracle>=all others per dataset; random
norm_rank~0.5; global_fixed<=random regret.
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
ARMS = {"adbench_dami": "hadb_v3.csv", "oddbench": "hadb_oddbench.csv",
        "ovrbench": "hadb_ovrbench.csv", "ucr": "hadb_ts_ucr.csv",
        "tsbad_u": "hadb_ts_tsbad.csv", "tsbad_m": "hadb_ts_mts.csv"}
CRIT = ["em", "mv", "consensus", "model_centrality", "hits"]

rows = []
for corp, f in ARMS.items():
    D = pd.read_csv(os.path.join(S, f))
    D = D[D.dataset.isin(inc_by_corpus.get(corp, set()))].copy()
    if not len(D):
        continue
    D["ap_norm"] = (D.ap - D.base_rate) / (1 - D.base_rate)
    D["modality"] = "tabular" if corp in ("adbench_dami", "oddbench", "ovrbench") else "timeseries"
    D["corpus"] = corp
    rows.append(D)
A = pd.concat(rows, ignore_index=True)

# per (dataset, variant): mean ap_norm (test) and mean criteria (validation)
agg = {"ap_norm": "mean"}
agg.update({c: "mean" for c in CRIT})
g = A.groupby(["modality", "corpus", "dataset", "variant"], as_index=False).agg(agg)

# dataset -> dedup group, to weight the paired test by effective N
grp = inc.set_index("dataset").dedup_group.to_dict()

# global fixed per modality
global_fixed = {}
for mod in g.modality.unique():
    global_fixed[mod] = g[g.modality == mod].groupby("variant").ap_norm.mean().idxmax()
print("global_fixed per modality:", global_fixed)

def is_iforest(v):
    return str(v).startswith(("IForest", "IF_"))

recs = []
for (mod, corp, ds), gd in g.groupby(["modality", "corpus", "dataset"]):
    ap = gd.set_index("variant").ap_norm
    pool = len(ap)
    best = ap.max()
    order = ap.sort_values(ascending=False)
    rankpos = {v: i for i, v in enumerate(order.index)}

    def add(name, picked_variant):
        pv = picked_variant
        recs.append(dict(dataset=ds, modality=mod, corpus=corp, selector=name, pool=pool,
                         dedup_group=grp.get(ds, ds),
                         regret=best - ap[pv], norm_rank=rankpos[pv] / (pool - 1) if pool > 1 else 0.0))

    recs.append(dict(dataset=ds, modality=mod, corpus=corp, selector="oracle_best", pool=pool,
                     dedup_group=grp.get(ds, ds), regret=0.0, norm_rank=0.0))
    recs.append(dict(dataset=ds, modality=mod, corpus=corp, selector="anti_oracle", pool=pool,
                     dedup_group=grp.get(ds, ds), regret=best - ap.min(), norm_rank=1.0))
    recs.append(dict(dataset=ds, modality=mod, corpus=corp, selector="random", pool=pool,
                     dedup_group=grp.get(ds, ds), regret=best - ap.mean(), norm_rank=0.5))
    vf = global_fixed[mod]
    if vf in ap.index:
        add("global_fixed", vf)
    ifs = [v for v in ap.index if is_iforest(v)]
    if ifs:
        recs.append(dict(dataset=ds, modality=mod, corpus=corp, selector="iforest_random",
                         pool=pool, dedup_group=grp.get(ds, ds),
                         regret=best - ap[ifs].mean(),
                         norm_rank=float(np.mean([rankpos[v] for v in ifs]) / (pool - 1))))
    # label-free criteria (validation-side); argmax except MV which is argmin
    gv = gd.set_index("variant")
    for crit in CRIT:
        s = gv[crit].dropna()
        if len(s) < 3:
            continue
        pick = s.idxmin() if crit == "mv" else s.idxmax()
        add(crit, pick)

R = pd.DataFrame(recs)
R.to_csv(os.path.join(S, "HADB_SELECTORS_V2.csv"), index=False)

# invariants
print("\n=== INVARIANTS ===")
print(f"  I1 oracle==0        : {'PASS' if R[R.selector=='oracle_best'].regret.abs().max()<1e-9 else 'FAIL'}")
anti = R[R.selector=='anti_oracle'].set_index('dataset').regret
bad = 0
for sel in R.selector.unique():
    if sel == 'anti_oracle': continue
    s = R[R.selector==sel].set_index('dataset').regret
    common = s.index.intersection(anti.index)
    bad += int((s.loc[common] > anti.loc[common] + 1e-9).sum())
print(f"  I2 anti>=all others : {'PASS' if bad==0 else f'FAIL ({bad})'}")
rr = R[R.selector=='random'].norm_rank.mean()
print(f"  I3 random rank ~0.5 : {rr:.3f} {'PASS' if abs(rr-0.5)<0.02 else 'FAIL'}")
gf = R[R.selector=='global_fixed'].regret.mean(); rnd = R[R.selector=='random'].regret.mean()
print(f"  I4 global<=random   : {gf:.4f} vs {rnd:.4f} {'PASS' if gf<=rnd else 'FAIL'}")

# results, Holm-corrected vs random over dedup groups
def collapse(sub):
    return sub.groupby("dedup_group").regret.mean()

def table(sub, title):
    print(f"\n--- {title} (n_datasets={sub.dataset.nunique()}, "
          f"effN={sub.dedup_group.nunique()}) ---")
    rnd_g = collapse(sub[sub.selector=="random"])
    out = []
    order = ["oracle_best","global_fixed","em","mv","consensus","model_centrality","hits",
             "iforest_random","random","anti_oracle"]
    pvals = {}
    for sel in order:
        s = sub[sub.selector==sel]
        if not len(s): continue
        row = dict(selector=sel, regret=round(s.regret.mean(),4),
                   median=round(s.regret.median(),4),
                   norm_rank=round(s.norm_rank.mean(),3),
                   pct_opt=round(100*(s.regret<1e-9).mean(),1))
        if sel not in ("oracle_best","random","anti_oracle"):
            sg = collapse(s); common = sg.index.intersection(rnd_g.index)
            if len(common) > 5 and (sg.loc[common]!=rnd_g.loc[common]).any():
                try: pvals[sel] = wilcoxon(sg.loc[common], rnd_g.loc[common]).pvalue
                except Exception: pvals[sel] = np.nan
        out.append(row)
    # Holm correction across the deployable label-free selectors
    fam = {k:v for k,v in pvals.items() if k in CRIT+["iforest_random"]}
    holm = {}
    if fam:
        srt = sorted(fam.items(), key=lambda kv: (np.nan_to_num(kv[1],nan=1)))
        m = len(srt)
        for i,(k,p) in enumerate(srt):
            holm[k] = min(1.0, (m-i)*p) if p==p else np.nan
    T = pd.DataFrame(out)
    T["p_vs_random"] = T.selector.map(lambda s: round(pvals[s],4) if s in pvals and pvals[s]==pvals[s] else "")
    T["p_holm"] = T.selector.map(lambda s: round(holm[s],4) if s in holm and holm[s]==holm[s] else "")
    T["beats_random"] = T.selector.map(lambda s: "YES" if s in holm and holm[s]==holm[s] and holm[s]<0.05
                                       and sub[sub.selector==s].regret.mean() < rnd_g.mean() else "")
    print(T.to_string(index=False))

print("\n" + "="*80)
print("SELECTOR COMPARISON (regret on ap_norm; VALIDATION-side selection, TEST-side scoring)")
print("="*80)
table(R, "OVERALL")
table(R[R.modality=="tabular"], "tabular")
table(R[R.modality=="timeseries"], "timeseries")
print("\n  written HADB_SELECTORS_V2.csv")
