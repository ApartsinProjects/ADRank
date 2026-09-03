# -*- coding: utf-8 -*-
"""HADB deep review: formats, the full filter funnel per corpus, per-dataset statistics,
distributions, and redundancy/coverage analysis. Prints a structured report and exports
HADB_INCLUDED.csv (the per-dataset table) and HADB_FUNNEL.csv.
"""
import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 40)
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"

M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
inc = M[M.include].copy()

# arm -> (corpus label in manifest, steps file, modality)
ARMS = [
    ("adbench_dami", "hadb_v3_steps.csv", "tabular"),
    ("oddbench",     "hadb_oddbench_steps.csv", "tabular"),
    ("ovrbench",     "hadb_ovrbench_steps.csv", "tabular"),
    ("tsbad_u",      "hadb_ts_tsbad_steps.csv", "timeseries"),
    ("ucr",          "hadb_ts_ucr_steps.csv", "timeseries"),
    ("tsbad_m",      "hadb_ts_mts_steps.csv", "timeseries"),
]


def drop_bucket(reason):
    r = str(reason)
    if r in ("too_small_or_degenerate", "too_few_anomalies"):
        return "too_small"
    if r == "rate_too_high":
        return "rate_high"
    if r in ("keogh_trivial", "trivial_calibrated"):
        return "trivial(series)"
    if r.startswith("only_") and "double_hard" in r:
        return "too_few_hard(all_trivial)"
    if r == "too_short_or_dense":
        return "too_short/dense"
    if r in ("no_variant_ran", "window", "window_align"):
        return "no_valid_score"
    return r


print("#" * 78)
print("# HADB DEEP REVIEW")
print("#" * 78)

# -------------------------------------------------------------- 1. FILTER FUNNEL
print("\n" + "=" * 78)
print("1. FILTER FUNNEL  (entered -> dropped by stage -> scored -> floor -> included)")
print("=" * 78)
funnel_rows = []
for corpus, sf, modality in ARMS:
    st = pd.read_csv(os.path.join(S, sf))
    entered = len(st)
    drops = st[st.status == "dropped"].reason.map(drop_bucket).value_counts().to_dict() \
        if "status" in st else {}
    scored = int((st.status == "scored").sum()) if "status" in st else len(st)
    mm = M[M.corpus == corpus]
    n_manifest = len(mm)                      # scored AND >=5 variants (enters manifest)
    floor = int((mm.zone == "floor").sum())
    low_spread = int(((mm.zone == "live") & (mm.spread_ap_norm < 0.10)).sum())
    rate_net = int((mm.base_rate > 0.2501).sum())
    included = int(mm.include.sum())
    row = dict(corpus=corpus, modality=modality, entered=entered, **drops,
               scored=scored, in_manifest=n_manifest, floor=floor,
               low_spread=low_spread, rate_net=rate_net, included=included)
    funnel_rows.append(row)
F = pd.DataFrame(funnel_rows).fillna(0)
intcols = [c for c in F.columns if c not in ("corpus", "modality")]
F[intcols] = F[intcols].astype(int)
print(F.to_string(index=False))
F.to_csv(os.path.join(S, "HADB_FUNNEL.csv"), index=False)
print(f"\n  TOTAL entered={F.entered.sum()}  scored={F.scored.sum()}  "
      f"included={F.included.sum()}  (yield {100*F.included.sum()/F.entered.sum():.0f}%)")

# -------------------------------------------------------------- 2. PER-DATASET TABLE
print("\n" + "=" * 78)
print("2. PER-DATASET STATISTICS (included). Full table -> HADB_INCLUDED.csv")
print("=" * 78)
cols = ["dataset", "modality", "corpus", "n_variants", "base_rate",
        "best_ap_norm", "mean_ap_norm", "spread_ap_norm", "best_auc", "gt_best"]
T = inc[cols].sort_values(["modality", "corpus", "spread_ap_norm"],
                          ascending=[True, True, False]).reset_index(drop=True)
T.to_csv(os.path.join(S, "HADB_INCLUDED.csv"), index=False)
print(f"  {len(T)} datasets written. Hardest 10 (lowest best_ap_norm) and easiest 5:")
print(T.sort_values("best_ap_norm").head(10).to_string(index=False))
print("  ...")
print(T.sort_values("best_ap_norm").tail(5).to_string(index=False))

# -------------------------------------------------------------- 3. DISTRIBUTIONS
print("\n" + "=" * 78)
print("3. DISTRIBUTIONS  (quantiles: min / q1 / median / q3 / max)")
print("=" * 78)
def qline(s, label):
    q = s.quantile([0, .25, .5, .75, 1]).round(4).tolist()
    print(f"  {label:22s} {q}")
print("  --- overall ---")
for c in ["base_rate", "best_ap_norm", "spread_ap_norm", "best_auc", "n_variants"]:
    qline(inc[c], c)
for mod in ["tabular", "timeseries"]:
    print(f"  --- {mod} ---")
    sub = inc[inc.modality == mod]
    for c in ["base_rate", "best_ap_norm", "spread_ap_norm", "best_auc"]:
        qline(sub[c], c)

# -------------------------------------------------------------- 4. DIMENSIONALITY / SIZE
print("\n" + "=" * 78)
print("4. SIZE & DIMENSIONALITY COVERAGE (from steps files)")
print("=" * 78)
for corpus, sf, modality in ARMS:
    st = pd.read_csv(os.path.join(S, sf))
    sc = st[st.status == "scored"] if "status" in st else st
    inc_names = set(inc[inc.corpus == corpus].dataset)
    sc = sc[sc.dataset.isin(inc_names)]
    if not len(sc):
        continue
    if modality == "tabular":
        print(f"  {corpus:14s} n: {int(sc.n.min())}-{int(sc.n.max())} "
              f"(med {int(sc.n.median())})   d: {int(sc.d.min())}-{int(sc.d.max())} "
              f"(med {int(sc.d.median())})   hard: {int(sc.n_hard.min())}-{int(sc.n_hard.max())}"
              if 'n_hard' in sc else f"  {corpus}")
    else:
        chcol = "n_ch" if "n_ch" in sc else None
        ch = f"   channels: {int(sc.n_ch.min())}-{int(sc.n_ch.max())}" if chcol else ""
        print(f"  {corpus:14s} len: {int(sc.n.min())}-{int(sc.n.max())} "
              f"(med {int(sc.n.median())})   anom_win: {int(sc.n_anom_win.min())}-"
              f"{int(sc.n_anom_win.max())}{ch}")

# -------------------------------------------------------------- 5. CANDIDATE POOL
print("\n" + "=" * 78)
print("5. CANDIDATE POOL per dataset (selection is over these)")
print("=" * 78)
print(f"  n_variants per dataset: {dict(inc.n_variants.value_counts().sort_index())}")
print(f"  (time-series arms use a smaller PyOD pool than tabular; a selection method ranks")
print(f"   whatever candidates a dataset lists, so differing pools across arms is expected)")

# -------------------------------------------------------------- 6. REDUNDANCY / CORRELATION
print("\n" + "=" * 78)
print("6. ARE THE DIFFICULTY KNOBS INDEPENDENT?  (Pearson r among dataset properties)")
print("=" * 78)
cc = inc[["base_rate", "best_ap_norm", "spread_ap_norm", "best_auc", "n_variants"]].corr().round(2)
print(cc.to_string())
print("\n  interpretation: |r| near 0 means the property adds independent difficulty signal;")
print("  a high |r| would mean two filters are secretly the same knob.")

# -------------------------------------------------------------- 7. WINNER STRUCTURE
print("\n" + "=" * 78)
print("7. WINNER STRUCTURE (the object a selector must predict)")
print("=" * 78)
vc = inc.gt_best.value_counts()
print(f"  distinct winners: {inc.gt_best.nunique()} over {len(inc)} datasets; "
      f"top {vc.index[0]} takes {100*vc.iloc[0]/len(inc):.0f}%")
print(f"  winner concentration (share taken by top-5): "
      f"{100*vc.head(5).sum()/len(inc):.0f}%")
print("  winners by modality:")
for mod in ["tabular", "timeseries"]:
    sub = inc[inc.modality == mod]
    v = sub.gt_best.value_counts()
    print(f"    {mod:11s}: {v.nunique()} distinct, top-3 {dict(v.head(3))}")
print("\n  done. exports: HADB_FUNNEL.csv, HADB_INCLUDED.csv")
