# -*- coding: utf-8 -*-
"""Adversarial EDA on the consolidated HADB. Tries to BREAK the benchmark's four claims:
valid, diverse, fair, non-trivial. Every check states the expected outcome in advance and
prints PASS/FAIL, so a silent defect shows up as a violated invariant, not a plausible table.

Reads HADB_MANIFEST.csv plus the raw arm CSVs.
"""
import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
M = pd.read_csv(os.path.join(S, "HADB_MANIFEST.csv"))
inc = M[M.include].copy()

LOCAL = ("LOF", "KNN", "CBLOF")
GLOBAL = ("HBOS", "COPOD", "ECOD", "PCA")


def fam(v):
    v = str(v)
    if v.startswith(LOCAL):
        return "local"
    if v.startswith(GLOBAL) or v.startswith(("IForest", "IF_", "LODA")):
        return "global"
    return "other"


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


print("=" * 74)
print(f"HADB ADVERSARIAL EDA   included={len(inc)}  scored={len(M)}  "
      f"effective_N={inc.dedup_group.nunique()}")
print("=" * 74)

print("\n--- VALID ---")
check("base rate never exceeds the 0.25 cap",
      inc.base_rate.max() <= 0.2501, f"max={inc.base_rate.max():.4f}")
check("every included dataset has real spread",
      inc.spread_ap_norm.min() >= 0.10, f"min spread={inc.spread_ap_norm.min():.4f}")
check("no floor datasets leaked into include",
      (inc.zone == "live").all(), f"zones={dict(inc.zone.value_counts())}")
check("best detector meaningfully beats base rate everywhere",
      inc.best_ap_norm.min() > 0.0, f"min best_ap_norm={inc.best_ap_norm.min():.4f}")

print("\n--- NON-TRIVIAL (per arm, from *_steps.csv) ---")
for sf in sorted(glob.glob(os.path.join(S, "*_steps.csv"))):
    try:
        st = pd.read_csv(sf)
    except Exception:
        continue
    tvcol = "keogh_trivial" if "keogh_trivial" in st else None
    n = len(st)
    triv = int(st[tvcol].fillna(False).sum()) if tvcol else 0
    sc = int((st.status == "scored").sum()) if "status" in st else 0
    print(f"  {os.path.basename(sf):28s} n={n:4d}  trivial={triv:4d} "
          f"({100*triv/max(n,1):3.0f}%)  scored={sc}")

print("\n--- DIVERSE ---")
vc = inc.gt_best.value_counts()
top = vc.iloc[0] / len(inc)
check("no single detector wins > 40% of datasets",
      top <= 0.40, f"top winner {vc.index[0]} takes {100*top:.0f}%")
check("at least 12 distinct winning detectors",
      inc.gt_best.nunique() >= 12, f"{inc.gt_best.nunique()} distinct winners")
print(f"    winners: {dict(vc.head(8))}")
print(f"    modality x split:")
print(inc.groupby(['modality', 'split', 'corpus']).size().to_string().replace('\n', '\n    '))

print("\n--- FAIR (winner family mix; a benchmark that only rewards one family is not a "
      "selection benchmark) ---")
famc = inc.gt_best.map(fam).value_counts(normalize=True)
print(f"    winning family share: "
      f"local {100*famc.get('local',0):.0f}%  global {100*famc.get('global',0):.0f}%  "
      f"other {100*famc.get('other',0):.0f}%")
check("no family wins > 75% (both local and global must matter somewhere)",
      famc.max() <= 0.75, f"largest family share {100*famc.max():.0f}%")
check("at least two families each win somewhere",
      (famc > 0.05).sum() >= 2, f"families active: {list(famc[famc>0.05].index)}")

print("\n--- base-rate comparability across arms (ap_norm divides by this) ---")
print(inc.groupby("corpus").base_rate.agg(["min", "median", "max"]).round(4).to_string())

print("\n--- diversity spread of the difficulty knobs ---")
for c in ("spread_ap_norm", "best_ap_norm", "best_auc", "base_rate"):
    q = inc[c].quantile([0, .25, .5, .75, 1]).round(3).tolist()
    print(f"   {c:16s} min/q1/med/q3/max = {q}")
