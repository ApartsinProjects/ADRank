# -*- coding: utf-8 -*-
"""HADB consolidation. Rewritten after the adversarial audit found three defects here.

  BUG 1 (was critical). ARMS listed only three files, and the time-series entry pointed at
     output from the DEPRECATED transductive script, whose schema the consolidator could not
     tell apart from the normals-only one. The UCR, OvrBench and multivariate arms were not
     listed at all, so re-running could never see them. Fixed: every arm is listed, and a
     SCHEMA GUARD rejects any input lacking `seed`, which is the cheap signature of the old
     generation. An arm whose file is missing is reported, not silently skipped.

  BUG 6 (was moderate). Zone and inclusion were decided TWICE on TWO metrics: each arm
     script judged on AUC, the consolidator re-judged on ap_norm spread with an AUC-only
     best. Dev inclusion was 13 datasets under one rule and 27 under the other, so the
     headline count depended on which file you read. The mixed-metric ceiling rule
     (best_auc >= 0.95 AND ap_norm spread < 0.03) fired 0 times in 124 rows: dead code.
     Fixed: zone and inclusion are decided ONCE, HERE, on the PRIMARY metric alone. Arm
     scripts emit rows; they do not decide membership.

  BUG 8 (was latent). The dedup "fingerprint confirmation" compared only dimensionality and
     ignored the moments it computed, and for any corpus outside adbench/dami the
     fingerprint was None for every member, where the code treated absent as CONFIRMED and
     merged on NAME ALONE. FootballBetting exists in OddBench at d=4 and OvrBench at d=7:
     different data, one dedup group, one of them silently dropped from the effective count.
     Fixed: the full tuple must match, fingerprints are computed for every tabular corpus,
     and absent evidence blocks a merge instead of granting one.
"""
import os, re, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
OUT = os.path.join(S, "HADB_MANIFEST.csv")

# BUG 1: every arm, explicitly. modality, split, filename, corpus label.
ARMS = [("tabular",    "dev",  "hadb_v3.csv",        "adbench_dami"),
        ("tabular",    "test", "hadb_oddbench.csv",  "oddbench"),
        ("tabular",    "test", "hadb_ovrbench.csv",  "ovrbench"),
        ("timeseries", "dev",  "hadb_ts_tsbad.csv",  "tsbad_u"),
        ("timeseries", "test", "hadb_ts_ucr.csv",    "ucr"),
        ("timeseries", "test", "hadb_ts_mts.csv",    "tsbad_m")]
REQUIRED = {"dataset", "variant", "seed", "auc", "ap", "base_rate"}

PRIMARY = "ap_norm"
# BUG 6: one metric, one rule, one place. FLOOR means no detector meaningfully beats the
# base rate, so the observed "best" is mostly noise and the ground-truth label is unreliable.
# Everything else is LIVE, and inclusion additionally requires real spread.
#
# There is deliberately NO ceiling zone. The audit found the old mixed-metric ceiling rule
# fired 0 times in 124 rows, and restating it on ap_norm fires 0 times too - for a good
# reason rather than a tuning accident: "too easy" means EVERY detector does well, which is
# exactly a small spread, and MIN_SPREAD already excludes that. A dataset whose best detector
# is perfect while the field trails it has a large spread and selection genuinely matters
# there, so it belongs in. Keeping a branch that can never fire would misreport the filter
# as doing work it does not do.
FLOOR_BEST, MIN_SPREAD = 0.10, 0.10

# Which data dirs back each corpus label. Resolving per-corpus matters: a name shared by two
# corpora (FootballBetting in oddbench AND ovrbench) must fingerprint from its OWN corpus, or
# both would resolve to the same file and be wrongly merged.
CORP_DIRS = {"adbench_dami": ("adbench", "dami"), "oddbench": ("oddbench",),
             "ovrbench": ("ovrbench",)}


def norm_name(s):
    s = re.sub(r"\.(npz|csv|txt)$", "", str(s))
    s = re.sub(r"^\d+_", "", s)
    return s.lower().replace("_", "").replace("-", "")


def fingerprint(name, corpus):
    """Dimensionality d of the raw features, resolved in the dataset's OWN corpus dir, else
    None. Dimensionality is the merge key, NOT the moments: adbench and dami redistribute the
    same classic datasets (Stamps, Waveform, Wilt) with DIFFERENT normalisation, so their
    moments differ while they are the same underlying data and must merge for effective-N.
    A name collision across unrelated corpora (FootballBetting: oddbench d=4 vs ovrbench d=7)
    shows up as different d and is correctly kept apart."""
    dirs = CORP_DIRS.get(corpus)
    if not dirs:
        return None
    base = re.sub(r"\.(npz|csv|txt)$", "", str(name))
    for sub in dirs:
        p = os.path.join(ROOT, "data", sub, base + ".npz")
        if not os.path.exists(p):
            continue
        try:
            d = np.load(p, allow_pickle=True)
            if "X" in d:
                X = np.asarray(d["X"], float)
            elif "train" in d and "test" in d:
                X = np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)])
            else:
                return None
            return int(X.shape[1])
        except Exception:
            return None
    return None


frames, missing = [], []
for modality, split, fn, corpus in ARMS:
    p = os.path.join(S, fn)
    if not os.path.exists(p):
        missing.append(fn); continue
    d = pd.read_csv(p)
    if not len(d):
        missing.append(fn + " (empty)"); continue
    lack = REQUIRED - set(d.columns)
    if lack:                                     # BUG 1: generation guard
        print(f"  [REJECT] {fn}: missing {sorted(lack)} - stale generation, re-run the arm")
        missing.append(fn + " (stale schema)"); continue
    d["modality"], d["split"] = modality, split
    d["corpus"] = corpus
    frames.append(d)
    print(f"  [load] {fn:24s} {len(d):6d} rows  {d.dataset.nunique():4d} datasets")

if missing:
    print(f"  [MISSING/REJECTED] {', '.join(missing)}")
if not frames:
    sys.exit("no usable arms")
D = pd.concat(frames, ignore_index=True)
D["ap_norm"] = (D["ap"] - D["base_rate"]) / (1 - D["base_rate"])

# ---------------- dedup groups ----------------
meta = D.groupby(["dataset", "modality", "split", "corpus"], as_index=False).size()
meta["key"] = meta.dataset.map(norm_name)
meta["fp"] = [fingerprint(n, c) if m == "tabular" else None
              for n, c, m in zip(meta.dataset, meta.corpus, meta.modality)]

groups, gid, merged = {}, 0, []
for key, g in meta.groupby("key"):
    if len(g) == 1:
        groups[g.dataset.iloc[0]] = f"g{gid}"; gid += 1; continue
    fps = list(g.fp)
    # BUG 8: merge only on POSITIVE evidence - every member fingerprinted, same dimensionality.
    # Same name + same d = the same underlying dataset (normalisation aside) -> one group,
    # the conservative choice for effective-N. Different d (or a missing fingerprint) blocks
    # the merge, since that is a genuine name collision between unrelated datasets.
    confirmed = all(f is not None for f in fps) and len(set(fps)) == 1
    if confirmed:
        for n in g.dataset:
            groups[n] = f"g{gid}"
        merged.append((key, list(g.dataset), fps[0])); gid += 1
    else:
        for n in g.dataset:
            groups[n] = f"g{gid}"; gid += 1
        if len([f for f in fps if f is not None]) == len(fps) and len(set(fps)) > 1:
            print(f"  [NOT merged] '{key}': {list(zip(g.dataset, g.corpus))} d={fps} differ "
                  f"-> distinct datasets sharing a name")

for key, names, fp in merged:
    print(f"  [dedup] '{key}': {names} -> one group  (same underlying data, d={fp})")

# ---------------- manifest ----------------
rows = []
for (ds, modality, split, corpus), g in D.groupby(["dataset", "modality", "split", "corpus"]):
    per_var = g.groupby("variant")[PRIMARY].mean()
    if len(per_var) < 5:
        continue
    best, mean_, worst = per_var.max(), per_var.mean(), per_var.min()
    spread = best - mean_
    zone = "floor" if best < FLOOR_BEST else "live"
    rank = per_var.sort_values(ascending=False)
    rows.append(dict(
        dataset=ds, modality=modality, split=split, corpus=corpus,
        dedup_group=groups.get(ds, "?"), n_variants=len(per_var),
        base_rate=round(float(g.base_rate.mean()), 4),
        gt_best=rank.index[0], gt_second=rank.index[1] if len(rank) > 1 else None,
        best_ap_norm=round(float(best), 4), mean_ap_norm=round(float(mean_), 4),
        worst_ap_norm=round(float(worst), 4), spread_ap_norm=round(float(spread), 4),
        best_auc=round(float(g.groupby("variant")["auc"].mean().max()), 4),
        zone=zone, gt_ranking="|".join(rank.index[:5])))
M = pd.DataFrame(rows)
# Safety net: a per-dataset base rate above the cap means an arm let a too-small test set
# through uncapped (the FIX 6b guard now drops those seeds, but this catches any that were
# scored before the guard was tightened, so no arm needs re-running just for a rare leaker).
MAX_TEST_RATE = 0.25
over = M[M.base_rate > MAX_TEST_RATE + 1e-6]
if len(over):
    print(f"  [rate net] dropping {len(over)} dataset(s) over base_rate {MAX_TEST_RATE}: "
          f"{dict(zip(over.dataset, over.base_rate.round(3)))}")
# PER-DATASET TRIVIALITY-RULE FILTER (tabular): the max|z| anomaly filter is per-datapoint and
# misses datasets whose survivors are still separable by a simple per-feature rule. Drop tabular
# datasets where the max|z| rule OR the HBOS-lite histogram rule reaches test AUC > TRIV_RULE_CUT
# (run hadb_trivial_rules.py first). Histogram catches the skewed/bimodal/categorical rarity the
# Gaussian max|z| misses. Time series keep the Wu-Keogh one-liner criterion (already per-series).
TRIV_RULE_CUT = 0.85
# Combine the tabular (feature) and time-series (window-feature) rule audits. Both check whether
# a dataset's surviving hard anomalies are still separable by a simple per-feature rule - max|z|
# (Gaussian) OR HBOS-lite (empirical histogram) - at test AUC > cut, in the space the DETECTORS
# use. Tabular: hadb_trivial_rules.py; time series (uni + multivariate): ts_trivial_rules.py.
frames = []
for pth in ("HADB_TRIVIAL_RULES.csv", "HADB_TS_TRIVIAL_RULES.csv"):
    fp = os.path.join(S, pth)
    if os.path.exists(fp):
        frames.append(pd.read_csv(fp)[["dataset", "corpus", "mz_rule_auc", "hbos_rule_auc"]])
if frames:
    TR = pd.concat(frames, ignore_index=True)
    TR["rule_auc"] = TR[["mz_rule_auc", "hbos_rule_auc"]].max(1)
    M = M.merge(TR, on=["dataset", "corpus"], how="left")
    M["rule_trivial"] = M.rule_auc > TRIV_RULE_CUT           # applies to ALL modalities now
    print(f"  [triviality-rule net] {int(M.rule_trivial.sum())} datasets solvable by a simple "
          f"per-feature rule (AUC>{TRIV_RULE_CUT}) will be excluded "
          f"({dict(M[M.rule_trivial.fillna(False)].modality.value_counts())})")
else:
    print("  [triviality-rule net] rule-audit CSVs not found - run hadb_trivial_rules.py + "
          "ts_trivial_rules.py; skipping the per-dataset rule filter")
    M["rule_trivial"] = False
M["include"] = ((M.zone == "live") & (M.spread_ap_norm >= MIN_SPREAD)
                & (M.base_rate <= MAX_TEST_RATE + 1e-6) & ~M.rule_trivial)
M.to_csv(OUT, index=False)

inc = M[M.include]
n_ds, n_eff = len(inc), inc.dedup_group.nunique()
print("\n" + "=" * 72)
print("HADB CONSOLIDATED")
print("=" * 72)
print(f"  scored datasets : {len(M)}")
print(f"  INCLUDED (live, spread_{PRIMARY} >= {MIN_SPREAD}) : {n_ds}")
print(f"  EFFECTIVE N (dedup groups) : {n_eff}"
      f"{'  <- correlated members collapsed' if n_eff < n_ds else ''}")
print(f"  zones (all scored): {M.zone.value_counts().to_dict()}")
if len(inc):
    print(f"\n  by modality / split / corpus:")
    print(inc.groupby(["modality", "split", "corpus"]).size().to_string())
    print(f"\n  --- does ONE detector win everywhere? (the acid test) ---")
    vc = inc.gt_best.value_counts()
    print(vc.head(8).to_string())
    print(f"  distinct winners: {inc.gt_best.nunique()} over {n_ds} datasets; "
          f"top takes {100*vc.iloc[0]/n_ds:.0f}%")
    if vc.iloc[0] / n_ds > 0.5:
        print("  !! one detector wins >50% - little to select, benchmark weak")
    print(f"\n  --- base-rate comparability across arms (ap_norm divides by this) ---")
    print(inc.groupby("corpus").base_rate.agg(["min", "median", "max"]).round(4).to_string())
    print(f"\n  --- diversity ---")
    for c in ("spread_ap_norm", "best_ap_norm", "best_auc"):
        q = inc[c].quantile([0, .25, .5, .75, 1]).round(3).tolist()
        print(f"   {c:16s} min/q1/med/q3/max = {q}")
