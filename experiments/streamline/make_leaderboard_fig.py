# -*- coding: utf-8 -*-
"""Paper leaderboard figure: label-free selection on the streamlined HADB benchmark."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
S = r"E:\tmp\claude\E--Projects-Submitted-ADRank\236d6247-42ad-4e62-9828-db625dfa055d"
D = os.path.join(S, "scratchpad", "streamline")
rk = pd.read_csv(os.path.join(D, "STREAM_RANK.csv"))
ifr = pd.read_csv(os.path.join(D, "STREAM_IFR.csv"))[["corpus", "dataset", "reg_ifr"]]
udr = pd.read_csv(os.path.join(D, "STREAM_UDR.csv"))[["corpus", "dataset", "reg_udr"]]
m = rk.merge(ifr, on=["corpus", "dataset"], how="left").merge(udr, on=["corpus", "dataset"], how="left")
def macro(c): return np.mean([m[m.truefam == fa][c].mean() for fa in ["local", "global"] if (m.truefam == fa).any()])
rr = m.reg_random.values
METH = [("NoMaS matched (ours)", "reg_matched", "ours"), ("NoMaS \u03b2=1 (ours)", "reg_beta1", "ours"),
        ("MV", "reg_mv", "em"), ("EM", "reg_em", "em"),
        ("UDR", "reg_udr", "base"), ("IFOREST-R", "reg_ifr", "base"), ("HITS", "reg_hits", "base"),
        ("ModelCentrality", "reg_model_centrality", "base"), ("consensus/UDR-graph", "reg_consensus", "base"),
        ("random", "reg_random", "rand")]
data = []
for nm, c, k in METH:
    a = m[c].values; ok = ~np.isnan(a)
    p = wilcoxon(a[ok], rr[ok]).pvalue if c != "reg_random" and (a[ok] != rr[ok]).any() else np.nan
    data.append((nm, macro(c), k, p))
data.sort(key=lambda x: x[1])
CL = {"ours": "#2a7fb8", "em": "#d1495b", "base": "#b8bcc4", "rand": "#6b7280"}
fig, ax = plt.subplots(figsize=(10, 5.4)); y = np.arange(len(data))
ax.barh(y, [d[1] for d in data], color=[CL[d[2]] for d in data], edgecolor="white", height=0.72)
for i, (nm, val, k, p) in enumerate(data):
    star = "" if (p != p or k == "rand") else ("***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else " n.s.")))
    ax.text(val + 0.003, i, f"{val:.3f}{'' if k in ('rand',) else '  vs rand '+star}", va="center", fontsize=9)
ax.axvline(macro("reg_random"), color="#6b7280", ls="--", lw=1, alpha=0.7)
ax.text(macro("reg_random"), len(data) - 0.4, " random", color="#6b7280", fontsize=8, va="top")
ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data], fontsize=10); ax.invert_yaxis()
ax.set_xlabel("regret on ap_norm  (family-balanced macro; lower = better)"); ax.set_xlim(0, 0.27)
ax.set_title(f"Label-free detector selection on streamlined HADB ({len(m)} datasets)\n"
             "NoMaS beats EM and every UOMS baseline incl. the IFOREST-R bar (Ma et al.); ties MV", fontsize=12)
ax.legend(handles=[Patch(color=CL["ours"], label="NoMaS (ours)"), Patch(color=CL["em"], label="EM / MV"),
                   Patch(color=CL["base"], label="UOMS baselines (\u2248 random)")], fontsize=9, loc="lower right")
fig.tight_layout(); out = os.path.join(S, "FIG_stream_leaderboard.png"); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
