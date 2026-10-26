#!/usr/bin/env python3
"""Figure 1, from data/storms_items.csv. Writes results/figure1.pdf and results/figure1.png.

  A  Percentage of studies reporting each non-control item, in pipeline order.
  B  The eight methods items by publication year with 95% Wilson bands; dashed line = STORMS (Nov 2021).
  C  Each study's percentage of the eight methods items against journal impact factor, with OLS fit.
  D  Negative controls, positive controls and contaminant removal by specimen type, Fisher's exact test.

Usage:  python scripts/figure1.py
"""
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Rectangle
from scipy.stats import fisher_exact

from common import DATA, PAL as P, RESULTS, apply_style, panel_label, restyle, wilson

ITEM = {
    "participant_n":       ("Participant N",       "Participant N",   P["blue"]),
    "final_sample_n":      ("Final sample N",      "Final sample N",  P["purple_mid"]),
    "batch_effects":       ("Batch effects",       "Batch effects",   P["orange"]),
    "sequencing_depth":    ("Sequencing depth",    "Seq. depth",      P["rust"]),
    "sequences_after_qc":  ("Sequences after QC",  "Seqs after QC",   P["orange_dark"]),
    "taxa_removal":        ("Taxa removal",        "Taxa removal",    P["purple_light"]),
    "contaminant_removal": ("Contaminant removal", "Contam. removal", P["blue_dark"]),
    "read_normalization":  ("Read normalization",  "Read norm.",      P["purple"]),
    "negative_controls":   ("Negative controls",   "Neg. controls",   P["teal"]),
    "positive_controls":   ("Positive controls",   "Pos. controls",   P["teal_dark"]),
}
A_PRE = ["participant_n", "final_sample_n", "batch_effects"]
A_POST = ["sequencing_depth", "sequences_after_qc", "taxa_removal", "contaminant_removal", "read_normalization"]
B_ITEMS = ["batch_effects", "sequencing_depth", "sequences_after_qc", "read_normalization",
           "taxa_removal", "contaminant_removal", "negative_controls", "positive_controls"]
D_ITEMS = ["negative_controls", "positive_controls", "contaminant_removal"]
YEAR_BINS = [(2014, 2020), (2021, 2021), (2022, 2022), (2023, 2023), (2024, 2024), (2025, 2025)]
YEAR_TICKS = ["2014–20", "2021", "2022", "2023", "2024", "2025"]
IF_TICKS = [2, 5, 10, 20, 50]
GROUPS = [(1, "Low biomass (tissue, fluid, swab)", P["orange"]), (0, "Stool / includes stool", P["blue"])]


def pct_ci(k, n):
    lo, hi = wilson(int(k), int(n))
    return 100 * k / n, 100 * lo, 100 * hi


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"


def draw_a(ax, d):
    keys = A_PRE + A_POST
    m, split = len(keys), len(A_PRE)
    W, xr, xband, xlab = 0.78, 52.0, 46.0, -57.0
    yb, ylo, yhi = m - split - 0.5, -0.62, m - 0.38
    for (lab, col), (a, b) in zip([("Before sequencing", P["teal"]), ("After sequencing", P["purple"])],
                                  [(yb, yhi), (ylo, yb)]):
        ax.add_patch(Rectangle((-xr, a), 2 * xr, b - a, facecolor=col, alpha=0.10, lw=0, zorder=0))
        ax.text(xband, (a + b) / 2, lab, ha="center", va="center", fontsize=6.3, color=col, rotation=270, zorder=4)
    ax.plot([-xr, xr], [yb, yb], color=P["grey"], lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.axvline(0, color=P["white"], lw=0.6, zorder=1)
    for i, k in enumerate(keys):
        y, pct = m - 1 - i, 100 * d[k].mean()
        half = pct * W / 2
        ax.barh(y, 2 * half, left=-half, height=0.66, color=ITEM[k][2], lw=0, zorder=3)
        ax.text(xlab, y, ITEM[k][0], ha="right", va="center", fontsize=6.8)
        if 2 * half >= 9.5:
            ax.text(0, y, f"{pct:.0f}%", ha="center", va="center", fontsize=6.8, color=P["white"], fontweight="bold", zorder=4)
        else:
            ax.text(half + 2.5, y, f"{pct:.0f}%", ha="left", va="center", fontsize=6.8, color=P["grey_dark"], zorder=4)
    ax.set_xlim(-114, 54); ax.set_ylim(ylo, yhi); ax.set_axis_off()


def draw_b(ax, d):
    gs = ax.get_subplotspec().subgridspec(2, 4, hspace=0.75, wspace=0.3)
    fig = ax.figure; ax.remove()
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(2)])
    d = d.assign(bin=[next(i for i, (lo, hi) in enumerate(YEAR_BINS) if lo <= y <= hi) for y in d.year])
    x = np.arange(len(YEAR_BINS))
    for a, k in zip(axs.flat, B_ITEMS):
        g = d.groupby("bin")[k].agg(["sum", "count"]).reindex(x)
        pct, lo, hi = (np.array(v) for v in zip(*[pct_ci(s, c) for s, c in zip(g["sum"], g["count"])]))
        col = ITEM[k][2]
        a.axvline(1.5, color=P["grey_dark"], lw=1.0, ls=(0, (3, 2)), zorder=1)
        a.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        a.plot(x, pct, color=col, lw=1.2); a.scatter(x, pct, s=9, color=col, zorder=3)
        a.set_ylim(0, 100); a.set_xlim(-0.3, len(x) - 0.7)
        a.set_xticks(x); a.set_xticklabels(YEAR_TICKS, rotation=90)
        a.set_yticks([0, 50, 100]); a.tick_params(labelsize=5.8, length=2)
        if a not in axs[:, 0]:
            a.set_yticklabels([])
        a.set_title(ITEM[k][1], fontsize=6.2, fontweight="bold", color=col, pad=2)
    axs[0, 0].text(1.55, 97, "STORMS", fontsize=5.2, color=P["grey_dark"], ha="left", va="top")
    for r in (0, 1):
        axs[r, 0].set_ylabel("%", fontsize=6.5)
    axs[1, 1].text(1.15, -0.50, "Publication year", transform=axs[1, 1].transAxes, ha="center", va="top", fontsize=7.3)
    return axs[0, 0]


def draw_c(ax, d):
    df = d.dropna(subset=["jif"]).copy()
    df["pct"] = 100 * df[B_ITEMS].sum(axis=1) / len(B_ITEMS)
    lx = np.log10(df.jif.to_numpy())
    fit = sm.OLS(df.pct.to_numpy(), sm.add_constant(lx)).fit()
    xs = np.linspace(lx.min(), lx.max(), 60)
    ax.plot(10 ** xs, fit.predict(sm.add_constant(xs)), color=P["grey_dark"], lw=1.2, zorder=2)
    yj = df.pct + np.random.default_rng(5).uniform(-1.6, 1.6, len(df))       # vertical jitter for visibility
    ax.scatter(df.jif, yj, s=26, color=P["grey"], edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax.text(0.02, 0.98, f"R² = {fit.rsquared:.3f}, {fmt_p(fit.pvalues[1])}", transform=ax.transAxes,
            fontsize=6.2, va="top", color=P["grey_dark"])
    ax.set_xscale("log"); ax.set_xticks(IF_TICKS); ax.set_xticklabels([str(t) for t in IF_TICKS]); ax.minorticks_off()
    ax.set_ylim(-4, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Journal impact factor (JCR 2024, log scale; n = {len(df)})")
    ax.set_ylabel("Methods items reported (% of 8)")


def draw_d(ax, d):
    x, w, tops = np.arange(len(D_ITEMS)), 0.36, []
    for j, (flag, name, col) in enumerate(GROUPS):
        sub = d[d.low_biomass == flag]
        pct, lo, hi = (np.array(v) for v in zip(*[pct_ci(sub[k].sum(), len(sub)) for k in D_ITEMS]))
        tops += list(hi)
        xx = x + (j - 0.5) * w
        ax.bar(xx, pct, width=w * 0.94, color=col, label=f"{name} (n = {len(sub)})", zorder=2)
        ax.errorbar(xx, pct, yerr=[np.clip(pct - lo, 0, None), np.clip(hi - pct, 0, None)], fmt="none",
                    ecolor=P["grey_dark"], elinewidth=0.8, capsize=2, zorder=3)
        for xi, k, h in zip(xx, D_ITEMS, hi):
            ax.text(xi, h + 1.5, f"{int(sub[k].sum())}/{len(sub)}", ha="center", va="bottom", fontsize=6.2, color=P["grey_dark"])
    ticks = []
    for k in D_ITEMS:
        tab = pd.crosstab(d.low_biomass, d[k]).reindex(index=[1, 0], columns=[0, 1], fill_value=0)
        ticks.append(f"{ITEM[k][0]}\nFisher {fmt_p(fisher_exact(tab.to_numpy())[1])}")
    ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=6.8)
    ymax = 10 * np.ceil((max(tops) + 12) / 10)
    ax.set_ylim(0, ymax); ax.set_yticks(np.arange(0, ymax + 1, 20))
    ax.set_ylabel("Studies reporting (%, 95% CI)")
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.legend(loc="upper right", fontsize=6.4, handlelength=1.2)


def main():
    d = pd.read_csv(os.path.join(DATA, "storms_items.csv"))
    apply_style()
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.955, bottom=0.10, hspace=0.32, wspace=0.26)
    for (letter, func, facet), cell in zip([("A", draw_a, False), ("B", draw_b, True), ("C", draw_c, False),
                                            ("D", draw_d, False)], [(0, 0), (0, 1), (1, 0), (1, 1)]):
        before = set(fig.axes)
        ax = fig.add_subplot(gs[cell])
        ret = func(ax, d)
        restyle([a for a in fig.axes if a not in before], tick=10, label=11.5, text_scale=1.3, legend=9, facet=facet)
        panel_label(ret if ret is not None else ax, letter, x=-0.42 if facet else -0.09, y=1.12 if facet else 1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"figure1.{ext}"), dpi=300)
    print("wrote results/figure1.pdf and results/figure1.png")


if __name__ == "__main__":
    main()
