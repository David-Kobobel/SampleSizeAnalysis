#!/usr/bin/env python3
"""Figure S3: the audited studies inside the PRISMA-guided corpus (audited = 1) against the rest of
the corpus, from data/prisma_corpus.csv.

Chi-square tests (cancer type, body site, first-author continent, journal impact tier) and Mann-Whitney
U tests (impact factor, publication year), Holm-corrected across the six; Cramer's V for categorical
variables. Writes results/figure_s3_tests.csv, results/figure_s3.pdf and results/figure_s3.png.

Usage:  python scripts/figure_s3.py
"""
import os
import textwrap

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats

from common import DATA, PAL as P, RESULTS, apply_style, holm, panel_label, restyle, wilson

AUD, COR = P["blue"], P["orange"]
TIERS = ["IF < 5", "IF 5–10", "IF 10–20", "IF > 20"]
ORDER = {
    "cancer_group": ["Colorectal", "Lung", "Gynecologic", "Skin / melanoma", "Oral / head & neck",
                     "Pancreatic / hepatobiliary", "Other (bladder, breast, heme, upper GI, anal)"],
    "body_site": ["Gut / faecal", "Oral / saliva", "Respiratory", "Urogenital", "Tumour / tissue",
                  "Skin", "Bile", "Blood", "Multiple sites", "Not stated in record"],
    "continent": ["Asia", "Europe", "North America", "South America", "Oceania", "Africa", "Not stated in record"],
    "journal_tier": TIERS,
}
ALIAS = {"Other (bladder, breast, heme, upper GI, anal)": "Other cancers", "Not stated in record": "Not stated",
         "Pancreatic / hepatobiliary": "Pancreatic /\nhepatobiliary"}
CATS = ["cancer_group", "body_site", "continent", "journal_tier"]


def load():
    df = pd.read_csv(os.path.join(DATA, "prisma_corpus.csv"))
    df["journal_tier"] = pd.cut(df.jif, [0, 5, 10, 20, np.inf], labels=TIERS)
    return df


def tests(a, o):
    rows = []
    for col in CATS:
        aa, oo = a[col].dropna(), o[col].dropna()
        levels = sorted(set(aa.unique()) | set(oo.unique()), key=str)
        tab = np.array([[int((aa == L).sum()) for L in levels], [int((oo == L).sum()) for L in levels]])
        tab = tab[:, tab.sum(0) > 0]
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        rows.append(dict(variable=col, test="chi-square", p=p, cramers_v=float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))))
    for col in ("jif", "year"):
        x, y = a[col].dropna().to_numpy(float), o[col].dropna().to_numpy(float)
        rows.append(dict(variable=col, test="Mann-Whitney", p=stats.mannwhitneyu(x, y, alternative="two-sided")[1],
                         cramers_v=np.nan))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    return t


def fit_label(level, width=14):
    s = ALIAS.get(level, level)
    lines = []
    for part in s.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [part])
    return "\n".join(lines[:2])


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def dumbbell(ax, a, o, col, t):
    present = {str(v) for v in pd.concat([a[col], o[col]]).dropna().astype(str)}
    levels = [L for L in ORDER[col] if L in present] + sorted(present - set(ORDER[col]))
    rows = []
    for L in levels:
        ka, ko = int((a[col].astype(str) == L).sum()), int((o[col].astype(str) == L).sum())
        la, ha = wilson(ka, len(a)); lo, ho = wilson(ko, len(o))
        rows.append((L, ka, 100 * ka / len(a), 100 * la, 100 * ha, ko, 100 * ko / len(o), 100 * lo, 100 * ho))
    y = np.arange(len(rows))[::-1]
    xhi = max(max(r[4] for r in rows), max(r[8] for r in rows)) + 26
    ax.set_xlim(-xhi * 0.58, xhi); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ticks = [v for v in MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, xhi) if -1e-9 <= v <= xhi + 1e-9]
    ax.set_xticks(ticks); ax.spines["bottom"].set_bounds(min(ticks), max(ticks))
    x0 = -xhi * 0.58
    for i, (L, ka, pa, la, ha, ko, po, lo, ho) in zip(y, rows):
        ax.plot([po, pa], [i, i], color=P["grey_light"], lw=2.2, zorder=1)
        ax.plot([la, ha], [i + 0.16, i + 0.16], color=AUD, lw=1.2, zorder=2, solid_capstyle="butt")
        ax.plot([lo, ho], [i - 0.16, i - 0.16], color=COR, lw=1.2, zorder=2, solid_capstyle="butt")
        ax.scatter([pa], [i + 0.16], s=26, color=AUD, zorder=3, edgecolor="white", linewidth=0.4)
        ax.scatter([po], [i - 0.16], s=26, color=COR, zorder=3, edgecolor="white", linewidth=0.4)
        ax.text(max(ha, ho) + 2.5, i, f"{ka}/{len(a)}  vs  {ko}/{len(o)}", va="center", fontsize=5.8, color=P["grey_dark"])
        unknown = L == "Not stated in record" and col == "continent"
        ax.text(x0 * 0.04, i, fit_label(L), ha="right", va="center", fontsize=6.2, linespacing=0.95,
                color=P["rust"] if unknown else P["ink"], fontweight="bold" if unknown else "normal")
    ax.set_xlabel("% of studies (Wilson 95% CI)")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    r = t[t.variable == col].iloc[0]
    sig = r.p_holm < 0.05
    ax.text(0.98, 0.02, f"chi2 {fmt_p(r.p)}   Holm {fmt_p(r.p_holm)}{' *' if sig else ''}   Cramer's V = {r.cramers_v:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
            color=P["rust"] if sig else P["grey_dark"], fontweight="bold" if sig else "normal")


def main():
    df = load()
    a, o = df[df.audited == 1], df[df.audited == 0]
    t = tests(a, o)
    t.to_csv(os.path.join(RESULTS, "figure_s3_tests.csv"), index=False)
    print(f"audited in corpus n = {len(a)}, rest of corpus n = {len(o)}")
    print(t.round(4).to_string(index=False))

    apply_style()
    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.955, bottom=0.085, hspace=0.30, wspace=0.26)
    for letter, col, cell in zip("ABCD", CATS, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax = fig.add_subplot(gs[cell])
        dumbbell(ax, a, o, col, t)
        if letter == "A":
            ax.legend(handles=[mlines.Line2D([], [], color=AUD, marker="o", lw=1.2, markersize=5, label=f"Study dataset (n = {len(a)})"),
                               mlines.Line2D([], [], color=COR, marker="o", lw=1.2, markersize=5, label=f"PRISMA query (n = {len(o):,})")],
                      loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handlelength=1.2, frameon=False)
        restyle([ax], tick=10, label=11.5, text_scale=1.25, legend=9)
        panel_label(ax, letter, x=-0.07, y=1.03)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"figure_s3.{ext}"), dpi=300)
    print("wrote results/figure_s3_tests.csv, results/figure_s3.pdf and results/figure_s3.png")


if __name__ == "__main__":
    main()
