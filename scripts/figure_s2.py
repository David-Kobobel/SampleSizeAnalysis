#!/usr/bin/env python3
"""Figure S2: manuscript-reported vs SRA/NCBI-reported sample sizes by disease group, across the
2,305-study dataset (Data S1). Disease groups with at least 25 studies are retained.

  (A) density of manuscript vs SRA sample sizes per disease group (axis capped at 500)
  (B) jitter boxplots of SRA vs manuscript sample sizes per disease group
  plus paired t-tests and Brown-Forsythe (median-centred Levene) variance tests per disease group

Usage:  python scripts/figure_s2.py path/to/Data_S1.xlsx
Writes results/figure_s2a_density.pdf, results/figure_s2b_boxplots.pdf/.png,
       results/figure_s2_paired_ttests.csv and results/figure_s2_variance_tests.csv
"""
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

from common import RESULTS

DISEASE, MANUSCRIPT, SRA = "Disease Group", "Manuscript Sample Size", "SRA Sample Size"
MIN_N = 25
DENSITY_GROUPS = ["Respiratory", "Cancer", "Infectious", "Metabolic", "Gastrointestinal", "STI", "Reproductive",
                  "Skin", "Autoimmune", "Oral", "Cardiovascular", "Neurodegenerative", "Mental Health",
                  "Immune Complications", "Liver", "Gynecological", "Ophthalmological", "Renal", "Neurological",
                  "Endocrine", "Genetic", "Anatomical"]


def read(path):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, encoding="latin1")
    for col in (MANUSCRIPT, SRA):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False).str.strip()
                                .replace({"": np.nan, "nan": np.nan, "None": np.nan}), errors="coerce")
    return df


def paired(df):
    """Studies with both sizes, restricted to disease groups with >= MIN_N studies; groups ordered by median."""
    f = df.dropna(subset=[DISEASE, MANUSCRIPT, SRA]).copy()
    counts = f[DISEASE].value_counts()
    f = f[f[DISEASE].isin(counts[counts >= MIN_N].index)]
    order = f.groupby(DISEASE)[MANUSCRIPT].median().sort_values().index.tolist()
    return f, order


def density(df):
    d = df[df[DISEASE].isin(DENSITY_GROUPS)].dropna(subset=[MANUSCRIPT, SRA])
    d = d[(d[MANUSCRIPT] > 0) | (d[SRA] > 0)].copy()
    for col in (MANUSCRIPT, SRA):
        d[col] = d[col].clip(lower=0, upper=500)
    counts = d.groupby(DISEASE).size().sort_index()
    keep = counts[counts >= MIN_N].index.tolist()
    d = d[d[DISEASE].isin(keep)]
    sns.set_theme(context="paper", style="whitegrid")
    colors = dict(zip(keep, sns.color_palette("colorblind", n_colors=len(keep))))
    fig, axes = plt.subplots(nrows=len(keep), ncols=1, figsize=(10, 2.6 * len(keep)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, disease in zip(axes, keep):
        sub, c = d[d[DISEASE] == disease], colors[disease]
        for col, ls, lab in ((MANUSCRIPT, "-", "Manuscript"), (SRA, "--", "SRA")):
            v = sub[col][sub[col] > 0]
            if len(v) >= 2:
                sns.kdeplot(x=v, ax=ax, color=c, lw=2, linestyle=ls, label=lab, bw_adjust=0.9)
        ax.set_title(f"{disease} (n={len(sub)})", loc="left", fontsize=11, pad=2)
        ax.set_xlim(0, 500); ax.set_ylim(0, 0.006); ax.set_ylabel("Density")
        if ax is axes[0]:
            ax.legend(title="Source", loc="upper right", frameon=True)
        elif ax.get_legend():
            ax.get_legend().remove()
    axes[-1].set_xlabel("Sample Size")
    fig.legend(handles=[Patch(facecolor=colors[k], edgecolor=colors[k], label=k) for k in keep], title="Disease Group",
               loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.savefig(os.path.join(RESULTS, "figure_s2a_density.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("density: disease groups", keep, "| studies", len(d))


def boxplots_and_tests(df):
    f, order = paired(df)
    tt, var = [], []
    for disease in order:
        sub = f[f[DISEASE] == disease]
        x, y = sub[MANUSCRIPT].values, sub[SRA].values
        t, p = stats.ttest_rel(y, x)
        tt.append({"Disease Group": disease, "n (paired)": len(sub), "Mean Manuscript": float(np.mean(x)),
                   "Mean SRA": float(np.mean(y)), "Mean Diff (SRA - Manuscript)": float(np.mean(y - x)),
                   "t-stat": float(t), "p-value": float(p)})
        vm, vs = np.var(x, ddof=1), np.var(y, ddof=1)
        bf, pb = stats.levene(y, x, center="median")
        var.append({"Disease Group": disease, "n (paired)": len(sub), "Variance Manuscript": vm, "Variance SRA": vs,
                    "Variance Ratio (SRA / Manuscript)": vs / vm if vm > 0 else np.nan,
                    "Brown-Forsythe stat": bf, "p-value": pb})
    pd.DataFrame(tt).sort_values("p-value").to_csv(os.path.join(RESULTS, "figure_s2_paired_ttests.csv"), index=False)
    pd.DataFrame(var).sort_values("p-value").to_csv(os.path.join(RESULTS, "figure_s2_variance_tests.csv"), index=False)

    long = f.melt(id_vars=[DISEASE], value_vars=[SRA, MANUSCRIPT], var_name="Source", value_name="Sample Size")
    long["Source"] = long["Source"].map({SRA: "SRA", MANUSCRIPT: "Manuscript"})
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(max(10, len(order)), 6))
    sns.boxplot(data=long, x=DISEASE, y="Sample Size", hue="Source", order=order, palette=["#BFBFBF", "#E0E0E0"],
                dodge=True, showfliers=False, ax=ax)
    sns.stripplot(data=long, x=DISEASE, y="Sample Size", hue="Source", order=order, dodge=True, jitter=0.22,
                  alpha=0.55, size=5, palette=["black", "black"], ax=ax)
    ax.set_ylim(0, 900); ax.set_xlabel(""); ax.set_ylabel("Sample Size"); ax.tick_params(axis="x", rotation=90)
    sns.despine(ax=ax)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:2], l[:2], frameon=False, title="")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"figure_s2b_boxplots.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("boxplots/tests: disease groups", order, "| paired studies", len(f))


def main(path):
    df = read(path)
    density(df)
    boxplots_and_tests(df)
    print("wrote results/figure_s2a_density.pdf, figure_s2b_boxplots.pdf/.png, "
          "figure_s2_paired_ttests.csv, figure_s2_variance_tests.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
