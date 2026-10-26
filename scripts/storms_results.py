#!/usr/bin/env python3
"""Table 1 and every statistic reported in the Results, from data/storms_items.csv.

Writes results/table1.csv and results/storms_results.json, and prints a summary.
Usage:  python scripts/storms_results.py
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact

from common import DATA, ITEMS, LABELS, METHODS_ITEMS, RESULTS, STORMS, holm, wilson


def prop(k, n):
    lo, hi = wilson(int(k), int(n))
    return dict(k=int(k), n=int(n), pct=round(100 * k / n, 1), lo=round(100 * lo, 1), hi=round(100 * hi, 1))


def fisher(a, b):
    return fisher_exact([[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]])[1]


def main():
    d = pd.read_csv(os.path.join(DATA, "storms_items.csv"))
    n = len(d)
    out = {"n": n}

    # Table 1
    rows = [dict(item=LABELS[it], storms=STORMS[it], yes=int(d[it].sum()), no=int(n - d[it].sum()),
                 pct_yes=round(100 * d[it].mean())) for it in ITEMS]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "table1.csv"), index=False)
    out["items"] = {it: prop(d[it].sum(), n) for it in ITEMS}

    # items reported per study
    m = d[METHODS_ITEMS].sum(axis=1)
    out["per_study"] = dict(median_methods_of_8=float(m.median()), range=[int(m.min()), int(m.max())],
                            studies_zero=int((m == 0).sum()), studies_ge5=int((m >= 5).sum()))

    # before vs after STORMS (published November 2021): <=2021 vs >=2022
    d["post_storms"] = (d.year >= 2022).astype(int)
    pre, post = d[d.post_storms == 0], d[d.post_storms == 1]
    ps = [fisher(pre[it], post[it]) for it in ITEMS]
    adj = holm(ps)
    out["storms_fisher"] = {it: dict(pre=prop(pre[it].sum(), len(pre)), post=prop(post[it].sum(), len(post)),
                                     p=round(p, 4), p_holm=round(q, 4)) for it, p, q in zip(ITEMS, ps, adj)}
    long = d.melt(id_vars=["study", "year", "post_storms"], value_vars=METHODS_ITEMS, var_name="item", value_name="y")
    long["yc"] = long.year - 2022
    gee = smf.gee("y ~ C(item) + yc + post_storms + yc:post_storms", groups="study", data=long,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    out["storms_gee"] = {}
    for term, key in (("post_storms", "level_change"), ("yc:post_storms", "slope_change")):
        lo, hi = np.exp(gee.conf_int().loc[term])
        out["storms_gee"][key] = dict(OR=round(float(np.exp(gee.params[term])), 2), lo=round(float(lo), 2),
                                      hi=round(float(hi), 2), p=round(float(gee.pvalues[term]), 3))

    # completeness (% of the eight methods items) vs journal impact factor
    d["pct_methods"] = 100 * m / len(METHODS_ITEMS)
    x = d.dropna(subset=["jif"])
    ols = smf.ols("pct_methods ~ np.log10(jif)", data=x).fit()
    out["impact_factor"] = dict(n=len(x), slope=round(float(ols.params.iloc[1]), 2), r2=round(float(ols.rsquared), 3),
                                p=round(float(ols.pvalues.iloc[1]), 2))

    # controls by specimen type
    low, stool = d[d.low_biomass == 1], d[d.low_biomass == 0]
    out["specimen_type"] = {it: dict(low_biomass=prop(low[it].sum(), len(low)), stool=prop(stool[it].sum(), len(stool)),
                                     p=round(fisher(low[it], stool[it]), 2))
                            for it in ("negative_controls", "positive_controls", "contaminant_removal")}

    # taxa removal by sequencing type
    out["taxa_removal_by_seq_type"] = {st: prop(g.taxa_removal.sum(), len(g)) for st, g in d.groupby("seq_type")}

    with open(os.path.join(RESULTS, "storms_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nper study:", out["per_study"])
    print("STORMS GEE:", out["storms_gee"], "| min Holm p:", round(float(adj.min()), 2))
    print("impact factor:", out["impact_factor"])
    print("wrote results/table1.csv and results/storms_results.json")


if __name__ == "__main__":
    main()
