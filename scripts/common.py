"""Shared helpers: paths, palette, Wilson intervals, Holm correction and figure styling."""
import math
import os

import matplotlib as mpl
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

ITEMS = ["participant_n", "final_sample_n", "batch_effects", "sequencing_depth", "sequences_after_qc",
         "read_normalization", "negative_controls", "positive_controls", "contaminant_removal", "taxa_removal"]
METHODS_ITEMS = ITEMS[2:]          # all items except the two counts
LABELS = {"participant_n": "Participant N", "final_sample_n": "Final sample N", "batch_effects": "Batch effects",
          "sequencing_depth": "Sequencing depth", "sequences_after_qc": "Sequences after QC",
          "read_normalization": "Read normalization", "negative_controls": "Negative controls/blanks",
          "positive_controls": "Positive controls", "contaminant_removal": "Contaminant removal",
          "taxa_removal": "Taxa removal"}
STORMS = {"participant_n": "3.1/3.6", "final_sample_n": "3.6", "batch_effects": "4.13", "sequencing_depth": "4.12",
          "sequences_after_qc": "7.1", "read_normalization": "7.0", "negative_controls": "4.8",
          "positive_controls": "4.7", "contaminant_removal": "4.9", "taxa_removal": "7.1"}

PAL = {
    "blue": "#2c7fb8", "blue_light": "#a6bddb", "blue_dark": "#1b4f7a",
    "teal": "#41b6c4", "teal_dark": "#1c8a97",
    "orange": "#fe9929", "orange_dark": "#d97706",
    "rust": "#cc4c02", "purple": "#7a0177", "purple_light": "#c994c7", "purple_mid": "#a6459c",
    "grey": "#8c8c8c", "grey_light": "#d9d9d9", "grey_dark": "#4d4d4d", "ink": "#222222", "white": "#ffffff",
}


def wilson(k, n, z=1.96):
    """Wilson 95% CI for a proportion, as (lo, hi) in [0, 1]."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def holm(pvals):
    """Holm-adjusted p-values, input order preserved."""
    p = np.asarray(pvals, float)
    adj, run = np.empty(len(p)), 0.0
    for i, idx in enumerate(np.argsort(p)):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def apply_style():
    """Base rcParams; Helvetica where available (macOS), otherwise matplotlib's default sans."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PAL["grey_dark"],
        "axes.labelcolor": PAL["ink"], "text.color": PAL["ink"], "xtick.color": PAL["grey_dark"],
        "ytick.color": PAL["grey_dark"], "axes.linewidth": 0.8, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "legend.frameon": False, "pdf.fonttype": 42,
    })
    src = "/System/Library/Fonts/Helvetica.ttc"
    if os.path.exists(src):
        _register_helvetica_bold(src)
        mpl.rcParams["font.family"] = "Helvetica"


def _register_helvetica_bold(src):
    """matplotlib registers only the Regular face of Helvetica.ttc; extract Bold to a temp cache."""
    import tempfile
    from matplotlib import font_manager as fm
    out = os.path.join(tempfile.gettempdir(), "storms_fonts", "Helvetica-Bold.ttf")
    if not os.path.exists(out):
        try:
            from fontTools.ttLib import TTCollection
        except ImportError:
            return
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for f in TTCollection(src).fonts:
            if f["name"].getDebugName(6) == "Helvetica-Bold":
                f.save(out)
                break
    if os.path.exists(out):
        fm.fontManager.addfont(out)


def panel_label(ax, letter, x=-0.10, y=1.05, size=17):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size, fontweight="bold", va="bottom", ha="right",
            color=PAL["ink"])


def restyle(axes, tick, label, text_scale, legend, facet=False, facet_tick=8, facet_label=9.5, facet_title=8.5):
    """Final-figure font sizes; panel titles removed except facet titles."""
    for ax in axes:
        if facet:
            ax.title.set_fontsize(facet_title)
        else:
            for loc in ("left", "center", "right"):
                ax.set_title("", loc=loc)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(facet_tick if facet else tick)
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontsize(facet_label if facet else label)
        for t in ax.texts:
            t.set_fontsize(t.get_fontsize() * text_scale)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(legend)
