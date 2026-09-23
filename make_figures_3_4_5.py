"""Figures 3, 4, 5a and 5b of the manuscript, drawn directly from the pipeline output.

Input : reproduction_all_tables.xlsx (sheet "Table 5 - Models"), written by reproduce_all.py
Output: Figure_3.jpg, Figure_4.jpg, Figure_5a.jpg, Figure_5b.jpg (600 dpi)

Every plotted value is read from the Excel file; nothing is typed in by hand.
Usage:  python make_figures_3_4_5.py            (reads results/reproduction_all_tables.xlsx)
        python make_figures_3_4_5.py path/to/reproduction_all_tables.xlsx
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd

XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results") / "reproduction_all_tables.xlsx"
if not XLSX.exists():
    sys.exit(f"{XLSX} not found - run reproduce_all.py first, or pass the path of the Excel file.")
OUT = XLSX.parent          # figures are written next to the Excel file
DPI = 600

# Style of the manuscript's existing bar charts (Figure 2): Times bold, grey text,
# Office blue/orange series, no gridlines, light baseline and frame.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "font.size": 11,
})
BLUE, ORANGE = "#4472C4", "#ED7D31"
INK, RULE = "#595959", "#D9D9D9"

ORDER = ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Random Forest", "XGBoost",
         "LightGBM", "CatBoost", "MLP", "CatBoost_NO_TE", "Linear_TE_only"]
ABLATION = [("CatBoost", "CatBoost (full model)"), ("CatBoost_NO_TE", "CatBoost_NO_TE"),
            ("Linear_TE_only", "Linear_TE_only")]

t5 = pd.read_excel(XLSX, sheet_name="Table 5 - Models", header=2).set_index("Model")
missing = [m for m in ORDER if m not in t5.index]
if missing:
    sys.exit(f"Missing rows in Table 5: {missing}")


def style_axes(ax, fig):
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", length=0, colors=INK, labelsize=11)
    ax.tick_params(axis="x", pad=4)
    ax.yaxis.label.set_color(INK)
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    # thin frame around the whole chart, as in the existing figures
    fig.add_artist(Rectangle((0.002, 0.002), 0.996, 0.996, transform=fig.transFigure,
                             fill=False, edgecolor=RULE, linewidth=0.9))


def r2_fmt(ax):
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))


def tl_fmt(ax):
    ax.yaxis.set_major_locator(MultipleLocator(50_000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, facecolor="white",
                pil_kwargs={"quality": 95, "subsampling": 0, "optimize": True})
    plt.close(fig)
    print("written:", path)


# ---------------------------------------------------------------- Figure 3
fig, ax = plt.subplots(figsize=(6.5, 3.9))
x = range(len(ORDER))
w, off = 0.26, 0.145
cv = [t5.loc[m, "CV R2"] for m in ORDER]
te = [t5.loc[m, "Test R2"] for m in ORDER]
ax.bar([i - off for i in x], cv, width=w, color=BLUE, label="CV R²", zorder=2)
ax.bar([i + off for i in x], te, width=w, color=ORANGE, label="Test R²", zorder=2)
ax.set_xticks(list(x), ORDER)
ax.set_xlim(-0.6, len(ORDER) - 0.4)
ax.set_ylim(0, 1.0)
r2_fmt(ax)
style_axes(ax, fig)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.47), ncol=2, frameon=False,
                handlelength=0.8, handleheight=0.8, handletextpad=0.4, columnspacing=1.6,
                fontsize=11.5)
for txt in leg.get_texts():
    txt.set_color(INK)
fig.subplots_adjust(left=0.09, right=0.985, top=0.96, bottom=0.36)
save(fig, "Figure_3.jpg")

# ---------------------------------------------------------------- Figure 4
fig, ax = plt.subplots(figsize=(6.5, 3.9))
mae = [t5.loc[m, "Test MAE (TL)"] for m in ORDER]
ax.bar(list(x), mae, width=0.31, color=BLUE, zorder=2)
ax.set_xticks(list(x), ORDER)
ax.set_xlim(-0.6, len(ORDER) - 0.4)
top = 50_000 * (int(max(mae) // 50_000) + 1)
ax.set_ylim(0, top)
tl_fmt(ax)
ax.set_ylabel("Test MAE (TL)", fontsize=12.5, labelpad=6)
style_axes(ax, fig)
fig.subplots_adjust(left=0.15, right=0.985, top=0.96, bottom=0.30)
save(fig, "Figure_4.jpg")


# ---------------------------------------------------------------- Figures 5a, 5b
def ablation_panel(col, name, fmt_axis, label_fmt, ylim, ylabel=None):
    fig, ax = plt.subplots(figsize=(3.4, 3.78))
    vals = [t5.loc[k, col] for k, _ in ABLATION]
    xs = range(len(ABLATION))
    ax.bar(list(xs), vals, width=0.36, color=BLUE, zorder=2)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + ylim * 0.012, label_fmt(v), ha="center", va="bottom",
                color=INK, fontsize=10.5, fontweight="bold")
    ax.set_xticks(list(xs), [lab for _, lab in ABLATION])
    ax.set_xlim(-0.55, len(ABLATION) - 0.45)
    ax.set_ylim(0, ylim)
    fmt_axis(ax)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12.5, labelpad=6)
    style_axes(ax, fig)
    fig.subplots_adjust(left=0.27 if ylabel else 0.19, right=0.97, top=0.97, bottom=0.30)
    save(fig, name)


r2_top = 1.08          # head-room for the value label above a bar near 1.0
ablation_panel("Test R2", "Figure_5a.jpg", r2_fmt, lambda v: f"{v:.4f}", r2_top)
mae_top = 50_000 * (int(max(t5.loc[k, "Test MAE (TL)"] for k, _ in ABLATION) // 50_000) + 1) + 25_000
ablation_panel("Test MAE (TL)", "Figure_5b.jpg", tl_fmt, lambda v: f"{v:,.0f}", mae_top,
               ylabel="Test MAE (TL)")
