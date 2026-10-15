"""Regenerate fig_P3_critical_dim with a readable layout.

The original figure placed five alpha panels in a single row at JMLR_W_FULL width, so the
per-panel y-axis labels collided with the neighbouring axes. This version:
  * uses a 2x3 grid (5 panels + 1 blank) so each panel has room;
  * shares the y axis across a row and labels it once per row;
  * labels the x axis once per column on the bottom row;
  * draws the interpolated crossing dimension d* as a vertical marker;
  * keeps the JMLR style (serif, no baked-in title, (a)-(e) panel labels).

Input : outputs/runs_raw.csv
Output: figures/fig_P3_critical_dim.pdf (vector) and .png (300 dpi)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/home/claude/res/kan_hidden_curse_v5/outputs/runs_raw.csv"
CRIT = "/home/claude/res/kan_hidden_curse_v5/outputs/critical_dimension.csv"
OUT = "/home/claude/kanms/figures"
os.makedirs(OUT, exist_ok=True)

# ---- JMLR-compatible style (matches the pipeline's save_pub_figure settings) ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "figure.dpi": 300,
})
JMLR_W_FULL = 6.5  # inches, JMLR text width

df = pd.read_csv(SRC, low_memory=False)
crit = pd.read_csv(CRIT)

NBIG = 128000
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

cmp_ = df[df.kind.isin(["full_kan", "mlp"]) & (df.regime == "synthetic")]
# best over learning rate per seed, then best over spline order, then mean over seeds
per_seed = (cmp_.groupby(["kind", "alpha", "k", "d", "N", "seed"], dropna=False)["rel_l2"]
                .min().reset_index())
best_k = (per_seed.groupby(["kind", "alpha", "d", "N", "seed"], dropna=False)["rel_l2"]
                  .min().reset_index())
agg = (best_k.groupby(["kind", "alpha", "d", "N"], dropna=False)["rel_l2"]
              .agg(["mean", "std", "count"]).reset_index())

fig, axes = plt.subplots(2, 3, figsize=(JMLR_W_FULL, JMLR_W_FULL * 0.62),
                         sharex=True, sharey=True)
axes = axes.ravel()

for j, al in enumerate(ALPHAS):
    ax = axes[j]
    for kind, style, lab, col in [("full_kan", "s-", "KAN", "#1f77b4"),
                                  ("mlp", "o--", "MLP", "#d62728")]:
        g = agg[(agg.kind == kind) & (agg.alpha == al) & (agg.N == NBIG)].sort_values("d")
        if len(g):
            ax.plot(g["d"], g["mean"], style, color=col, label=lab, markerfacecolor="none")
    # crossing dimension, where one exists
    row = crit[(crit.N == NBIG) & (crit.alpha == al)]
    if len(row):
        v = row.d_star.iloc[0]
        try:
            dstar = float(v)
            ax.axvline(dstar, color="0.45", lw=0.7, ls=":")
            ax.annotate(rf"$d^\ast\!=\!{dstar:g}$", xy=(dstar, 0.60),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color="0.30", rotation=90, va="center")
        except (TypeError, ValueError):
            ax.annotate("no crossing", xy=(0.06, 0.90), xycoords="axes fraction",
                        fontsize=6.5, color="0.30")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.text(0.03, 0.04, rf"({chr(97+j)}) $\alpha={al:g}$", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", va="bottom", ha="left")
    ax.grid(True, which="major", lw=0.3, alpha=0.35)

# the spare sixth panel carries the legend, so no panel loses plotting area to it
axes[-1].axis("off")
_h, _l = axes[0].get_legend_handles_labels()
axes[-1].legend(_h, _l, loc="center", frameon=False, handlelength=2.2,
                fontsize=8, title="best over $(k,\\eta_0)$ per seed", title_fontsize=7.5)

# one axis label per row / column, placed on the figure rather than per panel
for ax in (axes[0], axes[3]):
    ax.set_ylabel(r"relative $L_2$ error")
for ax in (axes[3], axes[4], axes[2]):
    ax.set_xlabel(r"input dimension $d$")
    ax.tick_params(labelbottom=True)

fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.6)
fig.savefig(os.path.join(OUT, "fig_P3_critical_dim.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_P3_critical_dim.png"), dpi=300, bbox_inches="tight")
print("written:", os.path.join(OUT, "fig_P3_critical_dim.pdf"))

# ---- verification: the plotted values must match the reported tables ------------------
chk = agg[(agg.N == NBIG)].pivot_table(index="d", columns=["alpha", "kind"], values="mean")
print("\nPlotted means at N=128,000 (must match Table 12):")
print(chk.round(4).to_string())
