"""
Render figures/fig1_timeline.png -- information-set architecture.

Two stacked time axes contrasting the leaked contemporaneous formulation
(top) with the causal 15-minute-ahead topology (bottom). Single-column
width 3.5 in, height 2.2 in, 300 DPI, 8 pt typography.
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "fig1_timeline.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Annotations are read from the weather-only leaked comparator (panel (a)
# draws G, T^m, T^a at the target stamp with no power history) and its causal
# reference on the same daylight test rows, so the figure cannot drift from
# the reported numbers. Table II reports the matched comparator instead.
_L = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results", "robustness", "plant1", "weather_only_leak.csv"))
_D = _L[_L["regime"] == "daylight"].set_index("model")
LEAK = _D.loc["Weather-only leaked RF (G, Tm, Ta at target)"]
CAUS = _D.loc["Causal RF forecast (13 features, P_{t+1} from I_t)"]

RED, GREEN, INK, GREY ="#c0392b", "#1e8449", "#1a1a1a", "#7f8c8d"
XS = {"prev": 0.22, "now": 0.50, "next": 0.80}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "font.family": "sans-serif",
})

fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.2), dpi=300)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.30)

Y_TL, Y_TICK, Y_LAB = 0.22, 0.07, 0.00   # timeline, tick half-height, labels
Y_MID, Y_TOP = 0.56, 0.93                # input row, title/badge row


def timeline(ax, title, badge_text, badge_color):
    """Shared axis, stamp ticks, panel title (left) and badge (right)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate("", xy=(0.97, Y_TL), xytext=(0.05, Y_TL),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    for key, lab in (("prev", "$t\\!-\\!1$"), ("now", "$t$"),
                     ("next", "$t\\!+\\!1$")):
        x = XS[key]
        ax.plot([x, x], [Y_TL - Y_TICK, Y_TL + Y_TICK], color=INK, lw=0.9)
        ax.text(x, Y_LAB, lab, ha="center", va="bottom", fontsize=7,
                color=INK)
    ax.text(0.0, Y_TOP, title, ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=INK)
    ax.text(1.0, Y_TOP, badge_text, ha="right", va="center", fontsize=5.6,
            color="white", zorder=5, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.30", facecolor=badge_color,
                      edgecolor="none"))


def box(ax, x, y, text, edge, face="white", pad=0.26):
    t = ax.text(x, y, text, ha="center", va="center", fontsize=7,
                color=INK, zorder=4)
    t.set_bbox(dict(boxstyle="round,pad=%.2f" % pad, facecolor=face,
                    edgecolor=edge, linewidth=0.8))


# ----------------------------------------------------------- top: leaked
ax = axes[0]
timeline(ax, "(a) Leaked formulation",
         "Weather-only leaked, daylight\n$R^2 = %.3f$,  RMSE $= %.0f$ kW"
         % (LEAK["R2"], LEAK["RMSE"]), RED)
box(ax, XS["now"], Y_MID + 0.08, "$G_t,\\; T^m_t,\\; T^a_t$", RED)
# Contemporaneous dependence: straight down onto the SAME stamp t.
ax.add_patch(FancyArrowPatch((XS["now"], Y_MID - 0.07), (XS["now"], Y_TL + 0.08),
                             arrowstyle="-|>", mutation_scale=7,
                             color=RED, lw=1.2, zorder=3))
ax.text(XS["now"] + 0.03, 0.38, "same stamp", ha="left", va="center",
        fontsize=5.8, color=RED, style="italic")
box(ax, XS["now"], Y_TL, "$P_t$", RED, face="#fdecea", pad=0.20)

# -------------------------------------------------------- bottom: causal
ax = axes[1]
timeline(ax, "(b) Causal 15-min-ahead formulation",
         "Causal forecast, daylight\n$R^2 = %.3f$,  RMSE $= %.0f$ kW"
         % (CAUS["R2"], CAUS["RMSE"]),
         GREEN)
BND = XS["now"] + 0.05
ax.add_patch(FancyBboxPatch((0.06, Y_MID - 0.14), BND - 0.06, 0.28,
                            boxstyle="round,pad=0.004",
                            facecolor="#eaf2f8", edgecolor=GREY,
                            linewidth=0.7, zorder=1))
ax.text(0.085, Y_MID, "$\\mathcal{I}_t$:", ha="left", va="center",
        fontsize=6.5, color=GREY, zorder=3)
ax.text(0.155, Y_MID, "$P_{t-1}$", ha="left", va="center", fontsize=6.5,
        color=INK, zorder=3)
ax.text(0.30, Y_MID, "$G_t,\\, T^m_t,\\, P_t$", ha="left", va="center",
        fontsize=6.5, color=INK, zorder=3)
# Forecast issue boundary at t.
ax.plot([BND, BND], [Y_TL - 0.04, Y_MID + 0.17], color=RED, lw=1.0,
        ls=(0, (3, 2)), zorder=4)
ax.text(BND - 0.015, 0.33, "issued at $t$", ha="right", va="center",
        fontsize=5.8, color=RED, style="italic")
# Forward causal arrow crossing the boundary into the future target.
ax.add_patch(FancyArrowPatch((BND + 0.01, Y_MID - 0.145),
                             (XS["next"] - 0.005, Y_TL + 0.10),
                             arrowstyle="-|>", mutation_scale=7,
                             connectionstyle="arc3,rad=-0.25",
                             color=GREEN, lw=1.2, zorder=3))
box(ax, XS["next"], Y_TL, "$\\hat{P}_{t+1}$", GREEN, face="#eafaf1", pad=0.20)

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.015)
print("wrote %s" % OUT)
