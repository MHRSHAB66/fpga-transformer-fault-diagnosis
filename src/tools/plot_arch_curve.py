"""Generate the architecture-search accuracy-versus-size figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

configs = [
    {"name": "P", "n_params":  1_582, "val_acc": 89.26},
    {"name": "J", "n_params":  2_044, "val_acc": 93.29},
    {"name": "G", "n_params":  2_572, "val_acc": 95.84},
    {"name": "H", "n_params":  5_488, "val_acc": 96.29},
    {"name": "I", "n_params":  7_412, "val_acc": 96.89},
    {"name": "F", "n_params":  9_492, "val_acc": 98.29},
    {"name": "A", "n_params": 18_036, "val_acc": 98.19},
    {"name": "B", "n_params": 26_356, "val_acc": 97.46},
    {"name": "E", "n_params": 26_580, "val_acc": 98.73},
    {"name": "C", "n_params": 39_724, "val_acc": 99.27},
    {"name": "D", "n_params": 69_860, "val_acc": 99.66},
]

ACCENT = "#0969DA"
ACCENT_DARK = "#074D9A"
OTHERS = "#91A9BE"
GRID_CLR = "#E5EAF0"
TEXT_CLR = "#172B3A"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
})

fig, ax = plt.subplots(figsize=(9, 5.25))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.axhspan(98, 101, color=ACCENT, alpha=0.035, zorder=0)
ax.axhline(98, color=ACCENT, linewidth=0.9, linestyle=(0, (4, 4)), alpha=0.55)
ax.text(
    118_000, 98.12, "98% validation band", ha="right", va="bottom",
    fontsize=8.2, color=ACCENT_DARK,
)

others = [c for c in configs if c["name"] != "F"]
ax.scatter([c["n_params"] for c in others],
           [c["val_acc"]  for c in others],
           color=OTHERS, s=58, zorder=3,
           edgecolors="white", linewidths=1.1)

f = next(c for c in configs if c["name"] == "F")
ax.scatter(f["n_params"], f["val_acc"],
           color=ACCENT, s=155, zorder=5,
           edgecolors="white", linewidths=1.8)
ax.scatter(f["n_params"], f["val_acc"],
           facecolors="none", edgecolors=ACCENT, s=270,
           linewidths=1.2, alpha=0.35, zorder=4)

label_cfg = {
    "P": (-9, 8, "right"),
    "J": (8, 5, "left"),
    "G": (8, 5, "left"),
    "H": (-9, 7, "right"),
    "I": (-9, 7, "right"),
    "A": (8, -11, "left"),
    "B": (-9, 7, "right"),
    "E": (8, 7, "left"),
    "C": (8, 7, "left"),
    "D": (8, -11, "left"),
}
for c in others:
    dx, dy, ha = label_cfg[c["name"]]
    ax.annotate(
        c["name"],
        xy=(c["n_params"], c["val_acc"]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=8.4, color="#435B70",
        ha=ha, va="bottom", zorder=7,
    )

ax.annotate(
    "F_1layer\n9,492 parameters  |  98.29% validation",
    xy=(f["n_params"], f["val_acc"]),
    xytext=(12_500, 97.0),
    fontsize=9.0, color=ACCENT_DARK, fontweight="bold",
    arrowprops=dict(
        arrowstyle="-", color=ACCENT, lw=1.2,
    ),
    bbox=dict(
        boxstyle="round,pad=0.45", fc="#F2F8FF", ec=ACCENT, lw=1.0,
    ),
    ha="left", va="top", zorder=8,
)

ax.set_xscale("log")
ax.set_xlim(900, 130_000)
ax.set_ylim(88, 100.6)

ax.set_xlabel("Number of Parameters  (log scale)",
              fontsize=10.5, color=TEXT_CLR, labelpad=8)
ax.set_ylabel("Validation Accuracy",
              fontsize=10.5, color=TEXT_CLR, labelpad=8)
ax.set_title("Architecture Search: Accuracy vs. Model Size",
             fontsize=13.2, fontweight="bold", color=TEXT_CLR,
             loc="left", pad=27)
ax.text(
    0.0, 1.015,
    "Eleven measured configurations; lower-left is smaller, upper-right is more accurate",
    transform=ax.transAxes, fontsize=8.7, color="#5C6F80", va="bottom",
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.tick_params(colors="#556677", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_CLR)

fig.text(
    0.99, 0.012,
    "Selected model full-test accuracy: 97.55% float  |  97.11% int10",
    ha="right", va="bottom", fontsize=8.2, color="#5C6F80",
)

plt.tight_layout(rect=(0, 0.04, 1, 1), pad=1.35)
out_path = OUT / "arch_curve.pdf"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"[OK] saved -> {out_path}")
