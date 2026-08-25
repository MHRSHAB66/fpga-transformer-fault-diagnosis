"""Generate accuracy-vs-size curve for all tuned architectures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

configs = [
    {"name": "P", "n_params":  1_582, "val_acc": 89.26},
    {"name": "J", "n_params":  2_044, "val_acc": 93.29},
    {"name": "H", "n_params":  5_488, "val_acc": 96.29},
    {"name": "I", "n_params":  7_412, "val_acc": 96.89},
    {"name": "F", "n_params":  9_492, "val_acc": 98.29},   # deployed
    {"name": "A", "n_params": 18_036, "val_acc": 98.19},
    {"name": "B", "n_params": 26_356, "val_acc": 97.46},
    {"name": "E", "n_params": 26_580, "val_acc": 98.73},
    {"name": "C", "n_params": 39_724, "val_acc": 99.27},
    {"name": "D", "n_params": 69_860, "val_acc": 99.66},
]

ACCENT   = "#1A6FBF"   # deployed point
OTHERS   = "#9DB4C8"   # other points
GRID_CLR = "#E4E8EE"
TEXT_CLR = "#1C2B3A"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
})

fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── grid ──────────────────────────────────────────────────────
ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# ── other points ──────────────────────────────────────────────
others = [c for c in configs if c["name"] != "F"]
ax.scatter([c["n_params"] for c in others],
           [c["val_acc"]  for c in others],
           color=OTHERS, s=70, zorder=3,
           edgecolors="white", linewidths=1.2)

# ── F point — glow effect ─────────────────────────────────────
f = next(c for c in configs if c["name"] == "F")
ax.scatter(f["n_params"], f["val_acc"],
           color=ACCENT, s=220, zorder=5,
           edgecolors="white", linewidths=2.0)
# outer glow ring
ax.scatter(f["n_params"], f["val_acc"],
           color=ACCENT, s=420, alpha=0.18, zorder=4,
           edgecolors="none")

# ── labels ────────────────────────────────────────────────────
# (dx_pts, dy_pts, ha)
label_cfg = {
    "P": (-12,  8, "right"),
    "J": ( 10,  6, "left"),
    "H": (-12,  7, "right"),
    "I": (-12,  7, "right"),
    "F": (-14,  10, "right"),
    "A": ( 10, -11, "left"),
    "B": (-12,  8, "right"),
    "E": ( 10,  7, "left"),
    "C": ( 10,  7, "left"),
    "D": ( 10, -11, "left"),
}
for c in configs:
    dx, dy, ha = label_cfg[c["name"]]
    is_f = c["name"] == "F"
    ax.annotate(
        c["name"],
        xy=(c["n_params"], c["val_acc"]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=9.5 if is_f else 8.5,
        fontweight="bold" if is_f else "normal",
        color=ACCENT if is_f else "#445566",
        ha=ha, va="bottom", zorder=7,
    )

# ── annotation box for F ──────────────────────────────────────
ax.annotate(
    "Selected for FPGA deployment\n"
    "9,492 params  ·  98.29% val  ·  97.11% int10",
    xy=(f["n_params"], f["val_acc"]),
    xytext=(28_000, 93.8),
    fontsize=8.5, color=TEXT_CLR,
    arrowprops=dict(
        arrowstyle="-|>",
        color=ACCENT, lw=1.3,
        connectionstyle="arc3,rad=0.25",
    ),
    bbox=dict(
        boxstyle="round,pad=0.5",
        fc="white", ec=ACCENT, lw=1.2,
        alpha=0.95,
    ),
    zorder=8,
)

# ── axes ──────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xlim(900, 130_000)
ax.set_ylim(87, 101.8)

ax.set_xlabel("Number of Parameters  (log scale)",
              fontsize=10.5, color=TEXT_CLR, labelpad=8)
ax.set_ylabel("Validation Accuracy",
              fontsize=10.5, color=TEXT_CLR, labelpad=8)
ax.set_title("Accuracy vs. Model Size — Architecture Search",
             fontsize=13, fontweight="bold", color=TEXT_CLR, pad=14)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.tick_params(colors="#556677", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_CLR)

# ── legend ────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=ACCENT, label="F_1layer — selected & deployed on FPGA"),
    mpatches.Patch(color=OTHERS, label="Other configurations"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
          framealpha=1.0, edgecolor=GRID_CLR,
          handlelength=1.2, handleheight=1.0)

plt.tight_layout(pad=1.4)
out_path = OUT / "arch_curve.pdf"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"[OK] saved -> {out_path}")
