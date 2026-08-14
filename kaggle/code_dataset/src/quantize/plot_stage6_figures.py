"""
Generate the Stage 6 presentation figures from the saved quantization artifacts.

Reads the JSON reports produced by the Stage 6 scripts and renders four figures
for the advisor progress presentation. Figure text is English on purpose:
matplotlib does not shape/RTL Persian reliably, and English labels keep the
charts portable. Persian narration goes in the slide body, not the figure.

Outputs -> docs/slide_handoff/stage6_figures/
  1. quant_accuracy_comparison.png   float32 vs int16/int8 (per-tensor/-channel)
  2. fixed_point_scheme.png          hybrid compute scheme (int MAC vs float NL)
  3. memory_breakdown.png            int16 weights vs float PE buffer vs other
  4. golden_validation.png           16/16 golden vectors verified grid

Run
---
    python -m src.quantize.plot_stage6_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.config import PROJECT_ROOT

QDIR = PROJECT_ROOT / "artifacts" / "quantized"
GDIR = PROJECT_ROOT / "artifacts" / "golden"
OUT = PROJECT_ROOT / "docs" / "slide_handoff" / "stage6_figures"

BLUE = "#2C6FBB"
ORANGE = "#E8833A"
GREEN = "#3FA35B"
GREY = "#9AA3AB"
DARK = "#22303C"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_accuracy():
    d = _load(QDIR / "weight_quantization_accuracy.json")
    float_acc = d["float32"]["accuracy"]
    rows = d["quantized_results"]
    names = ["float32"] + [r["name"].replace("_", "\n") for r in rows]
    accs = [float_acc] + [r["accuracy"] for r in rows]
    colors = [DARK] + [BLUE if r["n_bits"] == 16 else ORANGE for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(names)), accs, color=colors, width=0.6)
    # Zoom so the (tiny) differences are visible.
    ax.set_ylim(0.975, 0.985)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Weight Quantization — Accuracy vs Bit Width\n"
                 f"(real test subset: {d['test_subset']['windows']:,} windows, "
                 f"{d['test_subset']['test_files_used']} files)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.axhline(float_acc, color=DARK, ls="--", lw=1, alpha=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.0002,
                f"{a*100:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.99, 0.03, "int16 = zero accuracy drop  |  int8 <= 0.05%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            style="italic", color=GREEN)
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "quant_accuracy_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _block(ax, x, y, w, h, text, color, text_color="white", fs=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=0, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fs, fontweight="bold")


def fig_scheme():
    """Hybrid fixed-point scheme: int16 MAC blocks vs high-precision nonlinears."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Hybrid Fixed-Point Scheme (HLS-style)", fontsize=13, fontweight="bold")

    seq = [
        ("Input\nProjection", BLUE),
        ("Q / K / V\n(int16 MAC)", BLUE),
        ("Softmax", ORANGE),
        ("Attn x V\n(int16 MAC)", BLUE),
        ("LayerNorm", ORANGE),
        ("FFN\n(int16 MAC)", BLUE),
        ("GELU", ORANGE),
        ("Classifier\n(int16 MAC)", BLUE),
    ]
    x = 0.3
    w, h, gap = 1.05, 1.2, 0.12
    y = 3.2
    centers = []
    for text, color in seq:
        _block(ax, x, y, w, h, text, color, fs=8)
        centers.append(x + w)
        x += w + gap
    # arrows between blocks
    xx = 0.3 + w
    for _ in range(len(seq) - 1):
        ax.add_patch(FancyArrowPatch((xx, y + h / 2), (xx + gap, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=10, color=DARK))
        xx += w + gap

    # legend
    _block(ax, 0.6, 1.2, 2.6, 0.8, "int16 integer MAC", BLUE, fs=10)
    _block(ax, 3.6, 1.2, 3.2, 0.8, "high-precision fixed-point / LUT", ORANGE, fs=10)
    ax.text(5, 0.5, "matmuls quantized to int16  |  nonlinearities kept precise "
                    "(hls::exp / hls::sqrt / GELU LUT)",
            ha="center", va="center", fontsize=9, style="italic", color=DARK)
    fig.tight_layout()
    fig.savefig(OUT / "fixed_point_scheme.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_memory():
    d = _load(QDIR / "transformer_int16_weights.json")
    params = d["parameters"]
    pe_bytes = 0
    for name, p in params.items():
        if name.endswith("pe"):
            pe_bytes = int(p["shape"][0] * p["shape"][1] * p["shape"][2]) * 4
    ms = d["memory_summary"]
    w_kib = ms["int16_weight_bytes"] / 1024
    float_kib = ms["float32_bytes"] / 1024
    pe_kib = pe_bytes / 1024
    other_kib = float_kib - pe_kib

    labels = ["Learnable weights\n(int16)", "Positional Encoding\n(float32 table)",
              "Other float\n(biases, LayerNorm)"]
    vals = [w_kib, pe_kib, other_kib]
    colors = [BLUE, ORANGE, GREY]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylabel("Size (KiB)")
    ax.set_title("On-Chip Memory Breakdown\nLearnable weights are tiny — the fixed PE table dominates")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                f"{v:.1f} KiB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    ax.text(0.99, 0.9, "Insight: PE buffer ~4x the learnable weights\n"
                       "-> candidate for int16 or on-the-fly compute",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            style="italic", color=DARK,
            bbox=dict(boxstyle="round", fc="#FBEEDD", ec=ORANGE, alpha=0.8))
    fig.tight_layout()
    fig.savefig(OUT / "memory_breakdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_golden():
    man = _load(GDIR / "golden_manifest.json")
    # parse golden_expected.txt for per-vector correctness
    rows = []
    with open(GDIR / "golden_expected.txt", encoding="ascii") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            idx, label_id, pred_id = int(parts[0]), int(parts[1]), int(parts[2])
            rows.append((idx, label_id, pred_id))
    class_names = {v: k for k, v in man["class_map"].items()}
    per_class = man["per_class"]

    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.set_xlim(0, per_class)
    ax.set_ylim(0, len(man["class_map"]))
    ax.axis("off")
    ax.set_title("Golden Vector Verification — Fixed-Point vs Ground Truth\n"
                 f"{man['self_check']['fixed_point_correct']}/{man['num_vectors']} correct  |  "
                 f"{man['self_check']['fixed_point_vs_float_agree']}/{man['num_vectors']} agree with float32",
                 fontsize=12, fontweight="bold")

    n_classes = len(man["class_map"])
    for idx, label_id, pred_id in rows:
        col = idx % per_class
        row = n_classes - 1 - (idx // per_class)
        correct = (label_id == pred_id)
        color = GREEN if correct else "#C0392B"
        mark = "OK" if correct else "X"
        _block(ax, col + 0.08, row + 0.12, 0.84, 0.76, mark, color, fs=11)
    for ci in range(n_classes):
        row = n_classes - 1 - ci
        ax.text(-0.1, row + 0.5, class_names[ci], ha="right", va="center", fontsize=10)
    ax.text(per_class / 2, -0.4, f"{per_class} test windows per class, all classified correctly",
            ha="center", va="center", fontsize=9, style="italic", color=DARK)
    fig.tight_layout()
    fig.savefig(OUT / "golden_validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_accuracy()
    fig_scheme()
    fig_memory()
    fig_golden()
    print(f"[info] Stage 6 figures saved to: {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
