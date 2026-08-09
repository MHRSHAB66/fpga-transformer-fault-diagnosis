"""
Regenerate every presentation figure in the Slidesgo template's palette.

All deck figures are rendered here with one consistent visual language so the
slides look cohesive with the template: the sampled palette (purple / gold /
blue / grey) and Times New Roman typography. Output goes to
``docs/slide_handoff/figures_v2/`` and is consumed by ``build_presentation.py``.

Figures reproduce the same content/numbers as the originals; only the styling
changes. Data comes from the saved run artifacts (history.json, metrics.json)
and the Stage 6 quantization reports.

Run
---
    python -m src.tools.build_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.config import PROJECT_ROOT

HANDOFF = PROJECT_ROOT / "docs" / "slide_handoff"
QDIR = PROJECT_ROOT / "artifacts" / "quantized"
GDIR = PROJECT_ROOT / "artifacts" / "golden"
OUT = HANDOFF / "figures_v2"

# Template palette (sampled from the rendered template).
PURPLE = "#281538"
PURPLE2 = "#4A2E63"
GOLD = "#F1C232"
BLUE = "#83B6E3"
BLUE_D = "#3F6FA0"
GREY = "#999999"
LIGHT = "#EDEDED"
INK = "#1A1A1A"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["text.color"] = INK
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK

CMAP = LinearSegmentedColormap.from_list("tmpl", ["#FFFFFF", BLUE, PURPLE])


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _text_color(hex_fill):
    r, g, b = (int(hex_fill[i:i + 2], 16) for i in (1, 3, 5))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 140 else INK


def _block(ax, x, y, w, h, text, fill, fs=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=_text_color(fill), fontsize=fs, fontweight="bold")


def _flow(fname, title, blocks, subtitle=None):
    """Horizontal flow of colored blocks with arrows between them."""
    n = len(blocks)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", color=PURPLE, pad=12)
    cycle = [BLUE, PURPLE, GOLD, GREY, BLUE_D, PURPLE2]
    gap = 0.28
    w = (10 - gap * (n - 1)) / n
    h = 1.5
    y = 0.9
    for i, text in enumerate(blocks):
        x = i * (w + gap)
        _block(ax, x, y, w, h, text, cycle[i % len(cycle)], fs=10)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w, y + h / 2), (x + w + gap, y + h / 2),
                arrowstyle="-|>", mutation_scale=13, color="#555555", lw=1.6))
    if subtitle:
        ax.text(5, 0.25, subtitle, ha="center", va="center",
                fontsize=10, style="italic", color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline():
    _flow("pipeline_diagram.png",
          "Software Pipeline — MAFAULDA to Fault Class",
          ["MAFAULDA\nCSV (8ch,\n50 kHz)", "File-level\nSplit\n70/10/20",
           "Windowing\n1024 / 1024", "Per-window\nZ-score", "Lightweight\nTransformer",
           "Fault Class\n0–3"],
          "All preprocessing flags OFF  ·  file-level split prevents leakage  ·  checkpoint 380 KB")


def fig_system_pipeline():
    """End-to-end data-to-result flow (software done, hardware remaining)."""
    blocks = [
        ("MAFAULDA\ndata\n8ch · 50 kHz", BLUE),
        ("Preprocess\nwindow 1024\n+ z-score", BLUE),
        ("Transformer\nlightweight\n98.22%", PURPLE),
        ("int16\nQuantization\n0% drop", PURPLE),
        ("FPGA core\nfixed-point\ninference", GOLD),
        ("UART -> PC\nUI result\ndisplay", GREY),
    ]
    n = len(blocks)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
    gap = 0.22
    w = (10 - gap * (n - 1)) / n
    h, y = 1.7, 0.95
    for i, (text, color) in enumerate(blocks):
        x = i * (w + gap)
        _block(ax, x, y, w, h, text, color, fs=9)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + w + gap, y + h / 2),
                         arrowstyle="-|>", mutation_scale=12, color="#555555", lw=1.6))
    # done / remaining band
    ax.plot([0, w * 4 + gap * 3.3], [0.55, 0.55], color=PURPLE, lw=3)
    ax.text((w * 4 + gap * 3) / 2, 0.28, "done (software + quantization)",
            ha="center", fontsize=9, color=PURPLE, style="italic")
    ax.plot([w * 4 + gap * 4, 10], [0.55, 0.55], color=GOLD, lw=3)
    ax.text((w * 4 + gap * 4 + 10) / 2, 0.28, "remaining (hardware)",
            ha="center", fontsize=9, color="#B8860B", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "system_pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_architecture():
    _flow("architecture_diagram.png",
          "Lightweight Transformer — [B, 1024, 8] to 4 classes",
          ["Input\n[B,1024,8]", "Linear\n8 → 32", "Positional\nEncoding",
           "Encoder\n×2 (4 heads)", "Mean\nPooling", "Classifier\nLN→16→4",
           "4 logits"],
          "d_model=32  ·  layers=2  ·  heads=4  ·  ffn=64  ·  checkpoint 380 KB")


def fig_stage_status():
    stages = [
        ("0", "Setup"), ("1", "Data"), ("2", "Preproc"), ("3", "MLP"),
        ("4", "Transformer"), ("5", "Eval"), ("6", "Quantize"),
        ("7", "HW spec"), ("8", "FPGA core"), ("9", "Sensor"),
        ("10", "UART"), ("11", "UI"), ("12", "Integrate"),
    ]
    done, nxt = set("0 1 2 3 4 5 6".split()), {"7"}
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 13); ax.set_ylim(0, 2.7); ax.axis("off")
    ax.set_title("Project Roadmap — Stages 0–12", fontsize=15, fontweight="bold",
                 color=PURPLE, pad=10)
    per_row = 7
    for i, (num, name) in enumerate(stages):
        row = i // per_row
        col = i % per_row
        x = col * 1.85 + 0.1
        y = 1.45 - row * 0.95
        fill = PURPLE if num in done else (GOLD if num in nxt else LIGHT)
        _block(ax, x, y, 1.65, 0.78, f"{num}\n{name}", fill, fs=9)
    # legend row along the bottom
    for lx, (c, lab) in zip([3.2, 5.6, 8.0],
                            [(PURPLE, "Done"), (GOLD, "Next"), (LIGHT, "Planned")]):
        ax.add_patch(FancyBboxPatch((lx, 0.05), 0.3, 0.22,
                     boxstyle="round,pad=0.01", facecolor=c, linewidth=0))
        ax.text(lx + 0.42, 0.16, lab, va="center", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "stage_status_diagram.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_training_curves():
    h = _load(HANDOFF / "history.json")
    ep = h["epoch"]
    for kind, tr, va, ylabel, fname, best in [
        ("Loss", h["train_loss"], h["val_loss"], "Loss", "training_loss_curve.png", True),
        ("Accuracy", h["train_acc"], h["val_acc"], "Accuracy", "training_accuracy_curve.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(ep, tr, color=BLUE_D, lw=2.2, marker="o", ms=4, label=f"train {ylabel.lower()}")
        ax.plot(ep, va, color=GOLD, lw=2.2, marker="s", ms=4, label=f"val {ylabel.lower()}")
        ax.axvline(10, color=PURPLE, ls="--", lw=1.2, alpha=0.7)
        ax.text(10.2, ax.get_ylim()[0], "best (ep 10)", color=PURPLE, fontsize=9,
                rotation=90, va="bottom", ha="left")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"Training / Validation {ylabel}", color=PURPLE, fontweight="bold")
        ax.grid(True, ls="--", lw=0.5, alpha=0.4)
        ax.legend(frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=170, bbox_inches="tight")
        plt.close(fig)


def fig_confusion():
    m = _load(HANDOFF / "metrics.json")
    cm = np.array(m["test_confusion_matrix"])
    labels = ["normal", "imbalance", "misalign", "bearing"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap=CMAP)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix (95,404 windows)", color=PURPLE, fontweight="bold")
    vmax = cm.max()
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > vmax * 0.4 else INK, fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "test_confusion_matrix.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_per_class():
    m = _load(HANDOFF / "metrics.json")
    pc = m["test_per_class_accuracy"]
    labels = ["normal", "imbalance", "misalign", "bearing"]
    vals = [pc[str(i)] for i in range(4)]
    colors = [BLUE, GOLD, GREY, PURPLE]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_ylim(0.9, 1.0); ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Test Accuracy", color=PURPLE, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "test_per_class_accuracy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_quant_accuracy():
    d = _load(QDIR / "weight_quantization_accuracy.json")
    fa = d["float32"]["accuracy"]
    rows = d["quantized_results"]
    names = ["float32"] + [r["name"].replace("_per_", "\nper-") for r in rows]
    accs = [fa] + [r["accuracy"] for r in rows]
    colors = [PURPLE] + [BLUE if r["n_bits"] == 16 else GOLD for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(range(len(names)), accs, color=colors, width=0.62)
    ax.set_ylim(0.975, 0.985); ax.set_ylabel("Test accuracy")
    ax.set_title("Weight Quantization — Accuracy vs Bit Width", color=PURPLE, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.axhline(fa, color=PURPLE, ls="--", lw=1, alpha=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.0002, f"{a*100:.2f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.99, 0.04, "int16 = zero drop  ·  int8 <= 0.05%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, style="italic", color=BLUE_D)
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "quant_accuracy_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_scheme():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Hybrid Fixed-Point Scheme (HLS-style)", fontsize=14,
                 fontweight="bold", color=PURPLE)
    seq = [("Input\nProj", BLUE), ("Q/K/V\nint16", BLUE), ("Softmax", GOLD),
           ("Attn×V\nint16", BLUE), ("LayerNorm", GOLD), ("FFN\nint16", BLUE),
           ("GELU", GOLD), ("Classifier\nint16", BLUE)]
    x, w, h, gap, y = 0.3, 1.05, 1.25, 0.12, 3.1
    for i, (t, c) in enumerate(seq):
        _block(ax, x, y, w, h, t, c, fs=8)
        if i < len(seq) - 1:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + w + gap, y + h / 2),
                         arrowstyle="-|>", mutation_scale=10, color="#555555"))
        x += w + gap
    _block(ax, 0.6, 1.2, 3.0, 0.8, "int16 MAC (done)", BLUE, fs=10)
    _block(ax, 4.0, 1.2, 4.4, 0.8, "still float in Python ref -> HW: fixed-point/LUT", GOLD, fs=9)
    ax.text(5, 0.45, "All matmuls quantized to int16 (zero accuracy drop).  "
            "softmax / LayerNorm / GELU are still float in the reference and remain\n"
            "to be implemented as fixed-point / LUT operators on the FPGA.",
            ha="center", fontsize=9, style="italic", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fixed_point_scheme.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_memory():
    d = _load(QDIR / "transformer_int16_weights.json")
    pe = next((int(np.prod(p["shape"])) * 4 for n, p in d["parameters"].items()
               if n.endswith("pe")), 0)
    ms = d["memory_summary"]
    w_k = ms["int16_weight_bytes"] / 1024
    pe_k = pe / 1024
    other_k = ms["float32_bytes"] / 1024 - pe_k
    # activation buffers (int16): input window + one d_model token buffer
    input_k = 1024 * 8 * 2 / 1024
    act_k = 1024 * 32 * 2 / 1024
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    names = ["Learnable\nweights (int16)", "Positional\nEncoding (f32)",
             "Input buffer\n(int16)", "Activation\nbuffer (int16)"]
    vals = [w_k, pe_k, input_k, act_k]
    bars = ax.bar(names, vals, color=[BLUE, GOLD, BLUE_D, GREY], width=0.6)
    ax.set_ylabel("Size (KiB)")
    ax.set_title("On-Chip Memory (per-window)", color=PURPLE, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f} KiB",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.98, 0.95, "Note: attention scores are 1024x1024 per head\n"
            "-> must be streamed, not fully buffered.\n"
            "Next step: resource synthesis (LUT/FF/BRAM/DSP).",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, style="italic",
            color=INK, bbox=dict(boxstyle="round", fc="#FBEEDD", ec=GOLD, alpha=0.9))
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "memory_breakdown.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_golden():
    man = _load(GDIR / "golden_manifest.json")
    per_class = man["per_class"]
    class_names = {v: k for k, v in man["class_map"].items()}
    rows = []
    with open(GDIR / "golden_expected.txt", encoding="ascii") as f:
        for line in f:
            if not line.startswith("#"):
                p = line.split()
                rows.append((int(p[0]), int(p[1]), int(p[2])))
    n_classes = len(man["class_map"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(0, per_class); ax.set_ylim(0, n_classes); ax.axis("off")
    ax.set_title(f"Golden Vectors — {man['self_check']['fixed_point_correct']}/"
                 f"{man['num_vectors']} correct", fontsize=13, fontweight="bold", color=PURPLE)
    for idx, label_id, pred_id in rows:
        col = idx % per_class
        row = n_classes - 1 - (idx // per_class)
        ok = label_id == pred_id
        _block(ax, col + 0.08, row + 0.12, 0.84, 0.76,
               "OK" if ok else "X", PURPLE if ok else "#B03030", fs=12)
    for ci in range(n_classes):
        ax.text(-0.1, n_classes - 1 - ci + 0.5, class_names[ci], ha="right",
                va="center", fontsize=10, color=INK)
    ax.text(per_class / 2, -0.4, "int16 fixed-point vs ground truth — all correct",
            ha="center", fontsize=9.5, style="italic", color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "golden_validation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_dataset():
    """Clean dataset summary: three stat cards + a file-split bar."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    cards = [("1,951", "CSV files (8 ch, 50 kHz)", BLUE),
             ("95,404", "test windows (unseen files)", GOLD),
             ("4", "fault classes", PURPLE)]
    for i, (num, lab, col) in enumerate(cards):
        x = 0.4 + i * 3.15
        ax.add_patch(FancyBboxPatch((x, 2.2), 2.8, 1.7,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     facecolor=col, linewidth=0))
        ax.text(x + 1.4, 3.25, num, ha="center", va="center",
                color=_text_color(col if isinstance(col, str) else "#000000"),
                fontsize=26, fontweight="bold")
        ax.text(x + 1.4, 2.55, lab, ha="center", va="center",
                color=_text_color(col), fontsize=10)
    # split bar
    total = 1365 + 195 + 391
    segs = [("Train 1,365 files", 1365, BLUE_D), ("Val 195", 195, GOLD),
            ("Test 391", 391, PURPLE)]
    x0 = 0.4
    for lab, n, col in segs:
        w = 9.2 * n / total
        ax.add_patch(FancyBboxPatch((x0, 0.8), w, 0.7, boxstyle="round,pad=0.005",
                     facecolor=col, linewidth=0))
        ax.text(x0 + w / 2, 1.15, lab, ha="center", va="center",
                color=_text_color(col), fontsize=9, fontweight="bold")
        x0 += w
    ax.text(5, 0.35, "File-level split (no window leakage) · class shares are "
            "window-level · imbalance 22x (bearing vs normal)",
            ha="center", fontsize=9.5, style="italic", color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_test_results():
    """Two panels: confusion matrix + per-class recall / F2 bars."""
    m = _load(HANDOFF / "metrics.json")
    cm = np.array(m["test_confusion_matrix"])
    pc = m["test_f2_metrics"]["per_class"]
    labels = ["normal", "imbalance", "misalign", "bearing"]
    keys = ["normal", "imbalance", "misalignment", "bearing_fault"]
    recall = [pc[k]["recall"] for k in keys]
    f2 = [pc[k]["f2"] for k in keys]

    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.6, 1.05, 1.0], wspace=0.45)
    # KPI cards
    axk = fig.add_subplot(gs[0]); axk.axis("off")
    for big, small, col, yy in [("98.22%", "Test accuracy", PURPLE, 0.56),
                                ("0.978", "Macro F2", BLUE_D, 0.10)]:
        axk.add_patch(FancyBboxPatch((0.04, yy), 0.92, 0.34,
                      boxstyle="round,pad=0.02,rounding_size=0.08",
                      facecolor=col, linewidth=0, transform=axk.transAxes))
        axk.text(0.5, yy + 0.225, big, transform=axk.transAxes, ha="center",
                 color="white", fontsize=21, fontweight="bold")
        axk.text(0.5, yy + 0.075, small, transform=axk.transAxes, ha="center",
                 color="white", fontsize=10.5)
    axk.text(0.5, 0.0, "95,404 unseen\ntest windows", transform=axk.transAxes,
             ha="center", fontsize=9, style="italic", color=GREY)
    # confusion
    axc = fig.add_subplot(gs[1])
    axc.imshow(cm, cmap=CMAP)
    axc.set_xticks(range(4)); axc.set_yticks(range(4))
    axc.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axc.set_yticklabels(labels, fontsize=9)
    axc.set_xlabel("Predicted", fontsize=10); axc.set_ylabel("True", fontsize=10)
    axc.set_title("Confusion matrix", color=PURPLE, fontweight="bold", fontsize=12)
    vmax = cm.max()
    for i in range(4):
        for j in range(4):
            axc.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > vmax * 0.4 else INK, fontsize=9)
    # per-class recall & F2
    axb = fig.add_subplot(gs[2])
    x = np.arange(4); w = 0.38
    axb.bar(x - w / 2, recall, w, color=BLUE)
    axb.bar(x + w / 2, f2, w, color=GOLD)
    axb.set_ylim(0.9, 1.0); axb.set_xticks(x)
    axb.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    axb.set_title("Per-class  recall (blue) · F2 (gold)", color=PURPLE,
                  fontweight="bold", fontsize=12)
    for xi, (r, f) in enumerate(zip(recall, f2)):
        axb.text(xi - w / 2, r + 0.002, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
        axb.text(xi + w / 2, f + 0.002, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    fig.savefig(OUT / "test_results.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_safety():
    """Fault-vs-normal breakdown with explicit definitions (2x2 + text)."""
    m = _load(HANDOFF / "metrics.json")
    fn = m["test_fault_vs_normal"]
    tp = fn["captured_faults"]; fn_ = fn["missed_faults"]
    fp = fn["false_alarms"]; tn = fn["actual_normal"] - fp
    RED = "#C0392B"
    fig, (axm, axt) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    axm.set_xlim(0, 2); axm.set_ylim(0, 2); axm.axis("off")
    axm.set_title("Fault vs normal — test windows", color=PURPLE, fontweight="bold", fontsize=12)
    # 2x2 as drawn cells: (row=actual, col=predicted)
    cells = [(0, 1, "Captured fault", f"{tp:,}", PURPLE, "white"),
             (1, 1, "MISSED (false neg.)", f"{fn_:,}", RED, "white"),
             (0, 0, "False alarm (false pos.)", f"{fp:,}", GOLD, INK),
             (1, 0, "Correct normal", f"{tn:,}", BLUE, INK)]
    for col, row, lab, num, fill, tc in cells:
        _block(axm, col + 0.05, row + 0.05, 0.9, 0.9, f"{lab}\n\n{num}", fill, fs=11)
    axm.text(1.0, -0.12, "columns = predicted · rows = actual", ha="center",
             fontsize=9, style="italic", color=GREY)
    # KPI cards
    axt.axis("off")
    for big, small, col, yy in [(f"{fn['fault_recall']*100:.2f}%", "Fault recall", PURPLE, 0.60),
                                (f"{fn['false_alarm_rate']*100:.2f}%", "False-alarm rate", GOLD, 0.12)]:
        axt.add_patch(FancyBboxPatch((0.05, yy), 0.9, 0.32,
                      boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=col,
                      linewidth=0, transform=axt.transAxes))
        axt.text(0.5, yy + 0.21, big, transform=axt.transAxes, ha="center",
                 color=_text_color(col), fontsize=22, fontweight="bold")
        axt.text(0.5, yy + 0.06, small, transform=axt.transAxes, ha="center",
                 color=_text_color(col), fontsize=11)
    axt.text(0.5, 0.02, "Missed fault (FN) is worse than a false alarm (FP) ->\n"
             "recall on faulty windows is the primary safety metric.",
             transform=axt.transAxes, ha="center", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "safety.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_fpga_plan():
    """Remaining hardware path as a block diagram + validation note."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("FPGA Plan — remaining hardware work", fontsize=15,
                 fontweight="bold", color=PURPLE, pad=8)
    row = [
        ("Input buffer\n/ FIFO\n(1024x8)", BLUE),
        ("HLS core:\nint16 MAC +\nfixed-point\nsoftmax/LN/GELU", PURPLE),
        ("Class\ndecision\n(argmax)", GOLD),
        ("UART TX\nFRAME=id,CLASS,\nLABEL,CONF", BLUE_D),
        ("PC UI\n(serial RX +\nlive display)", GREY),
    ]
    n = len(row); gap = 0.25
    w = (11 - 0.6 - gap * (n - 1)) / n
    h = 1.9; y = 2.4; x = 0.3
    for i, (t, c) in enumerate(row):
        _block(ax, x, y, w, h, t, c, fs=9)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + w + gap, y + h / 2),
                         arrowstyle="-|>", mutation_scale=13, color="#555555", lw=1.6))
        x += w + gap
    ax.text(5.5, 1.5, "Validation: Python int16 reference  ->  HLS C/RTL co-simulation "
            "must match golden vectors (exact class, |logit| < 0.05)",
            ha="center", fontsize=10, style="italic", color=INK)
    ax.text(5.5, 0.8, "Then: resource synthesis (LUT / FF / BRAM / DSP) + latency / "
            "throughput measurement", ha="center", fontsize=10, color=PURPLE,
            fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fpga_plan.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_summary_status():
    """Honest done / to-do status board."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    done = ["Leakage-free data pipeline", "Architecture tuning — 3 configs evaluated",
            "Software baseline selected (A_baseline)", "1-layer FPGA candidate identified",
            "Wider FFN rejected (measured)", "Held-out test 98.22% · fault recall 99.75%",
            "INT16 quantization (0% drop) · INT8 compared",
            "Fixed-point reference + golden vectors 16/16"]
    todo = ["Final FPGA model selection", "Global-normalization experiment",
            "Hardware-compatible softmax / LayerNorm / GELU", "HLS core + C-sim + co-sim",
            "Synthesis + resources (LUT/FF/BRAM/DSP)", "UART + PC UI + integration"]
    ax.text(2.8, 4.7, "DONE", ha="center", fontsize=15, fontweight="bold", color=PURPLE)
    ax.text(8.3, 4.7, "TO-DO", ha="center", fontsize=15, fontweight="bold", color=GOLD)
    for i, t in enumerate(done):
        y = 4.1 - i * 0.48
        _block(ax, 0.3, y, 0.42, 0.42, "OK", PURPLE, fs=8)
        ax.text(0.9, y + 0.21, t, va="center", fontsize=10.5, color=INK)
    for i, t in enumerate(todo):
        y = 4.1 - i * 0.48
        ax.add_patch(FancyBboxPatch((5.9, y), 0.42, 0.42, boxstyle="round,pad=0.02",
                     facecolor="white", edgecolor=GOLD, linewidth=1.8))
        ax.text(6.5, y + 0.21, t, va="center", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "summary_status.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_frequency():
    """Copy the real measured cumulative-energy evidence into figures_v2."""
    src = (PROJECT_ROOT / "artifacts" / "results" / "spectrum_analysis" /
           "overall" / "overall_channels_cumulative_energy.png")
    if src.exists():
        import shutil
        shutil.copy(src, OUT / "frequency_evidence.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dataset(); fig_frequency()
    fig_pipeline(); fig_system_pipeline(); fig_architecture(); fig_stage_status()
    fig_training_curves(); fig_confusion(); fig_per_class(); fig_test_results()
    fig_safety(); fig_fpga_plan(); fig_summary_status()
    fig_quant_accuracy(); fig_scheme(); fig_memory(); fig_golden()
    print(f"[info] figures saved to: {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
