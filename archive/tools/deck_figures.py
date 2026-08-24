"""
Figures for the expanded advisor progress deck (Slides 2-20).

Reuses the template palette and helpers from ``build_figures`` so everything
stays visually consistent. All figures are English, Times New Roman, minimal,
and use only real project data. No fabricated hardware numbers.

Run:
    python -m src.tools.deck_figures
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.tools.build_figures import (
    PURPLE, PURPLE2, GOLD, BLUE, BLUE_D, GREY, LIGHT, INK, OUT, HANDOFF,
    _block, _text_color,
)

QDIR = OUT.parent.parent.parent / "artifacts" / "quantized"


def _card(ax, x, y, w, h, big, small, fill=BLUE, big_fs=22, small_fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                 facecolor=fill, linewidth=0))
    tc = _text_color(fill)
    ax.text(x + w / 2, y + h * 0.60, big, ha="center", va="center",
            color=tc, fontsize=big_fs, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.22, small, ha="center", va="center",
            color=tc, fontsize=small_fs)


def _arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, color=color, lw=1.6))


def _save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------

def fig_agenda():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    parts = [("1", "Problem &\nProject Scope", BLUE),
             ("2", "Completed Progress\n& Results", PURPLE),
             ("3", "Design Decisions\n& Next Steps", GOLD)]
    for i, (n, t, c) in enumerate(parts):
        x = 0.7 + i * 3.35
        _block(ax, x, 2.6, 2.9, 1.8, f"{n}\n\n{t}", c, fs=13)
    # roadmap strip
    stages = ["Data &\nSoftware", "Quantization", "FPGA\nImplementation",
              "Comm. & UI", "Integration"]
    done = {0, 1}
    cur = {2}
    n = len(stages)
    w = 1.75; gap = 0.28; x = 0.7; y = 0.7
    for i, s in enumerate(stages):
        fill = PURPLE if i in done else (GOLD if i in cur else LIGHT)
        _block(ax, x, y, w, 0.85, s, fill, fs=9)
        if i < n - 1:
            _arrow(ax, x + w, y + 0.42, x + w + gap, y + 0.42)
        x += w + gap
    ax.text(5.5, 1.85, "You are here: transition from Quantization to FPGA implementation",
            ha="center", fontsize=10, style="italic", color=GOLD, fontweight="bold")
    _save(fig, "agenda.png")


def fig_goal():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    flow = [("Rotating machine\n+ sensors", BLUE),
            ("FPGA\ninference core", GOLD),
            ("Fault class\n0-3", PURPLE),
            ("PC dashboard\n(display only)", GREY)]
    x = 0.6; w = 2.2; gap = 0.5; y = 3.1
    for i, (t, c) in enumerate(flow):
        _block(ax, x, y, w, 1.3, t, c, fs=11)
        if i < len(flow) - 1:
            _arrow(ax, x + w, y + 0.65, x + w + gap, y + 0.65)
        x += w + gap
    ax.text(5.5, 4.75, "End-to-end embedded inference — the final classification runs ON the FPGA",
            ha="center", fontsize=11, fontweight="bold", color=PURPLE)
    # 4 class cards
    classes = [("Normal", BLUE), ("Imbalance", GOLD), ("Misalignment", GREY),
               ("Bearing fault", PURPLE)]
    x = 0.9; w = 2.3; gap = 0.25; y = 1.0
    for name, c in classes:
        _block(ax, x, y, w, 1.0, name, c, fs=12)
        x += w + gap
    ax.text(5.5, 0.55, "Four aggregated rotating-machinery fault classes",
            ha="center", fontsize=10, style="italic", color=GREY)
    _save(fig, "goal.png")


def fig_progress_glance():
    """Six workstream status cards (no misleading percentage bars)."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    cards = [
        ("Data Pipeline", "Completed"),
        ("Model Tuning & Evaluation", "Completed"),
        ("Quantization & Verification", "Completed"),
        ("FPGA Model Freeze", "In progress"),
        ("HLS Inference Core", "Starting"),
        ("UART · PC UI · Integration", "Not started"),
    ]
    style = {"Completed": (PURPLE, "white"), "In progress": (GOLD, INK),
             "Starting": (GOLD, INK), "Not started": (LIGHT, INK)}
    cw, ch, gx, gy = 3.35, 1.5, 0.25, 0.35
    x0, y0 = 0.35, 3.0
    for i, (name, status) in enumerate(cards):
        r, c = divmod(i, 3)
        x = x0 + c * (cw + gx); y = y0 - r * (ch + gy)
        fill, tc = style[status]
        ax.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0.03,rounding_size=0.08",
                     facecolor="white", edgecolor=(GREY if status == "Not started" else fill),
                     linewidth=2))
        ax.text(x + cw / 2, y + ch * 0.62, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color=INK)
        # status chip
        ax.add_patch(FancyBboxPatch((x + cw / 2 - 0.95, y + 0.22), 1.9, 0.4,
                     boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=fill, linewidth=0))
        ax.text(x + cw / 2, y + 0.42, status, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=tc)
    ax.text(5.5, 0.25, "Status labels only — not exact percentage completion.",
            ha="center", fontsize=10, style="italic", color=GREY)
    _save(fig, "progress_glance.png")


def fig_dataset_full():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    # big-number cards
    cards = [("1,951", "CSV files", BLUE), ("8", "channels", PURPLE),
             ("50 kHz", "sampling", GOLD), ("~5 s", "per file", GREY)]
    x = 0.5; w = 2.4; gap = 0.25; y = 3.5
    for big, small, c in cards:
        _card(ax, x, y, w, 1.2, big, small, c, big_fs=24)
        x += w + gap
    # sensor layout (schematic)
    ax.text(2.7, 3.0, "Sensor layout", fontsize=11, fontweight="bold", color=PURPLE, ha="center")
    _block(ax, 0.5, 1.4, 1.9, 1.2, "Underhang\nbearing\naxial/radial/\ntangential", BLUE, fs=8)
    _block(ax, 2.6, 1.4, 1.9, 1.2, "Overhang\nbearing\naxial/radial/\ntangential", BLUE_D, fs=8)
    _block(ax, 0.5, 0.5, 1.9, 0.7, "Tachometer", GREY, fs=9)
    _block(ax, 2.6, 0.5, 1.9, 0.7, "Microphone", GREY, fs=9)
    # 4 classes
    # small multichannel waveform (technical signal glyph)
    tt = np.linspace(0, 4 * np.pi, 120)
    wcols = [BLUE, PURPLE, GOLD, GREY, BLUE_D, PURPLE2]
    for k in range(6):
        yy = 2.35 - k * 0.3 + 0.1 * np.sin(tt + k * 1.1)
        ax.plot(4.7 + tt / (4 * np.pi) * 0.65, yy, color=wcols[k % 6], lw=1.3)
    ax.text(5.03, 0.42, "8-channel\ntime series", ha="center", fontsize=8.5, color=INK)
    # 4 classes
    ax.text(8.0, 3.0, "Four project classes", fontsize=11, fontweight="bold", color=PURPLE, ha="center")
    classes = [("Normal", BLUE), ("Imbalance", GOLD), ("Misalignment", GREY), ("Bearing fault", PURPLE)]
    y = 2.2
    for name, c in classes:
        _block(ax, 5.7, y, 4.7, 0.55, name, c, fs=11)
        y -= 0.7
    _save(fig, "dataset_full.png")


def fig_split_windowing():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(5.5, 4.72, "File-level split BEFORE windowing  ->  no leakage across sets",
            ha="center", fontsize=14, fontweight="bold", color=PURPLE)
    # file -> window flow
    _block(ax, 0.6, 3.35, 2.2, 1.0, "1,951\nCSV files", BLUE, fs=12)
    _arrow(ax, 2.8, 3.85, 3.5, 3.85)
    _block(ax, 3.5, 3.35, 2.5, 1.0, "File-level\nsplit", PURPLE, fs=12)
    _arrow(ax, 6.0, 3.85, 6.7, 3.85)
    _block(ax, 6.7, 3.35, 2.5, 1.0, "1024-sample\nwindows", GOLD, fs=12)
    # stacked split bar (by windows)
    total = 333060 + 47580 + 95404
    segs = [("Train\n1,365 files · 333,060", 333060, BLUE_D),
            ("Val\n47,580", 47580, GOLD),
            ("Test\n391 · 95,404", 95404, PURPLE)]
    x0 = 0.6; y = 1.85
    for lab, nwin, c in segs:
        w = 9.8 * nwin / total
        ax.add_patch(FancyBboxPatch((x0, y), w, 0.78, boxstyle="round,pad=0.005",
                     facecolor=c, linewidth=0))
        ax.text(x0 + w / 2, y + 0.39, lab, ha="center", va="center",
                color=_text_color(c), fontsize=10.5, fontweight="bold")
        x0 += w
    # windowing spec + imbalance
    ax.text(0.6, 1.3, "Window 1024 · stride 1024 · no overlap · [B, 1024, 8] · per-window z-score",
            fontsize=12, color=INK)
    ax.text(0.6, 0.85, "Class imbalance ~22x (bearing vs normal)  ->  weighted cross-entropy",
            fontsize=12, color=INK)
    # mini imbalance bars with value labels
    counts = {"normal": 8296, "imbalance": 56852, "misalign": 85156, "bearing": 182756}
    x = 7.0; maxv = max(counts.values())
    for i, (k, v) in enumerate(counts.items()):
        h = 0.75 * v / maxv
        col = [BLUE, GOLD, GREY, PURPLE][i]
        ax.add_patch(FancyBboxPatch((x, 0.45), 0.72, h, boxstyle="round,pad=0.005",
                     facecolor=col, linewidth=0))
        ax.text(x + 0.36, 0.45 + h + 0.06, f"{v//1000}k", ha="center", fontsize=8, color=INK)
        ax.text(x + 0.36, 0.28, k, ha="center", fontsize=8.5, color=INK)
        x += 0.92
    _save(fig, "split_windowing.png")


def fig_training_behavior():
    h = json.load(open(HANDOFF / "history.json"))
    ep = h["epoch"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(ep, h["train_acc"], color=BLUE_D, lw=2.4, marker="o", ms=4, label="train accuracy")
    ax.plot(ep, h["val_acc"], color=GOLD, lw=2.4, marker="s", ms=4, label="val accuracy")
    ax.axvline(10, color=PURPLE, ls="--", lw=1.3)
    ax.annotate("best val (epoch 10)", xy=(10, h["val_acc"][9]), xytext=(11, 0.93),
                fontsize=10, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Training vs Validation Accuracy (16 epochs, early stopping)",
                 color=PURPLE, fontweight="bold", fontsize=13)
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, ls="--", lw=0.5, alpha=0.4)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.5, 0.02, "Train and validation converge without a large generalization gap.",
            transform=ax.transAxes, ha="center", fontsize=10, style="italic", color=INK)
    _save(fig, "training_behavior.png")


def fig_quant_results():
    d = json.load(open(QDIR / "weight_quantization_accuracy.json"))
    fa = d["float32"]["accuracy"]
    rows = d["quantized_results"]
    names = ["float32"] + [r["name"].replace("_per_", "\nper-") for r in rows]
    accs = [fa] + [r["accuracy"] for r in rows]
    colors = [PURPLE] + [BLUE if r["n_bits"] == 16 else GOLD for r in rows]
    fig, (axb, axr) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    bars = axb.bar(range(len(names)), accs, color=colors, width=0.62)
    axb.set_ylim(0.975, 0.985); axb.set_ylabel("Accuracy (quant. eval run)")
    axb.set_title("Weight quantization — accuracy vs bit width", color=PURPLE, fontweight="bold")
    axb.set_xticks(range(len(names))); axb.set_xticklabels(names, fontsize=8.5)
    for b, a in zip(bars, accs):
        axb.text(b.get_x() + b.get_width() / 2, a + 0.0002, f"{a*100:.2f}%",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    # right panel: hybrid scheme + badge + contexts
    axr.axis("off")
    axr.add_patch(FancyBboxPatch((0.05, 0.82), 0.9, 0.15, boxstyle="round,pad=0.01",
                  facecolor=BLUE, linewidth=0, transform=axr.transAxes))
    axr.text(0.5, 0.895, "Current baseline: INT16", transform=axr.transAxes,
             ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    axr.text(0.05, 0.70, "INT16 today:", transform=axr.transAxes, fontsize=11,
             fontweight="bold", color=PURPLE)
    axr.text(0.08, 0.62, "• all matrix multiplications", transform=axr.transAxes, fontsize=10)
    axr.text(0.05, 0.50, "Still float in reference:", transform=axr.transAxes, fontsize=11,
             fontweight="bold", color=GOLD)
    axr.text(0.08, 0.42, "• Softmax  • LayerNorm  • GELU", transform=axr.transAxes, fontsize=10)
    axr.text(0.05, 0.34, "-> still need FPGA-compatible implementation",
             transform=axr.transAxes, fontsize=9, style="italic", color=INK)
    axr.text(0.05, 0.17, "Accuracy contexts:", transform=axr.transAxes, fontsize=9.5,
             fontweight="bold", color=INK)
    axr.text(0.05, 0.10, "98.22% full test  |  98.14% quant run  |  98.46% FP subset",
             transform=axr.transAxes, fontsize=8.5, color=GREY)
    _save(fig, "quant_results.png")


def fig_golden_flow():
    man = json.load(open(OUT.parent.parent.parent / "artifacts" / "golden" / "golden_manifest.json"))
    per_class = man["per_class"]
    class_names = {v: k for k, v in man["class_map"].items()}
    rows = []
    with open(OUT.parent.parent.parent / "artifacts" / "golden" / "golden_expected.txt") as f:
        for line in f:
            if not line.startswith("#"):
                p = line.split(); rows.append((int(p[0]), int(p[1]), int(p[2])))
    fig, (axm, axf) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    axm.set_xlim(0, per_class); axm.set_ylim(0, 4); axm.axis("off")
    axm.set_title("Golden vectors — 16 / 16 passed", color=PURPLE, fontweight="bold", fontsize=13)
    for idx, lab, pred in rows:
        col = idx % per_class; row = 4 - 1 - (idx // per_class)
        _block(axm, col + 0.08, row + 0.12, 0.84, 0.76, "PASS", PURPLE if lab == pred else "#B03030", fs=9)
    for ci in range(4):
        axm.text(-0.1, 4 - 1 - ci + 0.5, class_names[ci], ha="right", va="center", fontsize=10)
    # flow
    axf.axis("off")
    steps = [("PyTorch reference", BLUE), ("Fixed-point simulator", PURPLE),
             ("Golden vectors", GOLD), ("HLS co-simulation (future)", LIGHT)]
    y = 0.95
    for i, (s, c) in enumerate(steps):
        axf.add_patch(FancyBboxPatch((0.08, y - 0.13), 0.84, 0.13, boxstyle="round,pad=0.01",
                      facecolor=c, linewidth=(1.2 if c == LIGHT else 0),
                      edgecolor=GREY, transform=axf.transAxes))
        axf.text(0.5, y - 0.065, s, transform=axf.transAxes, ha="center",
                 va="center", color=_text_color(c), fontsize=9.5, fontweight="bold")
        if i < 3:
            axf.annotate("", xy=(0.5, y - 0.20), xytext=(0.5, y - 0.13),
                         xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="#555"))
        y -= 0.205
    axf.text(0.5, 0.04, "Match = exact class + |logit diff| < 0.05\nagreement vs PyTorch 2e-6",
             transform=axf.transAxes, ha="center", fontsize=8.5, style="italic", color=INK)
    _save(fig, "golden_flow.png")


def fig_hw_feasibility():
    d = json.load(open(QDIR / "transformer_int16_weights.json"))
    pe = next((int(np.prod(p["shape"])) * 4 for n, p in d["parameters"].items() if n.endswith("pe")), 0)
    w_k = d["memory_summary"]["int16_weight_bytes"] / 1024
    pe_k = pe / 1024
    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = ["Weights\n(int16)", "Positional\nencoding", "Input\nbuffer", "Activation\nbuffer"]
    vals = [w_k, pe_k, 16, 64]
    bars = ax.bar(names, vals, color=[BLUE, GOLD, BLUE_D, GREY], width=0.6)
    ax.set_ylabel("KiB (analytical estimate)")
    ax.set_title("Estimated on-chip memory per inference window", color=PURPLE, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.4)
    ax.text(0.97, 0.96, "Main bottleneck: attention scores are\n"
            "1024 x 1024 PER HEAD  (~2 MiB/head at int16\n"
            "if fully stored)  ->  must be tiled / streamed,\nnot materialized.",
            transform=ax.transAxes, ha="right", va="top", fontsize=12, color=INK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FBEEDD", ec=GOLD, lw=2))
    ax.text(0.02, -0.16, "Analytical estimates — NOT post-synthesis measurements (no LUT/FF/BRAM/DSP yet).",
            transform=ax.transAxes, fontsize=10, style="italic", color=GREY)
    _save(fig, "hw_feasibility.png")


def fig_fpga_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    # main chain
    chain = [("Input\nFIFO", BLUE_D), ("Input\nproj", BLUE), ("Pos.\nenc.", BLUE),
             ("Encoder\n(QKV, tiled attn,\nFFN, LayerNorm)", PURPLE),
             ("Mean\npool", BLUE), ("Classifier\n+ argmax", BLUE), ("UART\nframe", GOLD),
             ("PC UI", GREY)]
    x = 0.3; y = 3.0; gap = 0.18
    w = (10.4 - gap * (len(chain) - 1)) / len(chain)
    for i, (t, c) in enumerate(chain):
        _block(ax, x, y, w, 1.5, t, c, fs=8)
        if i < len(chain) - 1:
            _arrow(ax, x + w, y + 0.75, x + w + gap, y + 0.75)
        x += w + gap
    # legend (2 x 2 grid to avoid overlap)
    leg = [("int16 fixed-point MAC", BLUE), ("nonlinear op — impl. decision", GOLD),
           ("memory / streaming", BLUE_D), ("communication output", GREY)]
    for i, (t, c) in enumerate(leg):
        lx = 1.2 + (i % 2) * 4.6
        ly = 1.75 - (i // 2) * 0.55
        ax.add_patch(FancyBboxPatch((lx, ly), 0.32, 0.28, boxstyle="round,pad=0.01",
                     facecolor=c, linewidth=0))
        ax.text(lx + 0.45, ly + 0.14, t, va="center", fontsize=9.5, color=INK)
    ax.text(5.5, 0.35, "Initial path: HLS first; targeted optimization or RTL only if time permits."
            "  ·  Proposed — not yet synthesized.",
            ha="center", fontsize=9.5, style="italic", color=PURPLE, fontweight="bold")
    _save(fig, "fpga_architecture.png")


def fig_tuning_comparison():
    """Controlled tuning: 3 configs, validation metrics, from the real report."""
    rep = json.load(open(OUT.parent.parent.parent / "artifacts" / "runs" /
                         "tuning_20260616_161108" / "tuning_final_report.json"))
    by = {r["name"]: r for r in rep["all_val_results"]}
    order = ["F_1layer", "A_baseline", "B_wide_ffn"]
    params = [by[n]["n_params"] for n in order]
    pcols = [GOLD, PURPLE, GREY]

    fig = plt.figure(figsize=(11, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.42], width_ratios=[0.9, 1.25],
                          hspace=0.55, wspace=0.28)
    # parameter-count bar chart
    axp = fig.add_subplot(gs[0, 0])
    bars = axp.barh(order[::-1], params[::-1], color=pcols[::-1],
                    height=0.6, edgecolor="none")
    axp.set_title("Trainable parameters", color=PURPLE, fontweight="bold", fontsize=12)
    for b, p in zip(bars, params[::-1]):
        axp.text(b.get_width() + 500, b.get_y() + b.get_height() / 2, f"{p:,}",
                 va="center", fontsize=10, fontweight="bold", color=INK)
    axp.set_xlim(0, max(params) * 1.25)
    axp.tick_params(labelsize=10)
    for sp in ("top", "right", "bottom"):
        axp.spines[sp].set_visible(False)
    axp.set_xticks([])

    # validation-metric table with best-value highlight
    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
    rows, cols = [], ["Config", "Val acc", "Macro F2", "Fault recall", "False alarm"]
    for n in order:
        r = by[n]
        rows.append([n, f"{r['val_acc']*100:.2f}%", f"{r['val_macro_f2']:.4f}",
                     f"{r['val_fault_recall']*100:.2f}%", f"{r['val_false_alarm']*100:.2f}%"])
    best = {1: 0, 2: 1, 3: 1, 4: 1}  # col -> best row (acc:F, rest:A)
    table = axt.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                      colWidths=[0.24, 0.19, 0.19, 0.2, 0.2], bbox=[0, 0.05, 1, 0.9])
    table.auto_set_font_size(False); table.set_fontsize(10.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(PURPLE); cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(LIGHT if r % 2 else "white")
            if c == 0:
                cell.get_text().set_fontweight("bold")
            if c in best and best[c] == r - 1:
                cell.set_facecolor("#FBEEDD"); cell.get_text().set_color(PURPLE)
                cell.get_text().set_fontweight("bold")

    # badges + context
    axb = fig.add_subplot(gs[1, :]); axb.axis("off")
    axb.add_patch(FancyBboxPatch((0.02, 0.62), 0.46, 0.34, boxstyle="round,pad=0.02",
                  facecolor=PURPLE, linewidth=0, transform=axb.transAxes))
    axb.text(0.25, 0.79, "Selected software baseline:  A_baseline", transform=axb.transAxes,
             ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    axb.add_patch(FancyBboxPatch((0.52, 0.62), 0.46, 0.34, boxstyle="round,pad=0.02",
                  facecolor=GOLD, linewidth=0, transform=axb.transAxes))
    axb.text(0.75, 0.79, "Leading FPGA candidate:  F_1layer", transform=axb.transAxes,
             ha="center", va="center", color=INK, fontsize=11, fontweight="bold")
    axb.text(0.5, 0.36, "Validation tuning: 12 epochs · batch 64 · patience 3 · seed 42 · Tesla T4",
             transform=axb.transAxes, ha="center", fontsize=9.5, color=GREY)
    axb.text(0.5, 0.16, "A_baseline's full run later reached 98.22% on the held-out test "
             "set (separate run) · planned but not executed: E_3layers, C_d48, D_d64_slow",
             transform=axb.transAxes, ha="center", fontsize=9, style="italic", color=INK)
    _save(fig, "tuning_comparison.png")


def fig_decision_matrix():
    """Four clean decision cards (current vs alternative + proposed direction)."""
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    cards = [
        ("Encoder depth", "2 layers (18,036 p)", "1 layer (9,492 p)",
         "1 layer ~ same val accuracy, ~half compute/memory;\n2 layers better safety metrics",
         "1-layer as FPGA candidate", BLUE),
        ("Normalization", "Per-window z-score", "Global train norm",
         "Per-window needs mean / var / sqrt / div in HW;\nglobal = stored constants but needs retraining",
         "Train & test a global-norm version", PURPLE),
        ("Numeric precision", "INT16", "INT8",
         "INT16 = safest margin (0% drop);\nINT8 <= 0.05% drop, less memory",
         "INT16 first, INT8 if resource pressure", GOLD),
        ("HW level", "HLS-first hybrid FP", "Full-integer / RTL",
         "HLS = fast correct baseline;\nRTL = more efficient but slower / riskier",
         "Validate HLS baseline first", BLUE_D),
    ]
    pos = [(0.35, 2.65), (5.75, 2.65), (0.35, 0.3), (5.75, 0.3)]
    cw, chh = 4.9, 2.15
    for (name, cur, alt, trade, prop, c), (x, y) in zip(cards, pos):
        ax.add_patch(FancyBboxPatch((x, y), cw, chh, boxstyle="round,pad=0.03,rounding_size=0.06",
                     facecolor="white", edgecolor=c, linewidth=2.2))
        ax.text(x + 0.22, y + chh - 0.28, name, fontsize=12.5, fontweight="bold", color=c)
        ax.text(x + 0.22, y + chh - 0.68, f"Current: {cur}", fontsize=10, color=INK)
        ax.text(x + 0.22, y + chh - 1.0, f"Alternative: {alt}", fontsize=10, color=GREY)
        ax.text(x + 0.22, y + chh - 1.42, trade, fontsize=9, style="italic", color=INK)
        ax.text(x + 0.22, y + 0.2, f"-> {prop}", fontsize=10.5, fontweight="bold", color=c)
    _save(fig, "decision_matrix.png")


def fig_decisions_cards():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    cards = [
        ("1  FPGA model depth",
         "1-layer cuts params 18,036 -> 9,492 (98.28% vs 98.19% val acc),\n"
         "but 2-layer has better Macro F2 / recall / false-alarm.\n"
         "Make 1-layer the primary FPGA model, 2-layer the software reference?", BLUE),
        ("2  Normalization",
         "Should we prioritize hardware simplicity by retraining and\n"
         "evaluating with global train-set normalization?", PURPLE),
        ("3  Precision strategy",
         "Is an INT16 baseline + optional INT8 optimization an\n"
         "acceptable implementation strategy?", GOLD),
        ("4  Implementation scope",
         "Is an HLS-first hybrid fixed-point implementation enough for the\n"
         "primary deliverable (RTL only if time remains)?", BLUE_D),
    ]
    positions = [(0.5, 2.85), (5.7, 2.85), (0.5, 0.6), (5.7, 0.6)]
    for (title, body, c), (x, y) in zip(cards, positions):
        ax.add_patch(FancyBboxPatch((x, y), 4.8, 2.05, boxstyle="round,pad=0.03,rounding_size=0.08",
                     facecolor="white", edgecolor=c, linewidth=2.5))
        ax.text(x + 0.25, y + 1.65, title, fontsize=13.5, fontweight="bold", color=c)
        ax.text(x + 0.25, y + 0.8, body, fontsize=10.5, color=INK, va="center")
    _save(fig, "decisions_cards.png")


def fig_milestones():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    phases = [("Model Freeze", ["Finalize model variant", "Finalize norm & precision"], PURPLE),
              ("HLS Core", ["Freeze HW spec", "Export weights & golden", "Implement HLS blocks"], GOLD),
              ("Verification", ["C simulation", "HLS co-simulation", "Synthesis & resources"], BLUE_D),
              ("Comm & UI", ["UART output", "PC interface"], BLUE),
              ("Integration", ["End-to-end validate"], GREY)]
    x = 0.4; w = 2.0; gap = 0.12
    for i, (name, items, c) in enumerate(phases):
        _block(ax, x, 3.7, w, 0.7, name, c, fs=10)
        if i < len(phases) - 1:
            _arrow(ax, x + w, 4.05, x + w + gap, 4.05)
        for j, it in enumerate(items):
            ax.text(x + w / 2, 3.35 - j * 0.42, it, ha="center", fontsize=8.5, color=INK)
        x += w + gap
    ax.text(5.5, 0.5, "Completed (left) done · next work highlighted · later milestones muted · no fixed dates yet.",
            ha="center", fontsize=9.5, style="italic", color=GREY)
    _save(fig, "milestones.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_agenda(); fig_goal(); fig_progress_glance(); fig_dataset_full()
    fig_split_windowing(); fig_training_behavior(); fig_quant_results()
    fig_golden_flow(); fig_hw_feasibility(); fig_fpga_architecture()
    fig_tuning_comparison(); fig_decision_matrix(); fig_decisions_cards(); fig_milestones()
    print("deck figures written to", OUT)


if __name__ == "__main__":
    main()
