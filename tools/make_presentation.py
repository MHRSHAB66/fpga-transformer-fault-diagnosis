"""
Generate a progress-presentation PPTX for the FPGA Fault Diagnosis project.
Run: python tools/make_presentation.py
Output: docs/presentation_progress.pptx
"""

import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "artifacts" / "results"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
OUT = DOCS / "presentation_progress.pptx"

IMG = {
    "loss_curve":        RESULTS / "transformer_baseline_loss_curve.png",
    "acc_curve":         RESULTS / "transformer_baseline_accuracy_curve.png",
    "test_cm":           RESULTS / "transformer_baseline_test_confusion_matrix.png",
    "val_cm":            RESULTS / "transformer_baseline_val_confusion_matrix.png",
    "test_per_class":    RESULTS / "transformer_baseline_test_per_class_accuracy.png",
    "disc_scores":       RESULTS / "channel_analysis" / "discriminativity_scores.png",
    "axial_b1_bands":    RESULTS / "frequency_investigation" / "axial_b1_investigation" / "axial_b1_energy_bands_per_class.png",
    "axial_b1_fft":      RESULTS / "frequency_investigation" / "paper_style_fft" / "acc_axial_b1_fft_per_class.png",
    "report_panel":      RESULTS / "spectrum_analysis" / "report" / "report_class_comparison_panel.png",
    "dataloader_files":  RESULTS / "dataloader_file_counts.png",
}

# ── colour palette ─────────────────────────────────────────────────────────────
C_DARK   = RGBColor(0x1A, 0x1A, 0x2E)   # dark navy  – background
C_MID    = RGBColor(0x16, 0x21, 0x3E)   # slightly lighter navy
C_ACCENT = RGBColor(0x0F, 0x3C, 0x96)   # deep blue  – accent bar
C_BRIGHT = RGBColor(0x00, 0xB4, 0xD8)   # cyan       – headings
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT  = RGBColor(0xCC, 0xE5, 0xFF)   # light blue – body text
C_GREEN  = RGBColor(0x06, 0xD6, 0xA0)   # green      – highlights
C_YELLOW = RGBColor(0xFF, 0xD1, 0x66)   # yellow     – callout


def _rgb(color: RGBColor):
    from pptx.oxml.ns import qn
    import lxml.etree as etree
    elem = etree.Element(qn("a:srgbClr"))
    elem.set("val", f"{color[0]:02X}{color[1]:02X}{color[2]:02X}")
    return elem


def slide_background(slide, color: RGBColor):
    """Fill slide background with a solid colour."""
    from pptx.oxml.ns import qn
    import lxml.etree as etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=C_WHITE,
                align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    txb.word_wrap = wrap
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txb


def add_accent_bar(slide, left=0, top=0, width=10, height=0.06, color=C_ACCENT):
    rect = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    rect.fill.solid()
    rect.fill.fore_color.rgb = color
    rect.line.fill.background()
    return rect


def add_image_safe(slide, path, left, top, width, height=None):
    """Add image; skip silently if file missing."""
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] image not found: {p.name}")
        return
    if height is None:
        slide.shapes.add_picture(str(p), Inches(left), Inches(top), Inches(width))
    else:
        slide.shapes.add_picture(str(p), Inches(left), Inches(top), Inches(width), Inches(height))


def add_heading(slide, text, top=0.18):
    add_accent_bar(slide, 0.3, top - 0.04, 9.4, 0.05, C_BRIGHT)
    add_textbox(slide, text, 0.3, top, 9.4, 0.55,
                font_size=24, bold=True, color=C_BRIGHT)


def bullet_frame(slide, items, left, top, width, height, font_size=14):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = C_LIGHT
        p.space_before = Pt(4)


def colored_box(slide, text, left, top, width, height,
                bg=C_ACCENT, fg=C_WHITE, font_size=13, bold=False, align=PP_ALIGN.CENTER):
    rect = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    rect.fill.solid()
    rect.fill.fore_color.rgb = bg
    rect.line.fill.background()
    tf = rect.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = fg
    return rect


# ══════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width  = Inches(10)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]   # blank


# ── Slide 1 : Title ────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)

add_accent_bar(sl, 0, 0, 10, 0.12, C_BRIGHT)
add_accent_bar(sl, 0, 7.38, 10, 0.12, C_ACCENT)

add_textbox(sl,
    "Real-Time Machinery Condition Monitoring\nUsing a Transformer Neural Network on FPGA",
    0.6, 1.4, 8.8, 1.8, font_size=30, bold=True, color=C_WHITE,
    align=PP_ALIGN.CENTER)

add_textbox(sl, "Progress Presentation — Software Phase", 0.6, 3.3, 8.8, 0.5,
            font_size=18, color=C_BRIGHT, align=PP_ALIGN.CENTER)
add_textbox(sl, "Mehrdad Sheikh Abbasi", 0.6, 4.0, 8.8, 0.4,
            font_size=16, color=C_LIGHT, align=PP_ALIGN.CENTER)
add_textbox(sl, "B.Sc. Project  ·  Amirkabir University of Technology", 0.6, 4.45, 8.8, 0.4,
            font_size=14, color=C_LIGHT, align=PP_ALIGN.CENTER)
add_textbox(sl, "June 2026", 0.6, 4.9, 8.8, 0.35,
            font_size=13, color=C_YELLOW, align=PP_ALIGN.CENTER)


# ── Slide 2 : Project Goal & End-to-End Pipeline ───────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Project Goal & System Pipeline")

steps = [
    ("MAFAULDA\nDataset", C_ACCENT),
    ("Preprocessing\n& Windowing", C_ACCENT),
    ("Lightweight\nTransformer", C_ACCENT),
    ("Quantization\n& Export", RGBColor(0x6A, 0x4C, 0x93)),
    ("FPGA\nInference", RGBColor(0x6A, 0x4C, 0x93)),
    ("UART\nto PC", RGBColor(0x6A, 0x4C, 0x93)),
    ("UI\nDisplay", RGBColor(0x6A, 0x4C, 0x93)),
]
bw, bh, gap = 1.15, 0.9, 0.08
start_x = 0.3
for i, (label, color) in enumerate(steps):
    x = start_x + i * (bw + gap)
    colored_box(sl, label, x, 1.05, bw, bh, bg=color, font_size=10, bold=True)
    if i < len(steps) - 1:
        add_textbox(sl, "→", x + bw, 1.35, 0.08, 0.4, font_size=14,
                    color=C_BRIGHT, align=PP_ALIGN.CENTER)

# Status bar
add_textbox(sl, "DONE (Stages 0-5)", 0.3, 2.15, 3.8, 0.4,
            font_size=12, color=C_GREEN, bold=True)
add_textbox(sl, "NEXT (Stages 6-12)", 3.8, 2.15, 4.0, 0.4,
            font_size=12, color=C_YELLOW, bold=True)

bullet_frame(sl, [
    "Goal: classify machinery condition in real time on an FPGA board",
    "Input: 8-channel vibration + microphone windows from a rotating machine",
    "Output: one of 4 fault classes per window — normal / imbalance / misalignment / bearing fault",
    "Hardware constraint: Transformer must be lightweight enough for FPGA mapping",
    "End product: FPGA board produces a fault label → UART → PC display",
], 0.3, 2.7, 9.4, 3.5, font_size=14)


# ── Slide 3 : Dataset ─────────────────────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Dataset: MAFAULDA")

col1 = [
    "1,951 CSV files, 50,000 Hz acquisition rate",
    "8 channels per file:",
    "      col 0 — tachometer (rotation reference)",
    "      col 1–3 — underhang bearing (axial / radial / tangential)",
    "      col 4–6 — overhang bearing  (axial / radial / tangential)",
    "      col 7 — microphone (acoustic)",
    "",
    "4 fault classes:",
    "      0 — Normal",
    "      1 — Imbalance",
    "      2 — Misalignment",
    "      3 — Bearing Fault",
]
bullet_frame(sl, col1, 0.3, 0.85, 5.0, 5.2, font_size=13)

# Split stats boxes
stats = [
    ("Train", "1,365 files\n333,060 windows"),
    ("Val",   "195 files\n47,580 windows"),
    ("Test",  "391 files\n95,404 windows"),
]
for i, (label, val) in enumerate(stats):
    colored_box(sl, f"{label}\n{val}", 5.5 + i*1.5, 0.9, 1.35, 1.05,
                bg=C_ACCENT, font_size=11, bold=True)

add_textbox(sl, "NOTE: File-level stratified split (no window leakage between splits)",
            5.2, 2.2, 4.5, 0.5, font_size=11, color=C_YELLOW)

# Imbalance note
colored_box(sl, "Class imbalance ratio: 22×\n(normal is smallest class)\n→ weighted cross-entropy loss applied",
            5.2, 2.85, 4.5, 1.05, bg=RGBColor(0x1E, 0x1E, 0x3F),
            fg=C_LIGHT, font_size=12, align=PP_ALIGN.LEFT)

add_image_safe(sl, IMG["dataloader_files"], 5.2, 4.1, 4.5, 2.8)


# ── Slide 4 : Preprocessing Pipeline ─────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Stage 2: Preprocessing & Windowing Pipeline")

steps_pp = [
    ("Raw CSV\n50 kHz", "Load & quality filter\n(median fill,\nmissing < 20%)"),
    ("Per-channel\nZ-score norm", "μ=0, σ=1 per file\nper channel"),
    ("Fixed windows\n1024 samples", "Stride = 1024\n(no overlap)\n~20 ms per window"),
    ("Tensor shape\n[B, 1024, 8]", "batch × time × channel\nfloat32"),
    ("PyTorch\nDataLoader", "batch_size = 8\npreprocessed cache"),
]
bw2, bh2 = 1.65, 1.05
for i, (top, bot) in enumerate(steps_pp):
    x = 0.25 + i * (bw2 + 0.12)
    colored_box(sl, top, x, 0.9, bw2, 0.45, bg=C_ACCENT, font_size=11, bold=True)
    colored_box(sl, bot, x, 1.37, bw2, 0.65,
                bg=RGBColor(0x1E, 0x1E, 0x3F), fg=C_LIGHT, font_size=10)
    if i < len(steps_pp) - 1:
        add_textbox(sl, "→", x + bw2, 1.04, 0.12, 0.3, font_size=13,
                    color=C_BRIGHT, align=PP_ALIGN.CENTER)

bullet_frame(sl, [
    "Optional signal processing flags (all OFF in the current trained model):",
    "   • Lowpass filter (Butterworth)  |  • Wiener denoising  |  • Resampling",
    "   → configurable in src/config.py — no flags change the default pipeline",
    "",
    "Preprocessing cache: files are preprocessed once and stored to disk (*.npz).",
    "   Cold build: ~1,877 s  |  Warm build (cache hit): seconds",
    "",
    "Splitting rule: ALL windows from the same CSV file stay in the same split.",
    "   Prevents leakage from interpolation between adjacent windows of the same recording.",
], 0.3, 2.2, 9.4, 4.0, font_size=13)


# ── Slide 5 : Frequency Analysis ─────────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Stage 2: Frequency Analysis & Resampling Decision")

bullet_frame(sl, [
    "FFT + Welch PSD computed on all 1,951 files to decide whether to downsample.",
    "",
    "Key finding — acc_axial_b1 (underhang bearing, axial):",
    "   • Normal / Imbalance / Misalignment: ~80–90% of energy sits at the 22 kHz resonance peak.",
    "   • Bearing Fault: resonance is largely suppressed — only 2–4% energy above 22 kHz.",
    "   → The 22 kHz peak IS class-discriminative, not noise.",
    "",
    "Resampling risk: downsampling below 44 kHz without anti-alias filter aliases the 22 kHz spike.",
    "Decision: keep original 50 kHz until an ablation study (50 kHz vs 24 kHz) is done.",
], 0.3, 0.85, 5.0, 4.5, font_size=13)

add_image_safe(sl, IMG["axial_b1_bands"], 5.2, 0.85, 4.6, 3.1)
add_image_safe(sl, IMG["axial_b1_fft"],   5.2, 4.0,  4.6, 2.9)

add_textbox(sl,
    "acc_axial_b1: energy in 22–23 kHz band vs fault class",
    5.2, 0.78, 4.6, 0.25, font_size=9, color=C_YELLOW)
add_textbox(sl,
    "acc_axial_b1: FFT magnitude per class",
    5.2, 3.9, 4.6, 0.25, font_size=9, color=C_YELLOW)


# ── Slide 6 : Channel Discriminativity ───────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Stage 2: Channel Discriminativity Study (8 Channels)")

bullet_frame(sl, [
    "Metric: mean coefficient of variation (CV) of class-mean PSDs across frequency bins.",
    "Higher CV → more class-separable spectral content.",
    "",
    "Ranking (0–10 kHz band, most fault-relevant):",
    "  1. acc_tangential_b1   (1.71)  -- KEEP",
    "  2. acc_radial_b1       (1.67)  -- KEEP",
    "  3. acc_axial_b1        (1.31)  -- KEEP",
    "  4. acc_radial_b2       (0.97)  -- KEEP",
    "  5. acc_axial_b2        (0.81)  -- OPTIONAL",
    "  6. acc_tangential_b2   (0.71)  -- OPTIONAL",
    "  7. microphone          (0.28)  -- LOW VALUE",
    "  8. tachometer          (0.12)  -- LOW VALUE",
    "",
    "Decision: all 8 channels kept in the current model (CHANNEL_SELECTION='all').",
    "A 6-channel vibration-only variant is configurable but requires retraining.",
], 0.3, 0.85, 5.0, 6.3, font_size=12)

add_image_safe(sl, IMG["disc_scores"], 5.2, 0.9, 4.6, 5.5)


# ── Slide 7 : Transformer Architecture ───────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Stage 4: Lightweight Transformer Architecture")

arch_items = [
    ("Input shape",       "[batch, 1024 tokens, 8 channels]"),
    ("Input projection",  "Linear(8 → 32)  — project channels to d_model"),
    ("Pos. encoding",     "Sinusoidal, added to each token"),
    ("Encoder layers",    "2 × TransformerEncoderBlock"),
    ("  ↳ Self-attention","MultiheadAttention  (4 heads, d_model=32)"),
    ("  ↳ FFN",           "Linear(32→64) → GELU → Linear(64→32)"),
    ("  ↳ Residuals",     "LayerNorm after each sub-layer"),
    ("Pooling",           "Global mean over 1024 tokens"),
    ("Classifier head",   "LayerNorm → Linear(32→16) → GELU → Linear(16→4)"),
    ("Output",            "4 logits  (normal / imbalance / misalignment / bearing)"),
]

row_h = 0.43
for i, (key, val) in enumerate(arch_items):
    y = 0.9 + i * row_h
    bg = C_ACCENT if i == 0 else (RGBColor(0x1A, 0x1A, 0x3A) if i % 2 == 0 else C_MID)
    colored_box(sl, key, 0.3, y, 2.4, row_h - 0.03, bg=bg, fg=C_YELLOW,
                font_size=11, bold=True, align=PP_ALIGN.LEFT)
    colored_box(sl, val, 2.72, y, 6.9, row_h - 0.03,
                bg=RGBColor(0x12, 0x18, 0x2E), fg=C_LIGHT,
                font_size=11, align=PP_ALIGN.LEFT)

# Hyperparams
colored_box(sl,
    "Hyperparameters\n"
    "d_model=32  |  nhead=4  |  layers=2  |  ffn_dim=64  |  dropout=0.1\n"
    "LR=1e-3  |  optimizer=Adam  |  loss=CrossEntropy (class-weighted)  |  epochs=6",
    0.3, 5.45, 9.4, 0.9, bg=C_ACCENT, fg=C_WHITE, font_size=12, bold=False)


# ── Slide 8 : Training Curves ─────────────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Stage 4: Training Curves (6 Epochs)")

add_image_safe(sl, IMG["loss_curve"], 0.25, 0.85, 4.7, 3.2)
add_image_safe(sl, IMG["acc_curve"],  5.1,  0.85, 4.7, 3.2)

add_textbox(sl, "Loss (train + val)", 0.25, 0.78, 4.7, 0.25,
            font_size=11, color=C_YELLOW, align=PP_ALIGN.CENTER)
add_textbox(sl, "Accuracy (train + val)", 5.1, 0.78, 4.7, 0.25,
            font_size=11, color=C_YELLOW, align=PP_ALIGN.CENTER)

bullet_frame(sl, [
    "Best checkpoint: epoch 6  (val loss 0.0610, val acc 98.21%)",
    "Training converged smoothly — no overfitting observed within 6 epochs.",
    "Total training time: ~2.98 hours on CPU (no GPU acceleration used).",
    "Class-weighted loss applied to handle 22× class imbalance.",
], 0.3, 4.2, 9.4, 2.5, font_size=14)


# ── Slide 9 : Confusion Matrix ────────────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Stage 4/5: Test Set — Confusion Matrix")

add_image_safe(sl, IMG["test_cm"], 0.3, 0.85, 5.2, 5.0)

# Per-class metrics table
headers = ["Class", "Precision", "Recall", "F1"]
rows = [
    ["Normal",        "0.960", "0.956", "0.958"],
    ["Imbalance",     "0.952", "0.983", "0.967"],
    ["Misalignment",  "0.983", "0.992", "0.987"],
    ["Bearing Fault", "0.993", "0.979", "0.986"],
    ["Macro avg",     "0.972", "0.977", "0.975"],
]
col_w = [1.5, 0.85, 0.85, 0.85]
header_colors = [C_ACCENT, C_ACCENT, C_ACCENT, C_ACCENT]
x_start = 5.5
y_start = 0.85
rh = 0.46

cx = x_start
for j, (h, cw) in enumerate(zip(headers, col_w)):
    colored_box(sl, h, cx, y_start, cw, rh - 0.03,
                bg=C_BRIGHT, fg=C_WHITE, font_size=12, bold=True)
    cx += cw

for i, row in enumerate(rows):
    cx = x_start
    bg = C_ACCENT if i == len(rows) - 1 else (RGBColor(0x12, 0x18, 0x2E) if i % 2 == 0 else C_MID)
    for j, (cell, cw) in enumerate(zip(row, col_w)):
        colored_box(sl, cell, cx, y_start + (i + 1) * rh, cw, rh - 0.03,
                    bg=bg, fg=C_WHITE, font_size=12,
                    bold=(i == len(rows) - 1))
        cx += cw

add_textbox(sl, "Test accuracy: 98.21%  |  Test weighted F1: 0.982",
            5.5, 3.3, 4.3, 0.45, font_size=14, bold=True, color=C_GREEN,
            align=PP_ALIGN.CENTER)

bullet_frame(sl, [
    "Test set: 391 files (20% of dataset), never seen during training.",
    "Normal class has lowest F1 (0.958) — smallest class in the dataset.",
    "Misalignment & Bearing Fault best separated (F1 > 0.986).",
    "No test-set tuning performed.",
], 5.5, 3.85, 4.3, 2.8, font_size=12)


# ── Slide 10 : Per-Class Accuracy Bar ─────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Stage 5: Per-Class Accuracy — Validation & Test")

add_image_safe(sl, IMG["val_cm"],        0.25, 0.85, 4.7, 3.0)
add_image_safe(sl, IMG["test_per_class"], 5.05, 0.85, 4.7, 3.0)

add_textbox(sl, "Validation confusion matrix", 0.25, 0.78, 4.7, 0.25,
            font_size=10, color=C_YELLOW, align=PP_ALIGN.CENTER)
add_textbox(sl, "Test per-class accuracy", 5.05, 0.78, 4.7, 0.25,
            font_size=10, color=C_YELLOW, align=PP_ALIGN.CENTER)

summary_items = [
    "Val accuracy: 98.21%  |  Val macro F1: 0.977",
    "Test accuracy: 98.21%  |  Test macro F1: 0.975",
    "Consistent train/val/test performance → no overfitting, no data leakage.",
]
bullet_frame(sl, summary_items, 0.3, 4.0, 9.4, 2.0, font_size=14)


# ── Slide 11 : Completed Work Summary ─────────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Completed Work — Stages 0 → 5")

stages_done = [
    ("Stage 0", "Project setup, environment, config system, artifact directories"),
    ("Stage 1", "MAFAULDA discovery, class mapping, quality checks (1,951 files verified)"),
    ("Stage 2", "Preprocessing pipeline: z-score norm, 1024-sample windows, file-level split\n"
                "         Frequency analysis: 22 kHz discriminative finding; resampling deferred\n"
                "         Channel discriminativity ranking; optional filter flags added to config"),
    ("Stage 3", "MLP baseline: data pipeline verified, class distribution confirmed"),
    ("Stage 4", "Lightweight Transformer (d_model=32, 2 layers, 4 heads) trained to 98.21% accuracy"),
    ("Stage 5", "Clean evaluation script; metrics JSON; confusion matrix; per-class F1 exported"),
]
y0 = 0.9
for i, (stage, desc) in enumerate(stages_done):
    colored_box(sl, stage, 0.3, y0 + i * 0.88, 1.1, 0.75,
                bg=C_GREEN, fg=C_DARK, font_size=13, bold=True)
    colored_box(sl, desc, 1.5, y0 + i * 0.88, 8.2, 0.75,
                bg=RGBColor(0x12, 0x18, 0x2E), fg=C_LIGHT,
                font_size=11, align=PP_ALIGN.LEFT)


# ── Slide 12 : Next Step — Quantization ───────────────────────────────────────
sl = prs.slides.add_slide(blank_layout)
slide_background(sl, C_DARK)
add_heading(sl, "Next Step: Stage 6 — Quantization & Fixed-Point Preparation")

add_textbox(sl, "Why quantization?", 0.3, 0.95, 9.4, 0.4,
            font_size=16, bold=True, color=C_BRIGHT)
bullet_frame(sl, [
    "FPGAs operate on fixed-point integers, not floating-point weights.",
    "Quantization converts the trained float32 model to int8 or int16 arithmetic.",
    "Must measure accuracy drop and ensure the hardware path stays valid.",
], 0.3, 1.38, 9.4, 1.1, font_size=14)

add_textbox(sl, "Planned tasks", 0.3, 2.55, 9.4, 0.4,
            font_size=16, bold=True, color=C_BRIGHT)

tasks = [
    ("6a", "Post-training quantization (PyTorch PTQ) — int8 symmetric"),
    ("6b", "Fixed-point simulation in Python — compare float vs quantized accuracy"),
    ("6c", "Export weights / scales / zero-points in a format readable by RTL"),
    ("6d", "Generate golden input/output vectors for testbench validation"),
    ("6e", "Document quantized model config; measure accuracy drop"),
]
for i, (tag, desc) in enumerate(tasks):
    colored_box(sl, tag, 0.3, 3.0 + i * 0.75, 0.55, 0.62,
                bg=C_YELLOW, fg=C_DARK, font_size=13, bold=True)
    colored_box(sl, desc, 0.95, 3.0 + i * 0.75, 8.75, 0.62,
                bg=RGBColor(0x12, 0x18, 0x2E), fg=C_LIGHT,
                font_size=12, align=PP_ALIGN.LEFT)

add_textbox(sl, "After Stage 6: hardware specification → FPGA inference core → UART → UI",
            0.3, 6.85, 9.4, 0.45, font_size=13, color=C_YELLOW, align=PP_ALIGN.CENTER)


# ── Save ───────────────────────────────────────────────────────────────────────
prs.save(str(OUT))
print(f"\nSaved: {OUT}")
print(f"Slides: {len(prs.slides)}")

