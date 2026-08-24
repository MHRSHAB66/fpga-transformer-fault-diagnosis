"""
Rebuild the advisor deck with NATIVE, editable PowerPoint objects.

Every card, process box, roadmap stage, decision box, KPI card, label chip,
progress bar, comparison table, block-diagram component and callout is created
as a real PowerPoint shape / table / chart -- not a flattened image. Only the
measured frequency-spectrum plot stays as an embedded image (framed natively),
because it is a real analysis figure that cannot be reasonably redrawn as shapes.

Preserved from the previous deck:
  * the hand-edited title slide (index 0),
  * the footer style (gold rule + AUT logo + page number),
  * the template palette and slide order,
  * (via add_notes) the Persian speaker notes.

Added deck-wide:
  * a consistent dark charcoal outline (1.75 pt) around every block/card,
  * a deep-purple underline directly under every content-slide title.

Run (with the .pptx CLOSED in PowerPoint):
    python -m src.tools.build_native_deck
    python -m src.tools.add_notes
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION

from src.config import PROJECT_ROOT

# ---------------------------------------------------------------- paths -----
HANDOFF = PROJECT_ROOT / "docs" / "slide_handoff"
FIG = HANDOFF / "figures_v2"
QDIR = PROJECT_ROOT / "artifacts" / "quantized"
GDIR = PROJECT_ROOT / "artifacts" / "golden"
DECK = HANDOFF / "FPGA_Fault_Diagnosis_Progress.pptx"
LOGO = HANDOFF / "assets" / "AUT_en.png"

# -------------------------------------------------------------- palette -----
PURPLE = "281538"; PURPLE2 = "4A2E63"; GOLD = "F1C232"
BLUE = "83B6E3"; BLUE_D = "3F6FA0"; GREY = "999999"; GREY_D = "555555"
LIGHT = "EDEDED"; INK = "1A1A1A"; WHITE = "FFFFFF"; RED = "C0392B"
CHARCOAL = "2A2A33"          # consistent dark block outline
CREAM = "FBEEDD"             # highlight / callout background

TITLE_FONT = "Montserrat"
BODY_FONT = "Calibri"
FOOTER_TEXT = "Machinery Fault Diagnosis on FPGA"

PA = {"C": PP_ALIGN.CENTER, "L": PP_ALIGN.LEFT, "R": PP_ALIGN.RIGHT}
AN = {"M": MSO_ANCHOR.MIDDLE, "T": MSO_ANCHOR.TOP, "B": MSO_ANCHOR.BOTTOM}


# ------------------------------------------------------------- helpers -------
def C(h):
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _lum(h):
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


def tcolor(h):
    return WHITE if _lum(h) < 140 else INK


def _blend(a, b, t):
    t = max(0.0, min(1.0, t))
    out = []
    for i in (0, 2, 4):
        ca, cb = int(a[i:i + 2], 16), int(b[i:i + 2], 16)
        out.append(int(round(ca + (cb - ca) * t)))
    return "%02X%02X%02X" % tuple(out)


def fill_text(tf, items, anchor="M", align="C", wrap=True, margins=(4, 4, 2, 2)):
    """items: list of (text, size, bold, color_hex, font). '\\n' -> new paragraph."""
    tf.word_wrap = wrap
    tf.vertical_anchor = AN[anchor]
    ml, mr, mt, mb = margins
    tf.margin_left = Pt(ml); tf.margin_right = Pt(mr)
    tf.margin_top = Pt(mt); tf.margin_bottom = Pt(mb)
    first = True
    for text, size, bold, color, font in items:
        for line in str(text).split("\n"):
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.alignment = PA[align]
            r = p.add_run(); r.text = line
            r.font.size = Pt(size); r.font.bold = bold
            r.font.name = font; r.font.color.rgb = C(color)
    return tf


def card(slide, x, y, w, h, fill=WHITE, outline=CHARCOAL, line_w=1.75,
         rounded=True, radius=0.09):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = C(fill)
    if outline is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = C(outline); shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    if rounded:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    return shp


def block(slide, x, y, w, h, text, fill, size=10, tcol=None, outline=CHARCOAL,
          line_w=1.75, radius=0.09, font=TITLE_FONT):
    """Solid-fill rounded block with centered text (pipeline / class boxes)."""
    shp = card(slide, x, y, w, h, fill=fill, outline=outline, line_w=line_w, radius=radius)
    fill_text(shp.text_frame, [(text, size, True, tcol or tcolor(fill), font)], "M", "C")
    return shp


def textbox(slide, x, y, w, h, items, anchor="T", align="L", wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_text(tb.text_frame, items, anchor, align, wrap, margins=(2, 2, 1, 1))
    return tb


def chip(slide, x, y, w, h, text, fill, tcol=None, size=10.5, outline=None):
    shp = card(slide, x, y, w, h, fill=fill, outline=outline, line_w=1.0, radius=0.5)
    fill_text(shp.text_frame, [(text, size, True, tcol or tcolor(fill), TITLE_FONT)], "M", "C")
    return shp


def arrow(slide, x1, y1, x2, y2, color=GREY_D, w=1.6):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                      Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = C(color); conn.line.width = Pt(w)
    conn.shadow.inherit = False
    ln = conn.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return conn


def kpi(slide, x, y, w, h, big, small, fill):
    shp = card(slide, x, y, w, h, fill=fill, outline=CHARCOAL, radius=0.10)
    tc = tcolor(fill)
    fill_text(shp.text_frame,
              [(big, 22, True, tc, TITLE_FONT), (small, 10.5, False, tc, BODY_FONT)],
              "M", "C")
    return shp


def title_block(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.26), Inches(9.0), Inches(0.62))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.name = TITLE_FONT; r.font.size = Pt(22); r.font.bold = True
    r.font.color.rgb = C(INK)
    ul = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(0.99),
                                Inches(8.2), Pt(2.2))
    ul.fill.solid(); ul.fill.fore_color.rgb = C(PURPLE)
    ul.line.fill.background(); ul.shadow.inherit = False


def footer(slide, page, total):
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(5.28),
                                  Inches(9.0), Pt(1.4))
    rule.fill.solid(); rule.fill.fore_color.rgb = C(GOLD)
    rule.line.fill.background(); rule.shadow.inherit = False
    if LOGO.exists():
        slide.shapes.add_picture(str(LOGO), Inches(0.4), Inches(5.34), height=Inches(0.24))
    textbox(slide, 0.75, 5.34, 6.3, 0.24,
            [(FOOTER_TEXT, 11, False, GREY, "Times New Roman")], "M", "L")
    tb = slide.shapes.add_textbox(Inches(7.9), Inches(5.34), Inches(1.6), Inches(0.24))
    tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = f"{page} / {total}"
    r.font.name = "Times New Roman"; r.font.size = Pt(11); r.font.color.rgb = C(GREY)


def caption(slide, x, y, w, text, color=GREY, size=10, align="C", italic=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.4))
    p = tb.text_frame.paragraphs[0]; p.alignment = PA[align]
    r = p.add_run(); r.text = text
    r.font.name = BODY_FONT; r.font.size = Pt(size)
    r.font.italic = italic; r.font.color.rgb = C(color)


# --------------------------------------------------------- native tables ----
def set_cell(cell, text, size, bold, color, fill, align="C", font=BODY_FONT):
    cell.fill.solid(); cell.fill.fore_color.rgb = C(fill)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Pt(3); cell.margin_right = Pt(3)
    cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
    tf = cell.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PA[align]
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.name = font; r.font.color.rgb = C(color)


def new_table(slide, x, y, w, h, nrows, ncols):
    gf = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = gf.table
    tbl.first_row = False; tbl.horz_banding = False
    return tbl


# --------------------------------------------------------- native charts ----
def bar_chart(slide, x, y, w, h, cats, values, colors, ymin=None, ymax=None,
              num_fmt="0.00%", label_size=8, axis_size=8, horizontal=False,
              gap=70):
    cd = CategoryChartData(); cd.categories = cats
    cd.add_series("s", values)
    ctype = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(ctype, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = False; ch.has_title = False
    plot = ch.plots[0]; plot.gap_width = gap
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = num_fmt; dl.number_format_is_linked = False
    dl.font.size = Pt(label_size); dl.font.bold = True
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    s = plot.series[0]
    for i, pt in enumerate(s.points):
        pt.format.fill.solid(); pt.format.fill.fore_color.rgb = C(colors[i])
        pt.format.line.fill.background()
    va = ch.value_axis
    if ymin is not None:
        va.minimum_scale = ymin
    if ymax is not None:
        va.maximum_scale = ymax
    va.tick_labels.font.size = Pt(axis_size)
    va.tick_labels.number_format = num_fmt; va.tick_labels.number_format_is_linked = False
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(axis_size)
    return ch


def grouped_bar(slide, x, y, w, h, cats, series, ymin, ymax, axis_size=8,
                label_size=7, num_fmt="0.000"):
    """series: list of (name, values, color)."""
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                                Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart; ch.has_title = False
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False; ch.legend.font.size = Pt(8)
    plot = ch.plots[0]; plot.gap_width = 80
    for ser, (name, vals, color) in zip(plot.series, series):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = C(color)
        ser.format.line.fill.background()
    va = ch.value_axis
    va.minimum_scale = ymin; va.maximum_scale = ymax
    va.tick_labels.font.size = Pt(axis_size)
    va.tick_labels.number_format = "0.00"; va.tick_labels.number_format_is_linked = False
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(axis_size)
    return ch


def line_chart(slide, x, y, w, h, cats, series, ymin, ymax):
    """series: list of (name, values, color)."""
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS,
                                Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart; ch.has_title = False
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False; ch.legend.font.size = Pt(9)
    for ser, (name, vals, color) in zip(ch.plots[0].series, series):
        ser.format.line.color.rgb = C(color); ser.format.line.width = Pt(2.25)
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = C(color)
    va = ch.value_axis
    va.minimum_scale = ymin; va.maximum_scale = ymax
    va.tick_labels.font.size = Pt(9)
    va.tick_labels.number_format = "0.00"; va.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(9)
    return ch


def framed_image(slide, path, x, y, w, h, pad=0.12):
    """A native rounded frame with an image fitted inside it."""
    card(slide, x, y, w, h, fill=WHITE, outline=CHARCOAL, line_w=1.75, radius=0.04)
    bx, by = x + pad, y + pad
    bw, bh = w - 2 * pad, h - 2 * pad
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    if ar > bw / bh:
        pw = bw; ph = bw / ar
    else:
        ph = bh; pw = bh * ar
    slide.shapes.add_picture(str(path), Inches(bx + (bw - pw) / 2),
                             Inches(by + (bh - ph) / 2), Inches(pw), Inches(ph))


# ================================================================= slides ====
def s_agenda(s):
    parts = [("1", "Problem &\nProject Scope", BLUE),
             ("2", "Completed Progress\n& Results", PURPLE),
             ("3", "Design Decisions\n& Next Steps", GOLD)]
    x, w, gap = 0.55, 2.86, 0.28
    for n, t, c in parts:
        b = card(s, x, 1.45, w, 1.5, fill=c, outline=CHARCOAL, radius=0.08)
        fill_text(b.text_frame, [(n, 24, True, tcolor(c), TITLE_FONT),
                                 (t, 12.5, True, tcolor(c), TITLE_FONT)], "M", "C")
        x += w + gap
    stages = ["Data &\nSoftware", "Quantization", "FPGA\nImplementation",
              "Comm. & UI", "Integration"]
    done, cur = {0, 1}, {2}
    x, w, gap = 0.6, 1.5, 0.3
    for i, st in enumerate(stages):
        fill = PURPLE if i in done else (GOLD if i in cur else LIGHT)
        block(s, x, 3.45, w, 0.8, st, fill, size=9.5)
        if i < len(stages) - 1:
            arrow(s, x + w, 3.85, x + w + gap, 3.85)
        x += w + gap
    caption(s, 0.6, 4.5, 8.8,
            "You are here: transition from Quantization to FPGA implementation",
            color=GOLD, size=11)


def s_goal(s):
    caption(s, 0.5, 1.2, 9.0,
            "End-to-end embedded inference — the final classification runs ON the FPGA",
            color=PURPLE, size=12, italic=False)
    flow = [("Rotating machine\n+ sensors", BLUE), ("FPGA\ninference core", GOLD),
            ("Fault class\n0–3", PURPLE), ("PC dashboard\n(display only)", GREY)]
    x, w, gap = 0.6, 1.95, 0.42
    for i, (t, c) in enumerate(flow):
        block(s, x, 1.75, w, 1.15, t, c, size=11)
        if i < len(flow) - 1:
            arrow(s, x + w, 2.32, x + w + gap, 2.32)
        x += w + gap
    classes = [("Normal", BLUE), ("Imbalance", GOLD), ("Misalignment", GREY),
               ("Bearing fault", PURPLE)]
    x, w, gap = 0.6, 2.07, 0.28
    for name, c in classes:
        block(s, x, 3.5, w, 1.0, name, c, size=12.5)
        x += w + gap
    caption(s, 0.5, 4.65, 9.0, "Four aggregated rotating-machinery fault classes")


def s_pipeline(s):
    blocks = [("MAFAULDA\ndata\n8ch · 50 kHz", BLUE),
              ("Preprocess\nwindow 1024\n+ z-score", BLUE),
              ("Transformer\nlightweight\n98.22%", PURPLE),
              ("int16\nquantization\n0% drop", PURPLE),
              ("FPGA core\nfixed-point\ninference", GOLD),
              ("UART → PC\nUI result\ndisplay", GREY)]
    x, gap = 0.5, 0.2
    w = (9.0 - gap * (len(blocks) - 1)) / len(blocks)
    for i, (t, c) in enumerate(blocks):
        block(s, x, 1.95, w, 1.55, t, c, size=9.5)
        if i < len(blocks) - 1:
            arrow(s, x + w, 2.72, x + w + gap, 2.72)
        x += w + gap
    x4 = 0.5 + 4 * w + 3.5 * gap
    card(s, 0.5, 3.9, x4 - 0.5, 0.16, fill=PURPLE, outline=None, rounded=False)
    caption(s, 0.5, 4.12, x4 - 0.5, "done — software + quantization",
            color=PURPLE, size=10)
    card(s, x4 + 0.1, 3.9, 9.5 - (x4 + 0.1), 0.16, fill=GOLD, outline=None, rounded=False)
    caption(s, x4 + 0.1, 4.12, 9.5 - (x4 + 0.1), "remaining — hardware",
            color="B8860B", size=10)
    # merged from the former "Project Goal" slide: subtitle + subtle class line
    textbox(s, 0.5, 1.12, 9.0, 0.4,
            [("4-class fault diagnosis with final inference on FPGA",
              12.5, False, PURPLE, BODY_FONT)], "M", "C")
    caption(s, 3.3, 3.5, 6.2, "Normal · Imbalance · Misalignment · Bearing Fault",
            color=GREY, size=9.5)


def s_progress(s):
    caption(s, 0.5, 1.2, 9.0, "Six workstreams — status labels only, not exact "
            "percentage completion", size=10.5)
    cards = [("Data Pipeline", "Completed", PURPLE),
             ("Model Tuning & Evaluation", "Completed", PURPLE),
             ("Quantization & Verification", "Completed", PURPLE),
             ("FPGA Model Freeze", "In progress", GOLD),
             ("HLS Inference Core", "Starting", GOLD),
             ("UART · PC UI · Integration", "Not started", GREY)]
    cw, ch, gx, gy = 2.88, 1.42, 0.18, 0.22
    x0, y0 = 0.6, 1.6
    for i, (name, status, c) in enumerate(cards):
        r, col = divmod(i, 3)
        x = x0 + col * (cw + gx); y = y0 + r * (ch + gy)
        card(s, x, y, cw, ch, fill=WHITE, outline=CHARCOAL, radius=0.10)
        textbox(s, x + 0.15, y + 0.16, cw - 0.3, 0.6,
                [(name, 12, True, INK, TITLE_FONT)], "M", "C")
        chip(s, x + cw / 2 - 0.85, y + ch - 0.5, 1.7, 0.34, status, c)


def s_dataset(s):
    stat = [("1,951", "CSV files", BLUE), ("8", "channels", PURPLE),
            ("50 kHz", "sampling", GOLD), ("~5 s", "per file", GREY)]
    x, w, gap = 0.6, 2.07, 0.28
    for big, small, c in stat:
        b = card(s, x, 1.35, w, 1.1, fill=c, outline=CHARCOAL, radius=0.10)
        fill_text(b.text_frame, [(big, 23, True, tcolor(c), TITLE_FONT),
                                 (small, 10.5, False, tcolor(c), BODY_FONT)], "M", "C")
        x += w + gap
    # sensor layout (left)
    textbox(s, 0.6, 2.7, 4.0, 0.3, [("Sensor layout", 12, True, PURPLE, TITLE_FONT)], "M", "L")
    block(s, 0.6, 3.1, 1.95, 1.15, "Underhang bearing\naxial / radial /\ntangential", BLUE, size=9)
    block(s, 2.65, 3.1, 1.95, 1.15, "Overhang bearing\naxial / radial /\ntangential", BLUE_D, size=9)
    block(s, 0.6, 4.35, 1.95, 0.6, "Tachometer", GREY, size=10)
    block(s, 2.65, 4.35, 1.95, 0.6, "Microphone", GREY, size=10)
    # four classes (right)
    textbox(s, 5.0, 2.7, 4.4, 0.3, [("Four project classes", 12, True, PURPLE, TITLE_FONT)], "M", "L")
    classes = [("Normal", BLUE), ("Imbalance", GOLD), ("Misalignment", GREY), ("Bearing fault", PURPLE)]
    y = 3.1
    for name, c in classes:
        block(s, 5.0, y, 4.4, 0.42, name, c, size=11)
        y += 0.47


def s_split(s):
    caption(s, 0.5, 1.18, 9.0, "File-level split BEFORE windowing  →  no leakage "
            "across sets", color=PURPLE, size=13, italic=False)
    block(s, 0.6, 1.7, 2.0, 0.95, "1,951\nCSV files", BLUE, size=12)
    arrow(s, 2.6, 2.17, 3.15, 2.17)
    block(s, 3.15, 1.7, 2.3, 0.95, "File-level\nsplit", PURPLE, size=12)
    arrow(s, 5.45, 2.17, 6.0, 2.17)
    block(s, 6.0, 1.7, 2.3, 0.95, "1024-sample\nwindows", GOLD, size=12)
    # split bar by window counts
    total = 333060 + 47580 + 95404
    segs = [("Train · 1,365 files · 333,060", 333060, BLUE_D),
            ("Val · 47,580", 47580, GOLD), ("Test · 391 · 95,404", 95404, PURPLE)]
    x0 = 0.6
    for lab, n, c in segs:
        w = 8.8 * n / total
        b = card(s, x0, 2.95, w, 0.7, fill=c, outline=CHARCOAL, line_w=1.4, radius=0.05)
        fs = 10 if w > 2 else 8
        fill_text(b.text_frame, [(lab, fs, True, tcolor(c), BODY_FONT)], "M", "C")
        x0 += w
    textbox(s, 0.6, 3.85, 8.8, 0.3,
            [("Window 1024 · stride 1024 · no overlap · [B, 1024, 8] "
              "· per-window z-score", 11.5, False, INK, BODY_FONT)], "M", "L")
    textbox(s, 0.6, 4.2, 5.2, 0.3,
            [("Class imbalance ~22× (bearing vs normal) → weighted cross-entropy",
              11.5, False, INK, BODY_FONT)], "M", "L")
    # mini imbalance bars (native shapes)
    counts = [("normal", 8296, BLUE), ("imbalance", 56852, GOLD),
              ("misalign", 85156, GREY), ("bearing", 182756, PURPLE)]
    x = 6.2; maxv = max(v for _, v, _ in counts)
    for k, v, c in counts:
        h = 0.6 * v / maxv
        card(s, x, 4.9 - h, 0.62, h, fill=c, outline=CHARCOAL, line_w=1.0, radius=0.12)
        caption(s, x - 0.1, 4.9 - h - 0.24, 0.82, f"{v // 1000}k", color=INK, size=8, italic=False)
        caption(s, x - 0.1, 4.92, 0.82, k, color=INK, size=8.5, italic=False)
        x += 0.78


def s_frequency(s):
    # Section A --- prior work / alternative path (neutral, no criticism)
    card(s, 0.4, 1.2, 2.75, 2.9, fill=WHITE, outline=CHARCOAL, line_w=1.9, radius=0.06)
    card(s, 0.4, 1.2, 0.13, 2.9, fill=BLUE, outline=None, rounded=False)
    textbox(s, 0.63, 1.3, 2.44, 2.7,
            [("Prior work — an alternative path", 10.5, True, BLUE_D, TITLE_FONT),
             ("A prior study reduced the sampling rate after spectral analysis.",
              9, False, INK, BODY_FONT),
             ("Motivation: lower compute cost, simpler processing.", 9, False, INK, BODY_FONT),
             ("It reported strong results in its own setup.", 9, False, INK, BODY_FONT),
             ("→ Downsampling is a valid, precedented path.", 9.3, True, BLUE_D, BODY_FONT)],
            "T", "L")
    # Section B --- the discriminative-band chart is the visual focus
    textbox(s, 3.3, 1.16, 3.15, 0.3,
            [("acc_axial_b1 · 22–23 kHz energy fraction", 10.5, True, PURPLE, TITLE_FONT)],
            "M", "C")
    bar_chart(s, 3.2, 1.5, 3.3, 2.05, ["Normal", "Imbalance", "Misalign", "Bearing"],
              [86.6, 83.2, 82.7, 3.2], [BLUE, GOLD, GREY, PURPLE], ymin=0, ymax=100,
              num_fmt='0.0"%"', label_size=8.5, axis_size=7.5, gap=45)
    tk = card(s, 3.3, 3.6, 3.15, 0.48, fill=CREAM, outline=GOLD, line_w=1.6, radius=0.12)
    fill_text(tk.text_frame,
              [("22–23 kHz band appears highly class-discriminative in acc_axial_b1",
                8.7, True, PURPLE, BODY_FONT)], "M", "C")
    # Section C --- current decision
    card(s, 6.6, 1.2, 2.9, 2.9, fill=WHITE, outline=CHARCOAL, line_w=1.9, radius=0.06)
    card(s, 6.6, 1.2, 0.13, 2.9, fill=GOLD, outline=None, rounded=False)
    textbox(s, 6.83, 1.28, 2.6, 2.74,
            [("Decision", 11, True, "B8860B", TITLE_FONT),
             ("Downsampling gains", 8.8, True, INK, BODY_FONT),
             ("lower data rate · throughput · simpler acquisition", 8.4, False, GREY, BODY_FONT),
             ("Risk if applied now", 8.8, True, INK, BODY_FONT),
             ("removes bands above the lower Nyquist limit", 8.4, False, GREY, BODY_FONT),
             ("22–23 kHz may carry discriminative information", 8.4, False, GREY, BODY_FONT),
             ("impact not yet experimentally validated", 8.4, False, GREY, BODY_FONT),
             ("→ 50 kHz baseline now · downsampling = future ablation",
              8.8, True, PURPLE, BODY_FONT)], "T", "L")
    # optional note + bold conclusion banner
    caption(s, 0.4, 4.14, 9.1,
            "Low-frequency dominance alone does not justify downsampling — classification "
            "utility depends on discriminative information, not total energy.",
            color=GREY, size=8.3)
    ban = card(s, 0.4, 4.5, 9.1, 0.56, fill=PURPLE, outline=CHARCOAL, line_w=1.75, radius=0.10)
    fill_text(ban.text_frame,
              [("Decision: Keep 50 kHz as the current baseline; evaluate downsampling only "
                "via controlled ablation.", 12.5, True, WHITE, TITLE_FONT)], "M", "C")


def s_architecture(s):
    chain = [("Input\n[B,1024,8]", BLUE), ("Linear\n8 → 32", BLUE),
             ("Positional\nencoding", BLUE), ("Encoder ×2\n(4 heads)", PURPLE),
             ("Mean\npooling", BLUE), ("Classifier\nLN → 16 → 4", BLUE),
             ("4 logits", GOLD)]
    x, gap = 0.5, 0.16
    w = (9.0 - gap * (len(chain) - 1)) / len(chain)
    for i, (t, c) in enumerate(chain):
        block(s, x, 2.0, w, 1.4, t, c, size=9)
        if i < len(chain) - 1:
            arrow(s, x + w, 2.7, x + w + gap, 2.7)
        x += w + gap
    spec = card(s, 1.2, 3.75, 7.6, 0.62, fill=PURPLE, outline=CHARCOAL, radius=0.14)
    fill_text(spec.text_frame,
              [("d_model = 32   ·   layers = 2   ·   heads = 4   ·   "
                "ffn = 64   ·   checkpoint 380 KB", 12, True, WHITE, TITLE_FONT)], "M", "C")
    caption(s, 0.5, 4.5, 9.0, "Selected software baseline after tuning · FPGA "
            "simplification candidate: 1 encoder layer.", color=PURPLE)


def s_tuning(s, rep):
    by = {r["name"]: r for r in rep["all_val_results"]}
    order = ["F_1layer", "A_baseline", "B_wide_ffn"]
    params = [by[n]["n_params"] for n in order]
    pcols = [GOLD, PURPLE, GREY]
    # left: native horizontal param bar
    textbox(s, 0.5, 1.2, 4.2, 0.3, [("Trainable parameters", 12, True, PURPLE, TITLE_FONT)], "M", "L")
    bar_chart(s, 0.4, 1.5, 4.3, 2.0, order, params, pcols, ymin=0,
              num_fmt="#,##0", label_size=9, axis_size=9, horizontal=True, gap=60)
    # right: native metric table
    cols = ["Config", "Val acc", "Macro F2", "Fault recall", "False alarm"]
    tbl = new_table(s, 4.95, 1.35, 4.55, 2.15, 4, 5)
    for c, h in enumerate(cols):
        set_cell(tbl.cell(0, c), h, 9.5, True, WHITE, PURPLE)
    best_col = {1: "F_1layer", 2: "A_baseline", 3: "A_baseline", 4: "A_baseline"}
    for ri, n in enumerate(order, start=1):
        r = by[n]
        vals = [n, f"{r['val_acc']*100:.2f}%", f"{r['val_macro_f2']:.4f}",
                f"{r['val_fault_recall']*100:.2f}%", f"{r['val_false_alarm']*100:.2f}%"]
        for c, v in enumerate(vals):
            hi = (c in best_col and best_col[c] == n)
            fill = CREAM if hi else (LIGHT if ri % 2 else WHITE)
            col = PURPLE if hi else INK
            set_cell(tbl.cell(ri, c), v, 9.5, (c == 0 or hi), col, fill)
    # badges
    b1 = card(s, 0.5, 3.75, 4.3, 0.55, fill=PURPLE, outline=CHARCOAL, radius=0.14)
    fill_text(b1.text_frame, [("Selected software baseline:  A_baseline", 11, True, WHITE, TITLE_FONT)], "M", "C")
    b2 = card(s, 5.0, 3.75, 4.4, 0.55, fill=GOLD, outline=CHARCOAL, radius=0.14)
    fill_text(b2.text_frame, [("Leading FPGA candidate:  F_1layer", 11, True, INK, TITLE_FONT)], "M", "C")
    caption(s, 0.5, 4.45, 9.0, "Validation tuning: 12 epochs · batch 64 · patience 3 "
            "· seed 42 · T4  —  A_baseline later reached 98.22% on the held-out "
            "test set (separate run).", size=9)


def s_training(s, hist):
    ep = [str(e) for e in hist["epoch"]]
    line_chart(s, 0.6, 1.35, 6.3, 3.35, ep,
               [("train accuracy", hist["train_acc"], BLUE_D),
                ("val accuracy", hist["val_acc"], GOLD)], ymin=0.85, ymax=1.0)
    k = kpi(s, 7.2, 1.6, 2.3, 1.15, "98.68%", "best val accuracy", PURPLE)
    card(s, 7.2, 2.95, 2.3, 1.5, fill=WHITE, outline=CHARCOAL, radius=0.08)
    textbox(s, 7.35, 3.08, 2.0, 1.3,
            [("Epoch 10", 12, True, PURPLE, TITLE_FONT),
             ("early stopping active", 10, False, INK, BODY_FONT),
             ("train & val converge — no large generalization gap", 9.5, False, GREY, BODY_FONT)],
            "T", "L")


def s_test(s, m):
    fn = m["test_f2_metrics"]["per_class"]
    labels = ["normal", "imbalance", "misalign", "bearing"]
    keys = ["normal", "imbalance", "misalignment", "bearing_fault"]
    recall = [round(fn[k]["recall"], 3) for k in keys]
    f2 = [round(fn[k]["f2"], 3) for k in keys]
    kpi(s, 0.5, 1.4, 2.0, 1.3, "98.22%", "Test accuracy", PURPLE)
    kpi(s, 0.5, 2.85, 2.0, 1.3, "0.978", "Macro F2", BLUE_D)
    caption(s, 0.4, 4.25, 2.2, "95,404 unseen\ntest windows", size=9)
    # native confusion-matrix table with heat shading
    cm = np.array(m["test_confusion_matrix"])
    vmax = cm.max()
    tbl = new_table(s, 2.75, 1.35, 3.3, 3.25, 5, 5)
    set_cell(tbl.cell(0, 0), "T \\ P", 8, True, WHITE, PURPLE)
    for j, lab in enumerate(labels):
        set_cell(tbl.cell(0, j + 1), lab, 8, True, WHITE, PURPLE)
        set_cell(tbl.cell(j + 1, 0), lab, 8, True, WHITE, PURPLE)
    for i in range(4):
        for j in range(4):
            v = int(cm[i, j]); t = v / vmax
            fill = _blend(WHITE, BLUE, t / 0.5) if t < 0.5 else _blend(BLUE, PURPLE, (t - 0.5) / 0.5)
            set_cell(tbl.cell(i + 1, j + 1), f"{v:,}", 8.5, (i == j),
                     tcolor(fill), fill)
    caption(s, 2.75, 4.5, 3.3, "Confusion matrix", color=PURPLE, size=10, italic=False)
    # native grouped bar recall/F2
    grouped_bar(s, 6.15, 1.4, 3.35, 3.0, labels,
                [("recall", recall, BLUE), ("F2", f2, GOLD)],
                ymin=0.9, ymax=1.0)
    caption(s, 6.15, 4.42, 3.35, "Per-class recall (blue) · F2 (gold)", color=PURPLE,
            size=10, italic=False)


def s_safety(s, m):
    fn = m["test_fault_vs_normal"]
    tp, miss = fn["captured_faults"], fn["missed_faults"]
    fp = fn["false_alarms"]; tn = fn["actual_normal"] - fp
    caption(s, 0.5, 1.2, 4.5, "Fault vs normal — rows = actual, columns = predicted",
            color=PURPLE, size=10.5, italic=False)
    cells = [(0, 0, "Captured fault", f"{tp:,}", PURPLE),
             (0, 1, "False alarm (FP)", f"{fp:,}", GOLD),
             (1, 0, "MISSED (FN)", f"{miss:,}", RED),
             (1, 1, "Correct normal", f"{tn:,}", BLUE)]
    cw, chh, x0, y0 = 2.05, 1.35, 0.6, 1.6
    for r, cc, lab, num, fill in cells:
        x = x0 + cc * (cw + 0.15); y = y0 + r * (chh + 0.15)
        b = card(s, x, y, cw, chh, fill=fill, outline=CHARCOAL, radius=0.08)
        tc = tcolor(fill)
        fill_text(b.text_frame, [(lab, 11, True, tc, TITLE_FONT),
                                 (num, 16, True, tc, TITLE_FONT)], "M", "C")
    kpi(s, 5.25, 1.6, 4.15, 1.15, f"{fn['fault_recall']*100:.2f}%", "Fault recall", PURPLE)
    kpi(s, 5.25, 2.9, 4.15, 1.15, f"{fn['false_alarm_rate']*100:.2f}%", "False-alarm rate", GOLD)
    caption(s, 5.25, 4.2, 4.15, "A missed fault (FN) is worse than a false alarm (FP) "
            "— recall on faulty windows is the primary safety metric.",
            color=INK, size=10)


def s_quant(s, q):
    fa = q["float32"]["accuracy"]
    rows = q["quantized_results"]
    names = ["float32"] + [r["name"].replace("int16_", "16 ").replace("int8_", "8 ")
                           .replace("per_tensor", "p-tensor").replace("per_channel", "p-chan")
                           for r in rows]
    accs = [round(fa, 4)] + [round(r["accuracy"], 4) for r in rows]
    colors = [PURPLE] + [BLUE if r["n_bits"] == 16 else GOLD for r in rows]
    bar_chart(s, 0.5, 1.4, 5.3, 3.1, names, accs, colors, ymin=0.975, ymax=0.985,
              num_fmt="0.00%", label_size=8, axis_size=8, gap=55)
    caption(s, 0.5, 4.5, 5.3, "Weight quantization — accuracy vs bit width (quant. eval run)",
            color=PURPLE, size=9.5, italic=False)
    chip(s, 6.0, 1.4, 3.45, 0.45, "Current baseline: INT16", BLUE)
    c1 = card(s, 6.0, 2.0, 3.45, 0.95, fill=WHITE, outline=CHARCOAL, radius=0.08)
    fill_text(c1.text_frame, [("INT16 today", 11.5, True, PURPLE, TITLE_FONT),
              ("all matrix multiplications", 10, False, INK, BODY_FONT)], "M", "C")
    c2 = card(s, 6.0, 3.05, 3.45, 1.05, fill=WHITE, outline=CHARCOAL, radius=0.08)
    fill_text(c2.text_frame, [("Still float in reference", 11.5, True, "B8860B", TITLE_FONT),
              ("Softmax · LayerNorm · GELU", 10, False, INK, BODY_FONT),
              ("→ need FPGA-compatible impl.", 9, False, GREY, BODY_FONT)], "M", "C")
    caption(s, 6.0, 4.2, 3.45, "98.22% full test · 98.14% quant run · 98.46% FP subset",
            color=GREY, size=8.5)


def s_golden(s, man):
    rows = []
    with open(GDIR / "golden_expected.txt") as f:
        for line in f:
            if not line.startswith("#"):
                p = line.split(); rows.append((int(p[0]), int(p[1]), int(p[2])))
    per_class = man["per_class"]
    names = {v: k for k, v in man["class_map"].items()}
    chip(s, 0.5, 1.25, 5.4, 0.42, "Golden vectors — 16 / 16 passed", PURPLE)
    # 4x4 PASS grid with row (class) labels
    cw, chh, gx, gy = 1.02, 0.62, 0.12, 0.14
    gx0, gy0 = 1.55, 1.85
    for idx, lab, pred in rows:
        col = idx % per_class; row = idx // per_class
        x = gx0 + col * (cw + gx); y = gy0 + row * (chh + gy)
        fill = PURPLE if lab == pred else RED
        block(s, x, y, cw, chh, "PASS", fill, size=9)
    for ci in range(4):
        textbox(s, 0.45, gy0 + ci * (chh + gy), 1.05, chh,
                [(names[ci], 8.5, True, INK, BODY_FONT)], "M", "R")
    # right: verification flow (native boxes + arrows)
    steps = [("PyTorch reference", BLUE), ("Fixed-point simulator", PURPLE),
             ("Golden vectors", GOLD), ("HLS co-simulation (future)", LIGHT)]
    x, w, h = 6.35, 3.1, 0.62
    y = 1.5
    for i, (t, c) in enumerate(steps):
        block(s, x, y, w, h, t, c, size=10, outline=(GREY if c == LIGHT else CHARCOAL))
        if i < len(steps) - 1:
            arrow(s, x + w / 2, y + h, x + w / 2, y + h + 0.24)
        y += h + 0.24
    caption(s, 0.5, 4.75, 9.0, "Match = exact class + |logit diff| < 0.05 · agreement vs "
            "PyTorch 2e-6 — sanity check; full proof needs HLS co-simulation.", size=9)


def s_hw(s, w_int16):
    ms = w_int16["memory_summary"]
    pe = next((int(np.prod(p["shape"])) * 4 for n, p in w_int16["parameters"].items()
               if n.endswith("pe")), 0)
    w_k = round(ms["int16_weight_bytes"] / 1024, 1)
    pe_k = round(pe / 1024, 1)
    names = ["Weights\n(int16)", "Pos.\nencoding", "Input\nbuffer", "Activation\nbuffer"]
    vals = [w_k, pe_k, 16, 64]
    cols = [BLUE, GOLD, BLUE_D, GREY]
    bar_chart(s, 0.5, 1.4, 4.9, 3.0, names, vals, cols, ymin=0,
              num_fmt="0", label_size=9, axis_size=8, gap=60)
    caption(s, 0.5, 4.4, 4.9, "Estimated on-chip memory per window (KiB, analytical)",
            color=PURPLE, size=9.5, italic=False)
    cal = card(s, 5.6, 1.5, 3.85, 2.4, fill=CREAM, outline=GOLD, line_w=2.25, radius=0.08)
    fill_text(cal.text_frame,
              [("Main bottleneck", 13, True, RED, TITLE_FONT),
               ("Attention scores are 1024×1024 PER HEAD", 11, True, INK, BODY_FONT),
               ("(~2 MiB/head at int16 if fully stored)", 10.5, False, INK, BODY_FONT),
               ("→ must be tiled / streamed, not materialized.", 11, True, PURPLE, BODY_FONT)],
              "M", "C")
    caption(s, 5.6, 4.05, 3.85, "Analytical estimates — NOT post-synthesis "
            "(no LUT/FF/BRAM/DSP yet).", size=9)


def s_fpga(s):
    chain = [("Input\nFIFO", BLUE_D), ("Input\nproj", BLUE), ("Pos.\nenc.", BLUE),
             ("Encoder\ntiled attn\nFFN · LN", PURPLE), ("Mean\npool", BLUE),
             ("Classifier\n+ argmax", BLUE), ("UART\nframe", GOLD), ("PC\nUI", GREY)]
    x, gap = 0.4, 0.14
    w = (9.2 - gap * (len(chain) - 1)) / len(chain)
    for i, (t, c) in enumerate(chain):
        block(s, x, 1.75, w, 1.4, t, c, size=8)
        if i < len(chain) - 1:
            arrow(s, x + w, 2.45, x + w + gap, 2.45, w=1.3)
        x += w + gap
    leg = [("int16 fixed-point MAC", BLUE), ("nonlinear op — impl. decision", GOLD),
           ("memory / streaming", BLUE_D), ("communication output", GREY)]
    for i, (t, c) in enumerate(leg):
        lx = 1.3 + (i % 2) * 4.4; ly = 3.5 + (i // 2) * 0.5
        card(s, lx, ly, 0.32, 0.3, fill=c, outline=CHARCOAL, line_w=1.0, radius=0.15)
        textbox(s, lx + 0.42, ly - 0.02, 3.6, 0.34, [(t, 10, False, INK, BODY_FONT)], "M", "L")
    caption(s, 0.5, 4.6, 9.0, "Initial path: HLS first; targeted optimization or RTL only if "
            "time permits.  ·  Proposed — not yet synthesized.", color=PURPLE, size=9.5)


def s_decisions(s):
    cards = [
        ("Encoder depth", "2 layers (18,036 p)", "1 layer (9,492 p)",
         "1 layer ≈ same val accuracy, ~half compute; 2 layers better safety metrics",
         "1-layer as FPGA candidate", BLUE),
        ("Normalization", "Per-window z-score", "Global train norm",
         "Per-window needs mean/var/sqrt/div in HW; global = stored constants + retraining",
         "Train & test a global-norm version", PURPLE),
        ("Numeric precision", "INT16", "INT8",
         "INT16 = safest margin (0% drop); INT8 ≤ 0.05% drop, less memory",
         "INT16 first, INT8 if resource pressure", GOLD),
        ("HW level", "HLS-first hybrid FP", "Full-integer / RTL",
         "HLS = fast correct baseline; RTL = more efficient but slower / riskier",
         "Validate HLS baseline first", BLUE_D),
    ]
    cw, chh, gx, gy = 4.35, 1.62, 0.2, 0.18
    x0, y0 = 0.55, 1.5
    for i, (name, cur, alt, trade, prop, c) in enumerate(cards):
        r, col = divmod(i, 2)
        x = x0 + col * (cw + gx); y = y0 + r * (chh + gy)
        card(s, x, y, cw, chh, fill=WHITE, outline=CHARCOAL, line_w=1.9, radius=0.06)
        card(s, x, y, 0.14, chh, fill=c, outline=None, rounded=False)  # accent bar
        textbox(s, x + 0.28, y + 0.1, cw - 0.4, chh - 0.2,
                [(name, 12.5, True, c, TITLE_FONT),
                 (f"Current: {cur}", 9.5, False, INK, BODY_FONT),
                 (f"Alternative: {alt}", 9.5, False, GREY, BODY_FONT),
                 (trade, 8.7, False, INK, BODY_FONT),
                 (f"→ {prop}", 10, True, c, BODY_FONT)], "T", "L")


def s_advisor(s):
    cards = [
        ("1  FPGA model depth",
         "1-layer cuts params 18,036 → 9,492 (98.28% vs 98.19% val acc), but 2-layer "
         "has better Macro F2 / recall / false-alarm. Make 1-layer the primary FPGA model "
         "and 2-layer the software reference?", BLUE),
        ("2  Normalization",
         "Should we prioritize hardware simplicity by retraining and evaluating with a "
         "global train-set normalization instead of per-window z-score?", PURPLE),
        ("3  Precision strategy",
         "Is an INT16 baseline + optional INT8 optimization an acceptable implementation "
         "strategy for the hardware deliverable?", GOLD),
        ("4  Implementation scope",
         "Is an HLS-first hybrid fixed-point implementation enough for the primary "
         "deliverable, with RTL only if time remains?", BLUE_D),
    ]
    cw, chh, gx, gy = 4.35, 1.62, 0.2, 0.18
    x0, y0 = 0.55, 1.5
    for i, (title, body, c) in enumerate(cards):
        r, col = divmod(i, 2)
        x = x0 + col * (cw + gx); y = y0 + r * (chh + gy)
        card(s, x, y, cw, chh, fill=WHITE, outline=CHARCOAL, line_w=1.9, radius=0.08)
        card(s, x, y, 0.14, chh, fill=c, outline=None, rounded=False)
        textbox(s, x + 0.3, y + 0.14, cw - 0.45, chh - 0.28,
                [(title, 13, True, c, TITLE_FONT),
                 (body, 10, False, INK, BODY_FONT)], "T", "L")


def s_milestones(s):
    phases = [("Model Freeze", ["Finalize variant", "Finalize norm & precision"], PURPLE),
              ("HLS Core", ["Freeze HW spec", "Export weights & golden", "Implement HLS blocks"], GOLD),
              ("Verification", ["C simulation", "HLS co-simulation", "Synthesis & resources"], BLUE_D),
              ("Comm & UI", ["UART output", "PC interface"], BLUE),
              ("Integration", ["End-to-end validate"], GREY)]
    x, w, gap = 0.5, 1.66, 0.15
    for i, (name, items, c) in enumerate(phases):
        block(s, x, 1.6, w, 0.7, name, c, size=10.5)
        if i < len(phases) - 1:
            arrow(s, x + w, 1.95, x + w + gap, 1.95)
        card(s, x, 2.45, w, 1.7, fill=WHITE, outline=CHARCOAL, radius=0.06)
        textbox(s, x + 0.08, 2.55, w - 0.16, 1.5,
                [("\n".join("• " + it for it in items), 9, False, INK, BODY_FONT)], "T", "L")
        x += w + gap
    caption(s, 0.5, 4.5, 9.0, "Completed (left) done · next work highlighted · later "
            "milestones muted · no fixed dates yet.")


def s_summary(s):
    done = ["Leakage-free data pipeline", "Architecture tuning — 3 configs evaluated",
            "Software baseline selected (A_baseline)", "1-layer FPGA candidate identified",
            "Wider FFN rejected (measured)", "Held-out test 98.22% · fault recall 99.75%",
            "INT16 quantization (0% drop) · INT8 compared",
            "Fixed-point reference + golden vectors 16/16"]
    todo = ["Final FPGA model selection", "Global-normalization experiment",
            "Hardware-compatible softmax / LayerNorm / GELU", "HLS core + C-sim + co-sim",
            "Synthesis + resources (LUT/FF/BRAM/DSP)", "UART + PC UI + integration"]
    chip(s, 0.55, 1.25, 4.3, 0.42, "DONE", PURPLE)
    chip(s, 5.15, 1.25, 4.3, 0.42, "TO-DO", GOLD)
    for i, t in enumerate(done):
        y = 1.85 + i * 0.4
        block(s, 0.55, y, 0.34, 0.34, "✓", PURPLE, size=11, radius=0.2)
        textbox(s, 1.0, y - 0.03, 3.85, 0.4, [(t, 10, False, INK, BODY_FONT)], "M", "L")
    for i, t in enumerate(todo):
        y = 1.85 + i * 0.4
        card(s, 5.15, y, 0.34, 0.34, fill=WHITE, outline=GOLD, line_w=1.8, radius=0.2)
        textbox(s, 5.6, y - 0.03, 3.85, 0.4, [(t, 10, False, INK, BODY_FONT)], "M", "L")


# ================================================================= build =====
def build():
    hist = json.load(open(HANDOFF / "history.json"))
    metrics = json.load(open(HANDOFF / "metrics.json"))
    quant = json.load(open(QDIR / "weight_quantization_accuracy.json"))
    w_int16 = json.load(open(QDIR / "transformer_int16_weights.json"))
    man = json.load(open(GDIR / "golden_manifest.json"))
    rep = json.load(open(PROJECT_ROOT / "artifacts" / "runs" /
                         "tuning_20260616_161108" / "tuning_final_report.json"))

    slides = [
        ("Today's Meeting", lambda s: s_agenda(s)),
        ("End-to-End System Pipeline", lambda s: s_pipeline(s)),
        ("Progress at a Glance", lambda s: s_progress(s)),
        ("MAFAULDA Dataset", lambda s: s_dataset(s)),
        ("Leakage-Free Data Split and Windowing", lambda s: s_split(s)),
        ("Frequency Analysis and Sampling-Rate Decision", lambda s: s_frequency(s)),
        ("Selected Software Baseline Architecture", lambda s: s_architecture(s)),
        ("Model Tuning and Configuration Selection", lambda s: s_tuning(s, rep)),
        ("Final A_baseline Training Run", lambda s: s_training(s, hist)),
        ("Held-Out Test Performance", lambda s: s_test(s, metrics)),
        ("Safety-Oriented Evaluation", lambda s: s_safety(s, metrics)),
        ("Quantization Results", lambda s: s_quant(s, quant)),
        ("Fixed-Point Reference and Golden Vectors", lambda s: s_golden(s, man)),
        ("Hardware Feasibility and Main Bottlenecks", lambda s: s_hw(s, w_int16)),
        ("Proposed FPGA Inference Architecture", lambda s: s_fpga(s)),
        ("Key Design Decisions Before HLS", lambda s: s_decisions(s)),
        ("Decisions Requested from the Advisor", lambda s: s_advisor(s)),
        ("Next Milestones", lambda s: s_milestones(s)),
        ("Progress Summary", lambda s: s_summary(s)),
    ]

    prs = Presentation(str(DECK))
    prs.slide_width = Inches(10); prs.slide_height = Inches(5.625)

    # Drop every slide except the title (index 0). Remove the sldId entries
    # first -- that alone determines the deck's slide list and cannot add
    # slides -- then drop the now-orphan relationships so no stray parts remain.
    sldIdLst = prs.slides._sldIdLst
    old = list(sldIdLst)[1:]
    for sid in old:
        sldIdLst.remove(sid)
    assert len(list(prs.slides)) == 1, "removal left extra slides"
    for sid in old:
        prs.part.drop_rel(sid.get(qn("r:id")))

    blank = next((l for l in prs.slide_layouts if l.name.upper() == "BLANK"),
                 prs.slide_layouts[-1])
    total = len(slides) + 1
    for i, (title, draw) in enumerate(slides):
        s = prs.slides.add_slide(blank)
        title_block(s, title)
        draw(s)
        footer(s, i + 2, total)

    n = len(list(prs.slides))
    assert n == total, f"expected {total} slides, got {n}"
    prs.save(str(DECK))
    print(f"native deck assembled: {total} slides (title preserved) -> {DECK.name}")


if __name__ == "__main__":
    build()
