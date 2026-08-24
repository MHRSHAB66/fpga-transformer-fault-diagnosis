"""
Build the advisor presentation by editing the Slidesgo template's own slides.

The template's designed infographic slides (donuts, timelines, process boxes,
big numbers) are reused as-is; only their text is swapped for the project
content, matched by the template's placeholder strings so nested group text is
handled too. Persian runs switch to B Nazanin (English keeps the template
font). Data-figure slides keep the template title/background and drop in a
project figure (from figures_v2, already in the template palette).

Run
---
    python -m src.tools.build_figures
    python -m src.tools.build_presentation
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from src.config import PROJECT_ROOT

FOOTER_GOLD = RGBColor(0xF1, 0xC2, 0x32)
FOOTER_GREY = RGBColor(0x88, 0x88, 0x88)
FOOTER_TEXT = "Machinery Fault Diagnosis on FPGA"
LOGO_RATIO = 960 / 1164  # width / height of AUT.png

TEMPLATE = PROJECT_ROOT / "presntation" / "Annual Report Infographics by Slidesgo.pptx"
HANDOFF = PROJECT_ROOT / "docs" / "slide_handoff"
FIG = HANDOFF / "figures_v2"
OUTPUT = HANDOFF / "FPGA_Fault_Diagnosis_Progress.pptx"
LOGO = HANDOFF / "assets" / "AUT.png"
GROWTH = HANDOFF / "assets" / "growth.png"

PERSIAN_FONT = "B Nazanin"


def has_persian(s: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in s)


def _iter_text_shapes(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_text_shapes(sh.shapes)
        elif sh.has_text_frame:
            yield sh


def set_text(shape, text, size=None, persian_bump=6):
    tf = shape.text_frame
    lines = text.split("\n")
    p0 = tf.paragraphs[0]
    base_run = p0.runs[0] if p0.runs else p0.add_run()
    for r in list(p0.runs)[1:]:
        r._r.getparent().remove(r._r)
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)

    def style(run, para, txt):
        run.text = txt
        if has_persian(txt):
            para._p.get_or_add_pPr().set("rtl", "1")
            rPr = run._r.get_or_add_rPr()
            latin = rPr.find(qn("a:latin"))
            cs = rPr.find(qn("a:cs"))
            if cs is None:
                cs = (latin if latin is not None else rPr).makeelement(qn("a:cs"), {})
                latin.addnext(cs) if latin is not None else rPr.append(cs)
            cs.set("typeface", PERSIAN_FONT)
            if size is None and run.font.size:
                run.font.size = Pt(run.font.size.pt + persian_bump)
        if size is not None:
            run.font.size = Pt(size)

    style(base_run, p0, lines[0])
    for extra in lines[1:]:
        p = tf.add_paragraph()
        style(p.add_run(), p, extra)


def set_title(slide, text):
    for sh in _iter_text_shapes(slide.shapes):
        if sh.text_frame.text.strip() in ("Annual Report Infographics", "Icon pack"):
            set_text(sh, text)
            return


def replace_by_text(slide, mapping):
    """Replace text where the template text contains a key (substring match)."""
    for sh in _iter_text_shapes(slide.shapes):
        cur = sh.text_frame.text.strip()
        for key, new in mapping.items():
            if key in cur:
                set_text(sh, new)
                break


def clear_texts(slide, keys):
    for sh in _iter_text_shapes(slide.shapes):
        cur = sh.text_frame.text.strip()
        if any(k in cur for k in keys):
            for r in list(sh.text_frame.paragraphs[0].runs):
                r._r.getparent().remove(r._r)


def keep_only_title(slide):
    """Remove every top-level shape except the title placeholder."""
    for sh in list(slide.shapes):
        txt = sh.text_frame.text.strip() if sh.has_text_frame else ""
        if txt in ("Annual Report Infographics", "Icon pack"):
            continue
        sh._element.getparent().remove(sh._element)


def add_image(slide, path, bx, by, bw, bh):
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    if ar > bw / bh:
        w = bw; h = int(bw / ar)
    else:
        h = bh; w = int(bh * ar)
    slide.shapes.add_picture(str(path), bx + (bw - w) // 2, by + (bh - h) // 2,
                             width=w, height=h)


def figure_slide(slide, title, image, caption=None):
    keep_only_title(slide)   # keep the template title placeholder first ...
    set_title(slide, title)  # ... then rename it, so it is not removed
    add_image(slide, image, Inches(0.5), Inches(1.15), Inches(9.0), Inches(3.5))
    if caption:
        box = slide.shapes.add_textbox(Inches(0.5), Inches(4.72), Inches(9.0), Inches(0.4))
        set_text(box, caption, size=13)
        box.text_frame.paragraphs[0].alignment = 2  # center


def add_logo(slide, big=False):
    """Place the AUT logo: large top-left on the title, small top-right elsewhere."""
    if not LOGO.exists():
        return
    if big:
        h = Inches(1.5)
        slide.shapes.add_picture(str(LOGO), Inches(0.55), Inches(0.3), height=h)
    else:
        h = Inches(0.5)
        w = Inches(0.5 * LOGO_RATIO)
        slide.shapes.add_picture(str(LOGO), Inches(10) - w - Inches(0.25),
                                 Inches(0.12), height=h)


def add_footer(slide, page, total):
    """Thin bottom rule + footer label (left) and page number (right)."""
    line = slide.shapes.add_shape(1, Inches(0.5), Inches(5.28), Inches(9.0), Pt(1.4))
    line.fill.solid(); line.fill.fore_color.rgb = FOOTER_GOLD
    line.line.fill.background(); line.shadow.inherit = False
    lb = slide.shapes.add_textbox(Inches(0.5), Inches(5.33), Inches(6.5), Inches(0.26))
    r = lb.text_frame.paragraphs[0].add_run(); r.text = FOOTER_TEXT
    r.font.name = "Times New Roman"; r.font.size = Pt(8); r.font.color.rgb = FOOTER_GREY
    pb = slide.shapes.add_textbox(Inches(7.9), Inches(5.33), Inches(1.6), Inches(0.26))
    pp = pb.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.RIGHT
    r2 = pp.add_run(); r2.text = f"{page} / {total}"
    r2.font.name = "Times New Roman"; r2.font.size = Pt(8); r2.font.color.rgb = FOOTER_GREY


CLR = ["This template has been created by Slidesgo", "Follow the link"]

# --------------------------------------------------------------------------

def s_title(slide):
    # Rework the template title slide: bigger AUT logo (added later), the growth
    # icon replacing the Slidesgo bar chart, title + subtitle moved up, and the
    # names placed directly under the subtitle (not at the very bottom).
    for sh in list(slide.shapes):
        t = sh.text_frame.text.strip() if sh.has_text_frame else ""
        if t.startswith("Annual"):
            set_text(sh, "Real-Time Machinery\nCondition Monitoring on FPGA", size=26)
            sh.left, sh.top = Inches(0.8), Inches(2.0)
            sh.width, sh.height = Inches(5.4), Inches(1.7)
            sh.text_frame.vertical_anchor = MSO_ANCHOR.TOP
        elif t.startswith("Here is where"):
            set_text(sh, "Transformer-based fault diagnosis — progress report", size=14)
            sh.left, sh.top = Inches(0.8), Inches(3.55)
            sh.width, sh.height = Inches(5.4), Inches(0.6)
        elif t.startswith("This template"):
            set_text(sh, "Mehrdad Sheikh Abbasi", size=13)
            p = sh.text_frame.add_paragraph()
            r = p.add_run(); r.text = "Advisor: Dr. Morteza Saheb-Zamani"
            r.font.name = "Times New Roman"; r.font.size = Pt(12); r.font.color.rgb = FOOTER_GREY
            sh.left, sh.top = Inches(0.8), Inches(4.3)
            sh.width, sh.height = Inches(5.4), Inches(0.8)
            for para in sh.text_frame.paragraphs:
                para.alignment = PP_ALIGN.LEFT
        elif t in ("Q1", "Q2", "Q3", "Q4"):
            sh._element.getparent().remove(sh._element)
        elif sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            sh._element.getparent().remove(sh._element)  # drop template bar chart
    if GROWTH.exists():
        add_image(slide, GROWTH, Inches(6.0), Inches(1.5), Inches(3.4), Inches(3.0))


def s_pipeline(slide):
    set_title(slide, "Data-to-Result Pipeline")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "H1": "Data", "H2": "Model",
        "Mercury is the closest": "Raw 8-channel MAFAULDA signal, 50 kHz",
        "Venus has a beautiful": "1024-sample windows + per-window norm",
        "Despite being red": "Lightweight Transformer - 98.22%",
        "Jupiter is the biggest": "FPGA inference -> fault class",
        "Mercury": "MAFAULDA", "Venus": "Preprocess",
        "Mars": "Transformer", "Jupiter": "FPGA",
    })


def s_classes(slide):
    set_title(slide, "Fault Classes")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "It’s the closest": "Healthy machine",
        "Venus is the second": "Rotor mass imbalance",
        "Mars is actually": "Shaft misalignment",
        "Jupiter is the biggest": "Bearing fault",
        "Mercury": "Normal", "Venus": "Imbalance",
        "Mars": "Misalignment", "Jupiter": "Bearing",
        "100%": "4",
    })
    clear_texts(slide, ["40%", "15%", "20%", "25%"])  # avoid misleading donut shares


def s_dataset_info(slide):  # T18 three-row donut infographic
    set_title(slide, "Dataset: MAFAULDA")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "It’s the closest": "CSV, 8 channels, 50 kHz",
        "Venus is the second": "windows from unseen files (test)",
        "Mars is actually": "bearing vs normal - window-level",
        "Mercury": "1,951 files", "Venus": "95,404", "Mars": "22x ratio",
        "98%": "1,951", "80%": "95k", "50%": "22x",
    })


def s_dataset(slide):  # T16 big numbers (unused)
    set_title(slide, "Dataset: MAFAULDA  /  دیتاست")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "$125.00M": "1,951", "+567": "98.22%", "$15M": "95,404",
        "Venus is the second": "فایل CSV، ۸ کانال، ۵۰ kHz — تقسیم file-level",
        "Mars is actually": "دقت test روی داده دیده‌نشده",
        "Venus": "Files", "Mars": "Accuracy",
        "15%": "22×", "$125.00M": "1,951",
    })


def s_preprocess(slide):  # T4 donut + 4 callouts
    set_title(slide, "Preprocessing & Windowing")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "It’s the closest": "1024-sample window, stride 1024",
        "Venus is the second": "Per-window z-score normalization",
        "Mars is actually": "File-level split (no leakage)",
        "Jupiter is the biggest": "All 8 channels used",
        "Mercury": "Windowing", "Venus": "Normalize",
        "Mars": "Split", "Jupiter": "Channels",
    })
    clear_texts(slide, ["Q1", "Q2", "Q3", "Q4"])


def s_freq(slide):  # T18 three rows
    set_title(slide, "Frequency & Channels  /  تحلیل فرکانسی")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "It’s the closest": "رزونانس ~۲۲ kHz تمایزدهنده است",
        "Venus is the second": "نرخ ۵۰ kHz حفظ شد (بدون resample)",
        "Mars is actually": "هر ۸ کانال اطلاعات مفید دارند",
        "Mercury": "22 kHz", "Venus": "50 kHz", "Mars": "8 ch",
        "98%": "22k", "80%": "50k", "50%": "8ch",
    })


def s_architecture(slide):  # T8 timeline circles
    set_title(slide, "Lightweight Transformer")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "Venus is the second": "Input [B,1024,8] -> Linear 8->32",
        "It’s the biggest": "2 encoder layers (4 heads, ffn=64)",
        "Despite being red": "Mean pooling over time",
        "Saturn is a gas giant and has several": "Classifier -> 4 classes (380 KB)",
        "Venus": "Input", "Jupiter": "Encoder", "Mars": "Pooling", "Saturn": "Head",
        "V": "1", "J": "2", "M": "3", "S": "4",
    })


def s_safety(slide):  # T10 four donuts %
    set_title(slide, "Safety Metrics  /  معیارهای ایمنی")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "Venus is the second": "خطای از دست دادن خرابی بسیار پایین",
        "It’s the biggest": "نرخ هشدار اشتباه پایین",
        "Despite being red": "۲۳۶ از ۹۲٬۹۶۴ خرابی missed",
        "Saturn is a gas giant and has several": "۵۷ از ۲٬۴۴۰ normal اشتباه",
        "Venus": "Fault recall", "Jupiter": "False alarm",
        "Mars": "Missed", "Saturn": "False+",
        "98%": "99.75%", "80%": "2.34%", "75%": "236", "50%": "57",
    })
    clear_texts(slide, ["Q1", "Q2", "Q3", "Q4"])


def s_summary(slide):  # T15 H1/H2 bars
    set_title(slide, "Summary  /  جمع‌بندی")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "Mercury": "Software", "Venus": "Quantize",
        "Mars": "Golden", "Jupiter": "FPGA/UI",
        "H1": "Done", "H2": "To-do",
        "50%": "100%", "75%": "100%", "25%": "100%", "98%": "100%",
        "100%": "0%", "60%": "0%",
    })


def s_roadmap(slide):  # T7 gantt
    set_title(slide, "Project Roadmap")
    clear_texts(slide, CLR)
    replace_by_text(slide, {
        "Mercury is very small": "Transformer 98.22% (done)",
        "Mercury is a small planet": "Evaluation (done)",
        "Mercury is small": "Data + preprocessing (done)",
        "Venus has a toxic atmosphere": "Golden vectors (done)",
        "Venus is very hot": "int16 quantization (done)",
        "Despite being red, Mars is very cold": "UART + PC UI (to-do)",
        "Mars is full of iron oxide dust": "Integration (to-do)",
        "Mars is very cold": "HLS core (to-do)",
        "Mercury": "Software", "Venus": "Quantize", "Mars": "Hardware",
    })


def build():
    prs = Presentation(str(TEMPLATE))

    # (template_index, editor). Conceptual slides keep the template graphics and
    # relabel text; figure slides keep the title/background and drop in a figure.
    plan = [
        (0, s_title),
        (6, s_pipeline),
        (2, s_classes),
        (18, s_dataset_info),
        (4, s_preprocess),
        (3, lambda s: figure_slide(s, "Frequency Analysis",
             FIG / "frequency_evidence.png",
             "95% of vibration spectral energy below ~22 kHz · 50 kHz kept (> 44 kHz Nyquist) · no resampling")),
        (8, s_architecture),
        (11, lambda s: figure_slide(s, "Training Curves",
             FIG / "training_accuracy_curve.png",
             "16 epochs · early stopping at epoch 10 · train/val converge, no overfitting")),
        (5, lambda s: figure_slide(s, "Test Results",
             FIG / "test_results.png",
             "Accuracy 98.22% · recall — normal 97.7%, bearing 97.8% · Macro F2 = 0.978")),
        (10, lambda s: figure_slide(s, "Safety Metrics",
             FIG / "safety.png", None)),
        (1, lambda s: figure_slide(s, "Quantization (int16)",
             FIG / "fixed_point_scheme.png",
             "All matmuls -> int16 (zero drop) · softmax/LayerNorm/GELU still float -> remaining FPGA work")),
        (13, lambda s: figure_slide(s, "Quantization Accuracy",
             FIG / "quant_accuracy_comparison.png",
             "int16 = zero accuracy drop (per-tensor) -> selected · int8 <= 0.05%")),
        (9, lambda s: figure_slide(s, "Golden Vectors",
             FIG / "golden_validation.png",
             "16 vectors (4 per class) · float vs int16 · match: exact class + |logit|<0.05 · sanity check (full proof = HLS co-sim)")),
        (12, lambda s: figure_slide(s, "FPGA Memory",
             FIG / "memory_breakdown.png",
             "PE ~4x the weights · attention 1024x1024 must be streamed · next: resource synthesis")),
        (16, lambda s: figure_slide(s, "FPGA Plan",
             FIG / "fpga_plan.png",
             "HLS first for speed, then RTL for comparison if time permits")),
        (7, s_roadmap),
        (15, lambda s: figure_slide(s, "Summary",
             FIG / "summary_status.png",
             "Real software/quantization progress; the whole hardware path (HLS/UART/UI) remains")),
    ]

    for idx, fn in plan:
        fn(prs.slides[idx])

    sldIdLst = prs.slides._sldIdLst
    ids = list(sldIdLst)
    used = [idx for idx, _ in plan]
    for i, sid in enumerate(ids):
        if i not in used:
            prs.part.drop_rel(sid.get(qn("r:id")))
        sldIdLst.remove(sid)
    for idx in used:
        sldIdLst.append(ids[idx])

    # Footer + page number on every slide except the title (in final order).
    slides = list(prs.slides)
    total = len(slides)
    for i, s in enumerate(slides):
        if i == 0:
            add_logo(s, big=True)
            continue
        add_logo(s)
        add_footer(s, i + 1, total)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(f"[info] saved: {OUTPUT}  ({len(list(prs.slides))} slides)")


if __name__ == "__main__":
    build()
