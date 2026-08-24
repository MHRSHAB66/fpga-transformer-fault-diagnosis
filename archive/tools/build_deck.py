"""
Assemble the 21-slide advisor deck ON TOP of the existing title slide.

The hand-edited title slide (index 0) is preserved untouched. Every other
slide is rebuilt as: centered Montserrat title + one project figure + the
existing footer (gold rule, AUT logo bottom-left, "Machinery Fault Diagnosis on
FPGA", page number). All figures come from figures_v2 and use only real data.

Run (with the .pptx CLOSED in PowerPoint), after regenerating figures:
    python -m src.tools.build_figures
    python -m src.tools.deck_figures
    python -m src.tools.build_deck
"""

from __future__ import annotations

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from src.config import PROJECT_ROOT

HANDOFF = PROJECT_ROOT / "docs" / "slide_handoff"
FIG = HANDOFF / "figures_v2"
DECK = HANDOFF / "FPGA_Fault_Diagnosis_Progress.pptx"
LOGO = HANDOFF / "assets" / "AUT_en.png"

GOLD = RGBColor(0xF1, 0xC2, 0x32)
GREY = RGBColor(0x88, 0x88, 0x88)
BLACK = RGBColor(0x14, 0x14, 0x14)
FOOTER_TEXT = "Machinery Fault Diagnosis on FPGA"
TITLE_FONT = "Montserrat"

# (title, figure filename, optional caption)
SLIDES = [
    ("Today's Meeting", "agenda.png", None),
    ("Project Goal and Core Constraint", "goal.png", None),
    ("End-to-End System Pipeline", "system_pipeline.png", None),
    ("Progress at a Glance", "progress_glance.png", None),
    ("MAFAULDA Dataset", "dataset_full.png", None),
    ("Leakage-Free Data Split and Windowing", "split_windowing.png", None),
    ("Frequency Analysis", "frequency_evidence.png",
     "~95% of vibration spectral energy is below ~22 kHz · 50 kHz retained; no resampling applied."),
    ("Selected Software Baseline Architecture", "architecture_diagram.png",
     "Selected software baseline after tuning · FPGA simplification candidate: 1 encoder layer."),
    ("Model Tuning and Configuration Selection", "tuning_comparison.png", None),
    ("Final A_baseline Training Run", "training_behavior.png", None),
    ("Held-Out Test Performance", "test_results.png", None),
    ("Safety-Oriented Evaluation", "safety.png", None),
    ("Quantization Results", "quant_results.png", None),
    ("Fixed-Point Reference and Golden Vectors", "golden_flow.png", None),
    ("Hardware Feasibility and Main Bottlenecks", "hw_feasibility.png", None),
    ("Proposed FPGA Inference Architecture", "fpga_architecture.png", None),
    ("Key Design Decisions Before HLS", "decision_matrix.png", None),
    ("Decisions Requested from the Advisor", "decisions_cards.png", None),
    ("Next Milestones", "milestones.png", None),
    ("Progress Summary", "summary_status.png", None),
]


def _blank_layout(prs):
    for lay in prs.slide_layouts:
        if lay.name.upper() == "BLANK":
            return lay
    return prs.slide_layouts[-1]


def _title(slide, text):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.32), Inches(9.0), Inches(0.7))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.name = TITLE_FONT; r.font.size = Pt(22); r.font.bold = True
    r.font.color.rgb = BLACK


def _image(slide, path, caption):
    box_h = Inches(3.45) if caption else Inches(3.75)
    bx, by, bw, bh = Inches(0.5), Inches(1.15), Inches(9.0), box_h
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    if ar > bw / bh:
        w = bw; h = int(bw / ar)
    else:
        h = bh; w = int(bh * ar)
    slide.shapes.add_picture(str(path), bx + (bw - w) // 2, by + (bh - h) // 2,
                             width=w, height=h)
    if caption:
        cb = slide.shapes.add_textbox(Inches(0.5), Inches(4.72), Inches(9.0), Inches(0.4))
        cp = cb.text_frame.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
        cr = cp.add_run(); cr.text = caption
        cr.font.name = "Times New Roman"; cr.font.size = Pt(12)
        cr.font.italic = True; cr.font.color.rgb = RGBColor(0x28, 0x15, 0x38)


def _footer(slide, page, total):
    rule = slide.shapes.add_shape(1, Inches(0.5), Inches(5.28), Inches(9.0), Pt(1.4))
    rule.fill.solid(); rule.fill.fore_color.rgb = GOLD
    rule.line.fill.background(); rule.shadow.inherit = False
    if LOGO.exists():
        slide.shapes.add_picture(str(LOGO), Inches(0.4), Inches(5.34), height=Inches(0.24))
    lb = slide.shapes.add_textbox(Inches(0.75), Inches(5.34), Inches(6.3), Inches(0.24))
    lb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = lb.text_frame.paragraphs[0].add_run(); r.text = FOOTER_TEXT
    r.font.name = "Times New Roman"; r.font.size = Pt(11); r.font.color.rgb = GREY
    pb = slide.shapes.add_textbox(Inches(7.9), Inches(5.34), Inches(1.6), Inches(0.24))
    pb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    pp = pb.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.RIGHT
    r2 = pp.add_run(); r2.text = f"{page} / {total}"
    r2.font.name = "Times New Roman"; r2.font.size = Pt(11); r2.font.color.rgb = GREY


def build():
    prs = Presentation(str(DECK))
    prs.slide_width = Inches(10); prs.slide_height = Inches(5.625)

    # remove every slide except the title (index 0)
    sldIdLst = prs.slides._sldIdLst
    ids = list(sldIdLst)
    for sid in ids[1:]:
        prs.part.drop_rel(sid.get(qn("r:id")))
        sldIdLst.remove(sid)

    layout = _blank_layout(prs)
    total = len(SLIDES) + 1  # + title
    for i, (title, fig, caption) in enumerate(SLIDES):
        s = prs.slides.add_slide(layout)
        _title(s, title)
        _image(s, FIG / fig, caption)
        _footer(s, i + 2, total)  # title is page 1

    prs.save(str(DECK))
    print(f"assembled {total} slides (title preserved) -> {DECK.name}")


if __name__ == "__main__":
    build()
