"""
One-off: replace slide 2's template infographic with a real pipeline figure.

The Slidesgo "process" infographic (H1/H2 boxes + bank/people/document icons)
does not represent the data pipeline and its icons are finance-themed. This
strips that artwork from slide index 1 while keeping the title, the footer
(logo + label + page number) and the gold rule, then drops in the clean
end-to-end ``system_pipeline.png``. Other slides and the title slide are
untouched.

Run (with the .pptx CLOSED in PowerPoint):
    python -m src.tools.update_slide2_pipeline
"""

from __future__ import annotations

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches

from src.config import PROJECT_ROOT

DECK = PROJECT_ROOT / "docs" / "slide_handoff" / "FPGA_Fault_Diagnosis_Progress.pptx"
FIGURE = PROJECT_ROOT / "docs" / "slide_handoff" / "figures_v2" / "system_pipeline.png"
TITLE = "Data-to-Result Pipeline"


def _is_page_number(t):
    return "/" in t and t.replace("/", "").replace(" ", "").isdigit()


def main():
    prs = Presentation(str(DECK))
    slide = prs.slides[1]  # slide 2

    for sh in list(slide.shapes):
        if sh.has_text_frame:
            t = sh.text_frame.text.strip()
            if t == TITLE or t.startswith("Machinery") or _is_page_number(t):
                continue  # keep title + footer label + page number
            sh._element.getparent().remove(sh._element)
        elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE and sh.width < Inches(1.0):
            continue  # keep the corner logo
        elif (sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE
              and sh.width > Inches(5) and sh.height < Inches(0.1)):
            continue  # keep the gold rule
        else:
            sh._element.getparent().remove(sh._element)  # drop the infographic

    # add the pipeline figure, fit into the content area
    bx, by, bw, bh = Inches(0.5), Inches(1.2), Inches(9.0), Inches(3.6)
    with Image.open(FIGURE) as im:
        iw, ih = im.size
    ar = iw / ih
    if ar > bw / bh:
        w = bw; h = int(bw / ar)
    else:
        h = bh; w = int(bh * ar)
    slide.shapes.add_picture(str(FIGURE), bx + (bw - w) // 2, by + (bh - h) // 2,
                             width=w, height=h)

    prs.save(str(DECK))
    print("slide 2 replaced with system_pipeline figure")


if __name__ == "__main__":
    main()
