"""
One-off: tidy the bottom-left footer — logo clearly below the gold rule.

Edits the existing deck in place; the title slide (index 0) is left untouched.
On every other slide it:
  - lifts the gold footer rule up a little so there is room beneath it,
  - removes the old small corner logo (any picture under 1 inch wide),
  - drops the English AUT logo below the rule (bottom-left),
  - moves the footer label further right so it sits well clear of the logo,
  - vertically centers label + page number on the logo.

Run (with the .pptx CLOSED in PowerPoint):
    python -m src.tools.update_corner_logo
"""

from __future__ import annotations

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

from src.config import PROJECT_ROOT

DECK = PROJECT_ROOT / "docs" / "slide_handoff" / "FPGA_Fault_Diagnosis_Progress.pptx"
LOGO = PROJECT_ROOT / "docs" / "slide_handoff" / "assets" / "AUT_en.png"

RULE_Y = Inches(5.28)     # gold rule at its original position (do not move)
ROW_Y = Inches(5.34)      # logo + text row, just below the rule
ROW_H = Inches(0.24)
LOGO_X = Inches(0.4)
TEXT_X = Inches(0.75)     # just right of the logo, small gap, no overlap


def main():
    prs = Presentation(str(DECK))
    for i, slide in enumerate(prs.slides):
        if i == 0:
            continue  # keep the manually edited title slide as-is
        for sh in list(slide.shapes):
            # lift the thin, wide gold rule
            if (sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE
                    and sh.width > Inches(5) and sh.height < Inches(0.1)):
                sh.top = RULE_Y
            # drop the old small corner logo
            elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE and sh.width < Inches(1.0):
                sh._element.getparent().remove(sh._element)

        for sh in slide.shapes:
            if not sh.has_text_frame:
                continue
            t = sh.text_frame.text.strip()
            if t.startswith("Machinery"):
                sh.left, sh.top = TEXT_X, ROW_Y
                sh.width, sh.height = Inches(6.3), ROW_H
                sh.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(11)
            elif "/" in t and t.replace("/", "").replace(" ", "").isdigit():
                sh.top, sh.height = ROW_Y, ROW_H
                sh.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(11)

        slide.shapes.add_picture(str(LOGO), LOGO_X, ROW_Y, height=ROW_H)

    prs.save(str(DECK))
    print(f"updated {len(list(prs.slides))} slides (title untouched)")


if __name__ == "__main__":
    main()
