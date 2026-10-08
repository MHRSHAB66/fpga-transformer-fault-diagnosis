"""
Stage 10 -- FPGA -> PC frame protocol (text).

The FPGA sends one line per classified window:

    FRAME=<id>,CLASS=<class_id>,LABEL=<label>,CONF=<confidence>\n

- ``id``        : monotonically increasing window counter (wraps at 65536).
- ``class_id``  : 0..3, MUST match src.config.LABEL_MAP.
- ``label``     : human-readable class name (redundant with class_id; for logs).
- ``confidence``: 0..1 float if the core exposes it, else the sentinel ``NA``.

Keeping it plain text makes the link trivially debuggable with any serial
terminal before a binary protocol is ever considered (per the project plan).
This module is the single source of truth for formatting and parsing so the
mock source, the receiver and any tests agree byte-for-byte.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# class_id -> label, mirrors src.config.LABEL_MAP (kept local so the PC tool has
# no dependency on the training package / torch).
LABELS = {0: "normal", 1: "imbalance", 2: "misalignment", 3: "bearing_fault"}
LABEL_TO_ID = {v: k for k, v in LABELS.items()}

ID_WRAP = 1 << 16  # frame id wraps at 65536 (16-bit counter on the FPGA)

_FRAME_RE = re.compile(
    r"^FRAME=(?P<id>\d+),CLASS=(?P<cls>\d+),LABEL=(?P<label>[A-Za-z_]+),"
    r"CONF=(?P<conf>NA|\d+(?:\.\d+)?)\s*$"
)


@dataclass
class Frame:
    frame_id: int
    class_id: int
    label: str
    confidence: Optional[float]  # None when the core sent the NA sentinel

    def format(self) -> str:
        conf = "NA" if self.confidence is None else f"{self.confidence:.3f}"
        return (f"FRAME={self.frame_id},CLASS={self.class_id},"
                f"LABEL={self.label},CONF={conf}\n")


def format_frame(frame_id: int, class_id: int,
                 confidence: Optional[float] = None) -> str:
    """Build a wire line from a prediction (label derived from class_id)."""
    if class_id not in LABELS:
        raise ValueError(f"class_id {class_id} outside 0..{len(LABELS)-1}")
    return Frame(frame_id % ID_WRAP, class_id, LABELS[class_id], confidence).format()


class ParseError(ValueError):
    """Raised for a line that does not match the frame grammar."""


def parse_frame(line: str) -> Frame:
    """Parse one wire line into a Frame, validating the class/label contract."""
    m = _FRAME_RE.match(line.strip())
    if not m:
        raise ParseError(f"malformed frame: {line!r}")
    class_id = int(m["cls"])
    label = m["label"]
    if class_id not in LABELS:
        raise ParseError(f"class_id {class_id} out of range in {line!r}")
    if LABELS[class_id] != label:
        raise ParseError(
            f"label/class mismatch: CLASS={class_id} but LABEL={label} in {line!r}")
    conf = None if m["conf"] == "NA" else float(m["conf"])
    if conf is not None and not (0.0 <= conf <= 1.0):
        raise ParseError(f"confidence {conf} outside [0,1] in {line!r}")
    return Frame(int(m["id"]), class_id, label, conf)


def gap_after(prev_id: Optional[int], cur_id: int) -> int:
    """Number of dropped frames implied by two consecutive ids (wrap-aware)."""
    if prev_id is None:
        return 0
    return (cur_id - prev_id - 1) % ID_WRAP
