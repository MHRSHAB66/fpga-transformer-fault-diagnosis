"""Check live-run logits/CONF against the Python INT10 reference of the deployed model.

For every frame in a live_frames.jsonl (host.live_fpga evidence): the 4 logits
must equal the export-scale reference logits of that dataset index, the class
must be the lowest-index argmax of them, and the recorded CONF must equal the
softmax value recomputed with the deployed logits scale (6 decimals as stored).

Usage: python -m tools.check_live_logits_vs_reference <live_frames.jsonl> --out <check.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from host.confidence import SCALES_PATH, confidence_from_logits, logits_scale

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "artifacts" / "results" / "board_vs_reference_20261022" / "numpy_reference" / "numpy_export_logits.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ref = np.loadtxt(REF, dtype=np.int64)[:, 1:]
    n = logit_eq = cls_eq = conf_eq = no_logits = 0
    for line in a.frames.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        n += 1
        if r.get("logits") is None:
            no_logits += 1
            continue
        lg, c, i = np.array(r["logits"]), int(r["predicted_class"]), int(r["dataset_index"])
        logit_eq += bool((lg == ref[i]).all())
        cls_eq += c == int(lg.argmax())
        conf_eq += abs(round(confidence_from_logits(lg, c), 6) - float(r["conf"])) < 1e-9
    res = {"frames": str(a.frames), "n": n, "frames_without_logits": no_logits,
           "logits_equal_reference": logit_eq, "class_equals_argmax_logits": cls_eq,
           "conf_equals_recomputed": conf_eq, "reference": str(REF.relative_to(ROOT)),
           "scales": str(Path(SCALES_PATH).relative_to(ROOT)), "logits_scale": logits_scale()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
