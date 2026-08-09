"""
Validate the NumPy inference engine against the Stage 6d golden vectors.

Checks that:
1. Every predicted class matches the fixed-point simulator reference (exact).
2. Every logit is within the approved tolerance of the reference logits.

Run
---
    python -m src.pynq.validate

Expected output (all 16 golden vectors should pass):
    [validate] 16/16 class predictions correct
    [validate] 16/16 logits within tolerance (|delta| <= 0.05)
    [validate] PASS
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT
from src.pynq.numpy_inference import NumpyTransformer, CLASS_NAMES

GOLDEN_DIR = PROJECT_ROOT / "artifacts" / "golden"
WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "pynq" / "numpy_weights.npz"
LOGIT_TOLERANCE = 0.05


def validate(weights_path: Path = WEIGHTS_PATH,
             golden_dir: Path = GOLDEN_DIR) -> bool:

    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"[validate] weights file not found: {weights_path}")
        print("[validate] run `python -m src.pynq.export_weights` first")
        return False

    print(f"[validate] loading model from {weights_path}")
    model = NumpyTransformer.load(weights_path)

    golden = np.load(golden_dir / "golden_vectors.npz")
    normalized_inputs = golden["normalized_input"]   # [16, 1024, 8]
    expected_classes  = golden["fixed_point_pred"]   # [16]
    expected_logits   = golden["fixed_point_logits"] # [16, 4]
    true_labels       = golden["label_id"]           # [16]

    with open(golden_dir / "golden_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    n = manifest["num_vectors"]
    tol = manifest["match_criteria"]["logit_abs_tolerance"]

    print(f"[validate] {n} golden vectors  (logit tolerance ±{tol})")
    print()
    print(f"  {'#':>3}  {'true':<16} {'expected':<16} {'predicted':<16}  class  logit_max_err")
    print("  " + "-" * 72)

    class_ok = 0
    logit_ok = 0

    for i in range(n):
        class_id, label, probs, logits = model.predict(normalized_inputs[i])
        exp_class = int(expected_classes[i])
        exp_logits = expected_logits[i]
        true_label = CLASS_NAMES[int(true_labels[i])]

        c_match = class_id == exp_class
        max_err = float(np.abs(logits - exp_logits).max())
        l_match = max_err <= tol

        if c_match:
            class_ok += 1
        if l_match:
            logit_ok += 1

        class_mark = "ok " if c_match else "ERR"
        logit_mark = f"{max_err:.4f}" if l_match else f"{max_err:.4f} !"
        print(f"  {i:>3}  {true_label:<16} {CLASS_NAMES[exp_class]:<16} {label:<16}"
              f"  {class_mark}  {logit_mark}")

    print()
    print(f"[validate] {class_ok}/{n} class predictions correct")
    print(f"[validate] {logit_ok}/{n} logits within tolerance (|delta| <= {tol})")

    passed = class_ok == n and logit_ok == n
    print(f"[validate] {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    import sys
    ok = validate()
    sys.exit(0 if ok else 1)
