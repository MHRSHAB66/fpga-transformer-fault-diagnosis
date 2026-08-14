"""
Exploratory: full-integer accuracy at different bit-widths (int16 vs int8).

This is a NON-DESTRUCTIVE variant of ``evaluate_integer.py``. It never edits the
verified int16 pipeline; it monkeypatches the module-level bit-width constants of
``integer_model`` at runtime and rebuilds the model/LUTs/scales for each width, so
weights, activations and softmax probabilities are all quantized to ``bits``.

Both widths are evaluated on the SAME calibration and test subsets for a fair
comparison. Results are written to artifacts/quantized/integer_bits_sweep.json.
Nothing here changes the Stage-6f int16 artifacts.

Run
---
    python -m src.quantize.evaluate_integer_bits --bits 16 8 --calib-files 40 --test-files 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import src.quantize.integer_model as im
from src.quantize.integer_model import IntegerTransformer, CalibStats
from src.quantize.evaluate_integer import (
    build_train_subset, int_logits, float_logits, accuracy, OUTPUT_DIR,
)
from src.quantize.evaluate_quantized import DEFAULT_CHECKPOINT, load_model, build_test_subset
from src.config import NUM_CLASSES, RANDOM_SEED

REPORT_PATH = OUTPUT_DIR / "integer_bits_sweep.json"


def set_bits(bits: int) -> None:
    """Override every quantization bit-width in integer_model (runtime only)."""
    im.ACT_BITS = bits
    im.W_BITS = bits
    im.ATTN_BITS = bits
    # EXP_FRAC stays 15: internal fixed-point precision of the exp LUT, which is a
    # ROM constant, not an activation width. Leaving it fixed isolates the effect
    # of the tensor bit-width itself.


def calibrate_bits(model: IntegerTransformer, X: np.ndarray, bits: int, batch: int = 8) -> dict:
    stats = CalibStats.empty()
    for s in range(0, len(X), batch):
        model.collect(X[s:s + batch].astype(np.float64), stats)
    return stats.to_scales(act_bits=bits)          # explicit: default arg is baked at import


def run_one(bits: int, state_np: dict, model_cfg: dict,
            Xc: np.ndarray, Xt: np.ndarray, yt: np.ndarray) -> dict:
    set_bits(bits)
    model = IntegerTransformer(state_np, model_cfg)          # re-quantizes weights at `bits`
    scales = calibrate_bits(model, Xc, bits)
    model.set_scales(scales)

    fl = float_logits(model, Xt)
    float_preds = fl.argmax(1)
    float_acc, _ = accuracy(float_preds, yt)

    t0 = time.perf_counter()
    il = int_logits(model, Xt)
    int_preds = il.argmax(1)
    int_acc, int_pc = accuracy(int_preds, yt)
    agree = float((int_preds == float_preds).mean())
    dt = time.perf_counter() - t0

    print(f"[int{bits:>2}]  acc = {int_acc:.4f}   drop_vs_float = {float_acc-int_acc:+.4f}   "
          f"agreement = {agree:.4f}   ({dt:.1f}s)")
    return {
        "bits": bits,
        "float_accuracy": float_acc,
        "int_accuracy": int_acc,
        "accuracy_drop_vs_float": float_acc - int_acc,
        "prediction_agreement_vs_float": agree,
        "per_class_accuracy": int_pc,
    }


def main():
    ap = argparse.ArgumentParser(description="Integer accuracy at multiple bit-widths.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 8])
    ap.add_argument("--calib-files", type=int, default=40)
    ap.add_argument("--test-files", type=int, default=60)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    device = torch.device("cpu")
    ref_model, float_state, model_cfg = load_model(args.checkpoint, device)
    state_np = {k: v.cpu().numpy() for k, v in float_state.items()}

    print(f"[calib] building train subset ({args.calib_files} files)...")
    Xc, _, kc = build_train_subset(args.calib_files, args.seed)
    print(f"[calib] {len(Xc)} calibration windows from {kc} files")

    print(f"[eval] building test subset ({args.test_files} files)...")
    Xt, yt, sub = build_test_subset(args.test_files, args.seed)
    print(f"[eval] {sub['windows']} windows from {sub['test_files_used']} files\n")

    results = [run_one(b, state_np, model_cfg, Xc, Xt, yt) for b in args.bits]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "6f_bit_sweep_exploratory",
            "checkpoint": str(args.checkpoint),
            "note": "Non-destructive bit-width sweep on A_baseline; int16 Stage-6f "
                    "artifacts are unchanged. Weights+activations+softmax all at `bits`.",
            "test_subset": sub,
            "results": results,
        }, f, indent=2)
    print(f"\n[info] sweep report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
