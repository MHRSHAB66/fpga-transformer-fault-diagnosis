"""
Stage 6b — Validate and evaluate the fixed-point simulator.

Three checks, in order:

1. NUMERICAL CROSS-CHECK: run the manual forward pass with quantization OFF and
   confirm it matches the real PyTorch model's logits to a tight tolerance. This
   proves the hand-unrolled attention/LayerNorm/GELU path is faithful before any
   precision is reduced.
2. ACCURACY: evaluate the int16 fixed-point path on a real subset of the test
   split and compare accuracy against the float32 model.
3. REPORT: save everything to artifacts/quantized/fixed_point_accuracy.json.

Run
---
    python -m src.quantize.evaluate_fixed_point --test-files 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, NUM_CLASSES, LABEL_MAP, RANDOM_SEED
from src.quantize.evaluate_quantized import (
    DEFAULT_CHECKPOINT,
    load_model,
    build_test_subset,
)
from src.quantize.fixed_point_model import FixedPointTransformer, FixedPointConfig

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "quantized"
LABEL_NAMES_BY_ID = {v: k for k, v in LABEL_MAP.items()}


@torch.no_grad()
def batched_logits(fp_model, X, device, batch_size=128):
    X_t = torch.from_numpy(X).float()
    out = np.empty((len(X), NUM_CLASSES), dtype=np.float32)
    for start in range(0, len(X_t), batch_size):
        xb = X_t[start:start + batch_size].to(device)
        out[start:start + batch_size] = fp_model.forward(xb).cpu().numpy()
    return out


def accuracy_and_per_class(logits, y):
    preds = logits.argmax(axis=1)
    overall = float((preds == y).mean())
    per_class = {}
    for cls_id in range(NUM_CLASSES):
        mask = y == cls_id
        name = LABEL_NAMES_BY_ID.get(cls_id, f"class_{cls_id}")
        per_class[name] = float((preds[mask] == cls_id).mean()) if mask.any() else None
    return overall, per_class, preds


def main():
    parser = argparse.ArgumentParser(description="Stage 6b fixed-point simulator eval.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-files", type=int, default=60)
    parser.add_argument("--all-test-files", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--act-bits", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    ref_model, float_state, model_cfg = load_model(args.checkpoint, device)

    # --- 1. numerical cross-check (quant OFF must match the real model) -----
    torch.manual_seed(0)
    x_probe = torch.randn(4, model_cfg["seq_len"], model_cfg["input_dim"], device=device)
    ref_model.eval()
    with torch.no_grad():
        ref_logits = ref_model(x_probe).cpu().numpy()
    fp_float = FixedPointTransformer(float_state, model_cfg,
                                     FixedPointConfig(weight_bits=None, act_bits=None))
    fp_logits = fp_float.forward(x_probe).cpu().numpy()
    max_abs_diff = float(np.abs(ref_logits - fp_logits).max())
    same_argmax = bool((ref_logits.argmax(1) == fp_logits.argmax(1)).all())
    print(f"[check] manual-vs-pytorch max|logit diff| = {max_abs_diff:.3e}, "
          f"argmax match = {same_argmax}")
    if max_abs_diff > 1e-3 or not same_argmax:
        raise SystemExit("[FAIL] fixed-point float path does not match PyTorch model.")

    # --- 2. build test subset ----------------------------------------------
    num_files = None if args.all_test_files else args.test_files
    print(f"[info] building test subset "
          f"({'all' if num_files is None else num_files} files)...")
    t0 = time.perf_counter()
    X, y, subset_info = build_test_subset(num_files, args.seed)
    print(f"[info] {subset_info['windows']} windows from "
          f"{subset_info['test_files_used']} files ({time.perf_counter() - t0:.1f}s)")

    # float32 reference accuracy (real model)
    ref_model.eval()
    with torch.no_grad():
        ref_all = np.empty((len(X), NUM_CLASSES), dtype=np.float32)
        Xt = torch.from_numpy(X).float()
        for s in range(0, len(Xt), 256):
            ref_all[s:s + 256] = ref_model(Xt[s:s + 256].to(device)).cpu().numpy()
    float_acc, float_pc, _ = accuracy_and_per_class(ref_all, y)
    print(f"\n[float32]        acc = {float_acc:.4f}")

    # --- 3. fixed-point int16 path -----------------------------------------
    fp_cfg = FixedPointConfig(weight_bits=args.weight_bits, act_bits=args.act_bits)
    fp_model = FixedPointTransformer(float_state, model_cfg, fp_cfg)
    t1 = time.perf_counter()
    fp_all = batched_logits(fp_model, X, device)
    fp_acc, fp_pc, fp_preds = accuracy_and_per_class(fp_all, y)
    ref_preds = ref_all.argmax(axis=1)
    agreement = float((fp_preds == ref_preds).mean())
    print(f"[fixed-point w{args.weight_bits}/a{args.act_bits}] "
          f"acc = {fp_acc:.4f}   drop = {float_acc - fp_acc:+.4f}   "
          f"pred-agreement w/ float = {agreement:.4f}  "
          f"({time.perf_counter() - t1:.1f}s)")

    report = {
        "stage": "6b_fixed_point_simulation",
        "scheme": "hybrid: int matmul, float softmax/LayerNorm/GELU (HLS-style)",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "numerical_crosscheck": {
            "max_abs_logit_diff_vs_pytorch": max_abs_diff,
            "argmax_match": same_argmax,
        },
        "test_subset": subset_info,
        "float32": {"accuracy": float_acc, "per_class_accuracy": float_pc},
        "fixed_point": {
            "accuracy": fp_acc,
            "accuracy_drop_vs_float": float_acc - fp_acc,
            "prediction_agreement_vs_float": agreement,
            "per_class_accuracy": fp_pc,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "fixed_point_accuracy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[info] report saved: {out_path}")


if __name__ == "__main__":
    main()
