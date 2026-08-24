"""
Stage 6a/6e — Post-training weight quantization: accuracy comparison.

Loads the frozen float32 checkpoint, simulates fixed-point *weight* quantization
at several bit widths (fake-quant: quantize -> dequantize), and reports the
accuracy cost versus the float32 model on a real subset of the unseen test
split. This is the first, most defensible quantization step: it answers
"how much accuracy do we lose by storing weights as int16 / int8?" before any
activation quantization or hardware work.

Scope and honesty notes
------------------------
- WEIGHT-ONLY quantization. Activations, LayerNorm, softmax, and GELU still run
  in float here. Full integer/fixed-point activation simulation and golden
  vectors are later Stage 6 sub-steps; this script does not claim hardware
  parity.
- Only 2-D weight matrices (the matmul weights, including attention
  ``in_proj_weight``) are quantized. LayerNorm affine params, biases, and the
  fixed positional-encoding buffer are left in float (standard practice: these
  are kept at higher precision in hardware).
- The data split, normalization, and checkpoint are unchanged. A subset of the
  real test files is used only to keep the run fast; the split itself is the
  same deterministic file-level split (seed fixed in ``make_dataloaders``).

Run
---
    python -m src.quantize.evaluate_quantized                 # default 60 test files
    python -m src.quantize.evaluate_quantized --test-files 120
    python -m src.quantize.evaluate_quantized --all-test-files
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import (
    DATA_RAW_DIR,
    PROJECT_ROOT,
    NORMALIZATION_METHOD,
    NORMALIZATION_SCOPE,
    LABEL_MAP,
    NUM_CLASSES,
    RANDOM_SEED,
)
from src.data.load_data import build_file_index
from src.data.make_dataloaders import split_file_index, build_X_y_from_file_index
from src.models.transformer_baseline import TransformerBaseline
from src.quantize.fixed_point import fake_quantize

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "20260615_223905_fpga_fault_final_t4"
    / "checkpoints"
    / "transformer_best.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "quantized"

# Quantization configurations to compare. per_channel=True uses one scale per
# output feature (more accurate, still cheap in HW); per_channel=False uses a
# single scale per weight matrix (simplest possible HW).
QUANT_CONFIGS = [
    {"name": "int16_per_tensor", "n_bits": 16, "per_channel": False},
    {"name": "int16_per_channel", "n_bits": 16, "per_channel": True},
    {"name": "int8_per_tensor", "n_bits": 8, "per_channel": False},
    {"name": "int8_per_channel", "n_bits": 8, "per_channel": True},
]

LABEL_NAMES_BY_ID = {v: k for k, v in LABEL_MAP.items()}


def load_model(checkpoint_path: Path, device: torch.device):
    """Rebuild the Transformer from the config embedded in the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt:
        raise ValueError(
            "Checkpoint has no embedded model_config; refusing to guess "
            "architecture. Use a checkpoint saved by the current training script."
        )
    cfg = ckpt["model_config"]
    model = TransformerBaseline(
        input_dim=cfg["input_dim"],
        seq_len=cfg["seq_len"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    float_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, float_state, cfg


def quantizable_weight_names(state_dict) -> list[str]:
    """
    Names of the 2-D weight matrices that get quantized.

    A 2-D parameter is exactly a matmul weight (Linear weights and the attention
    ``in_proj_weight``). This rule excludes 1-D LayerNorm/bias params and the
    3-D positional-encoding buffer, which stay in float.
    """
    return [k for k, v in state_dict.items() if v.dim() == 2]


def apply_weight_quantization(model, float_state, n_bits, per_channel):
    """
    Load the pristine float weights, then fake-quantize every 2-D weight matrix
    in place. Returns stats about how many tensors/params were quantized.
    """
    model.load_state_dict(float_state)
    quantized = 0
    quantized_params = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() == 2:
                param.copy_(
                    fake_quantize(param.data, n_bits, per_channel=per_channel, ch_axis=0)
                )
                quantized += 1
                quantized_params += param.numel()
    return {"tensors_quantized": quantized, "params_quantized": quantized_params}


def build_test_subset(num_files, seed):
    """
    Materialize windows from a subset of the *real* test split.

    The full file index is split with the same deterministic file-level split
    used in training/evaluation, then ``num_files`` test files are sampled. This
    keeps the float baseline representative (~test accuracy) while staying fast.
    """
    index_df = build_file_index(DATA_RAW_DIR)
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    _, _, test_index_df = split_file_index(index_df)

    total_test_files = len(test_index_df)
    if num_files is not None and num_files < total_test_files:
        test_index_df = test_index_df.sample(
            n=num_files, random_state=seed
        ).reset_index(drop=True)

    X, y, kept, dropped, _ = build_X_y_from_file_index(
        test_index_df,
        normalization_scope=NORMALIZATION_SCOPE,
        normalization_method=NORMALIZATION_METHOD,
    )
    return X, y, {
        "total_test_files": int(total_test_files),
        "test_files_used": int(kept),
        "test_files_dropped": int(dropped),
        "windows": int(len(y)),
    }


@torch.no_grad()
def evaluate(model, X, y, device, batch_size=256):
    """Return overall accuracy and per-class accuracy for the given windows."""
    model.eval()
    X_t = torch.from_numpy(X)
    preds = np.empty(len(y), dtype=np.int64)
    for start in range(0, len(X_t), batch_size):
        xb = X_t[start : start + batch_size].to(device)
        logits = model(xb)
        preds[start : start + batch_size] = logits.argmax(dim=1).cpu().numpy()

    overall = float((preds == y).mean())
    per_class = {}
    for cls_id in range(NUM_CLASSES):
        mask = y == cls_id
        name = LABEL_NAMES_BY_ID.get(cls_id, f"class_{cls_id}")
        per_class[name] = (
            float((preds[mask] == cls_id).mean()) if mask.any() else None
        )
    return overall, per_class, preds


def main():
    parser = argparse.ArgumentParser(description="Stage 6 weight-quantization accuracy comparison.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-files", type=int, default=60,
                        help="Number of real test files to sample (default 60).")
    parser.add_argument("--all-test-files", action="store_true",
                        help="Use every test file instead of a subset.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    print(f"[info] checkpoint: {args.checkpoint}")

    model, float_state, model_cfg = load_model(args.checkpoint, device)
    weight_names = quantizable_weight_names(float_state)
    print(f"[info] quantizable 2-D weight matrices: {len(weight_names)}")

    num_files = None if args.all_test_files else args.test_files
    print(f"[info] building test subset "
          f"({'all' if num_files is None else num_files} test files)...")
    t0 = time.perf_counter()
    X, y, subset_info = build_test_subset(num_files, args.seed)
    print(f"[info] subset: {subset_info['windows']} windows from "
          f"{subset_info['test_files_used']} files "
          f"({time.perf_counter() - t0:.1f}s)")

    # Float32 baseline on this subset.
    float_acc, float_per_class, _ = evaluate(model, X, y, device, args.batch_size)
    print(f"\n[float32] accuracy = {float_acc:.4f}")

    results = []
    for cfg in QUANT_CONFIGS:
        qinfo = apply_weight_quantization(
            model, float_state, cfg["n_bits"], cfg["per_channel"]
        )
        acc, per_class, _ = evaluate(model, X, y, device, args.batch_size)
        drop = float_acc - acc
        results.append({
            "name": cfg["name"],
            "n_bits": cfg["n_bits"],
            "per_channel": cfg["per_channel"],
            "accuracy": acc,
            "accuracy_drop_vs_float": drop,
            "per_class_accuracy": per_class,
            **qinfo,
        })
        print(f"[{cfg['name']:>18}] acc = {acc:.4f}   drop = {drop:+.4f}")

    # Restore pristine float weights so the in-memory model is left clean.
    model.load_state_dict(float_state)

    report = {
        "stage": "6a_weight_quantization_accuracy",
        "scope_note": "weight-only fake-quant; activations/LayerNorm/softmax/GELU still float",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "model_config": model_cfg,
        "normalization_scope": NORMALIZATION_SCOPE,
        "test_subset": subset_info,
        "quantized_weight_tensors": weight_names,
        "float32": {"accuracy": float_acc, "per_class_accuracy": float_per_class},
        "quantized_results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "weight_quantization_accuracy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[info] report saved: {out_path}")

    # Compact summary table.
    print("\n=== Weight-quantization accuracy summary ===")
    print(f"{'config':>18} | {'acc':>7} | {'drop':>8}")
    print("-" * 42)
    print(f"{'float32':>18} | {float_acc:7.4f} | {'--':>8}")
    for r in results:
        print(f"{r['name']:>18} | {r['accuracy']:7.4f} | {r['accuracy_drop_vs_float']:+8.4f}")


if __name__ == "__main__":
    main()
