"""
Smoke test for the MAFAULDA preprocessing pipeline.

Tests:
1. Build DataLoaders with max_files=20 (fast subset)
2. Verify X and y tensor shapes match model expectations
3. Pass one batch through TransformerBaseline
4. Print per-split class distribution from stats
5. Save preprocessing config JSON artifact

Usage:
    python -m src.tools.test_pipeline
"""

from __future__ import annotations

import sys
import traceback

import torch

from src.config import NUM_CLASSES, PROJECT_ROOT
from src.data.make_dataloaders import build_dataloaders
from src.data.pipeline_validation import (
    save_preprocessing_config,
    validate_batch_shape,
)
from src.models.transformer_baseline import TransformerBaseline


_MAX_FILES = 20  # small subset for a fast smoke test


def _print_class_counts(label: str, counts: dict) -> None:
    total = sum(counts.values())
    print(f"\n  {label} ({total} windows):")
    for name, count in counts.items():
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"    {name}: {count:>5d}  ({pct:.1f}%)")


def main() -> int:
    print("=" * 60)
    print("Pipeline smoke test")
    print(f"max_files={_MAX_FILES}")
    print("=" * 60)

    # --- Step 1: build DataLoaders ---
    print("\n[1] Building DataLoaders ...")
    try:
        train_loader, val_loader, test_loader, stats = build_dataloaders(max_files=_MAX_FILES)
    except Exception as exc:
        print(f"[FAIL] build_dataloaders raised: {exc}")
        traceback.print_exc()
        return 1

    print(f"  train_samples: {stats['train_samples']}")
    print(f"  val_samples:   {stats['val_samples']}")
    print(f"  test_samples:  {stats['test_samples']}")
    print(f"  window_shape:  {stats['window_shape']}")

    # --- Step 2: per-split class distribution from stats ---
    print("\n[2] Class distribution from stats:")
    _print_class_counts("train", stats["train_class_window_counts"])
    _print_class_counts("val",   stats["val_class_window_counts"])
    _print_class_counts("test",  stats["test_class_window_counts"])

    # --- Step 3: batch shape validation ---
    print("\n[3] Validating batch shapes ...")
    shape_ok = validate_batch_shape(train_loader)
    if not shape_ok:
        print("[FAIL] Batch shape validation failed.")
        return 1

    # --- Step 4: forward pass through TransformerBaseline ---
    print("\n[4] Running forward pass through TransformerBaseline ...")
    device = torch.device("cpu")
    model = TransformerBaseline().to(device)
    model.eval()

    try:
        x_batch, y_batch = next(iter(train_loader))
        x_batch = x_batch.to(device)
        with torch.no_grad():
            logits = model(x_batch)
    except Exception as exc:
        print(f"[FAIL] Forward pass raised: {exc}")
        traceback.print_exc()
        return 1

    expected_out = (x_batch.shape[0], NUM_CLASSES)
    if tuple(logits.shape) != expected_out:
        print(f"[FAIL] Unexpected logits shape: {tuple(logits.shape)} (expected {expected_out})")
        return 1

    print(f"  logits shape: {tuple(logits.shape)}  [batch_size={x_batch.shape[0]}, num_classes={NUM_CLASSES}]")
    print("  Forward pass OK.")

    # --- Step 5: save preprocessing config ---
    print("\n[5] Saving preprocessing config ...")
    config_path = save_preprocessing_config()
    print(f"  Saved: {config_path}")

    print("\n" + "=" * 60)
    print("Smoke test PASSED.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
