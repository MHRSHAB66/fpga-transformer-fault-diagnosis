"""
Validation utilities for the MAFAULDA preprocessing pipeline.

Provides functions to inspect class distribution, verify tensor shapes,
and export the active preprocessing configuration as a JSON artifact.

Usage:
    from src.data.pipeline_validation import (
        print_split_summary,
        validate_batch_shape,
        save_preprocessing_config,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DATALOADER_NUM_WORKERS,
    EPSILON,
    EXPECTED_COLUMNS,
    EXPECTED_NUM_COLUMNS,
    INCLUDE_TAIL_WINDOW,
    LABEL_MAP,
    LOW_ENERGY_CHANNEL_IDX,
    LOW_ENERGY_FILTER_ENABLED,
    LOW_ENERGY_SEGMENT_SIZE,
    LOW_ENERGY_THRESHOLD_RMS,
    LOWPASS_CUTOFF_HZ,
    LOWPASS_FILTER_ENABLED,
    LOWPASS_FILTER_ORDER,
    MAFAULDA_SAMPLE_RATE,
    MISSING_FILL_METHOD,
    MISSING_RATIO_THRESHOLD,
    NOISE_INJECTION_STD,
    NORMALIZATION_METHOD,
    NUM_CLASSES,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESAMPLE_ENABLED,
    RESAMPLE_TARGET_SR,
    TAIL_WINDOW_MODE,
    TEST_RATIO,
    TRAIN_RATIO,
    USE_NOISE_INJECTION,
    USE_SMOTE_AUGMENTATION,
    VAL_RATIO,
    WIENER_FILTER_ENABLED,
    WIENER_KERNEL_SIZE,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)


def compute_class_distribution(loader: DataLoader) -> Dict[str, int]:
    """
    Count windows per class by scanning all batches.

    Parameters
    ----------
    loader : DataLoader
        A DataLoader whose second element per batch is integer class labels.

    Returns
    -------
    dict
        Mapping from label name to window count.
    """
    label_names_by_id = {v: k for k, v in LABEL_MAP.items()}
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for _, y_batch in loader:
        y_np = y_batch.cpu().numpy()
        for label_id in y_np:
            if 0 <= label_id < NUM_CLASSES:
                counts[label_id] += 1

    return {label_names_by_id.get(i, f"class_{i}"): int(counts[i]) for i in range(NUM_CLASSES)}


def validate_batch_shape(
    loader: DataLoader,
    expected_window_size: int = WINDOW_SIZE,
    expected_channels: int = EXPECTED_NUM_COLUMNS,
) -> bool:
    """
    Retrieve one batch and verify its shape matches the expected Transformer input.

    Parameters
    ----------
    loader : DataLoader
        Training or validation DataLoader.
    expected_window_size : int
        Expected number of time steps per window.
    expected_channels : int
        Expected number of channels (features) per time step.

    Returns
    -------
    bool
        True if batch shape is [batch_size, window_size, channels]; False otherwise.
    """
    try:
        x_batch, y_batch = next(iter(loader))
    except StopIteration:
        print("[VALIDATE] DataLoader produced no batches.")
        return False

    ok = True
    actual_shape = tuple(x_batch.shape)
    expected_shape = (x_batch.shape[0], expected_window_size, expected_channels)

    if actual_shape != expected_shape:
        print(
            f"[VALIDATE] Unexpected X shape: {actual_shape} "
            f"(expected {expected_shape})"
        )
        ok = False
    else:
        print(f"[VALIDATE] X batch shape OK: {actual_shape}")

    if y_batch.shape != (x_batch.shape[0],):
        print(f"[VALIDATE] Unexpected y shape: {tuple(y_batch.shape)}")
        ok = False
    else:
        print(f"[VALIDATE] y batch shape OK: {tuple(y_batch.shape)}")

    return ok


def print_split_summary(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """
    Print per-split class distribution and sample counts.

    Scans all batches so this call takes time proportional to dataset size.
    For large datasets, use the stats dict from build_dataloaders instead.
    """
    print("\n=== Split Summary (window counts per class) ===")
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        dist = compute_class_distribution(loader)
        total = sum(dist.values())
        print(f"\n  {name} ({total} windows total):")
        for label, count in dist.items():
            pct = 100.0 * count / total if total > 0 else 0.0
            print(f"    {label}: {count:>6d}  ({pct:.1f}%)")


def save_preprocessing_config(
    output_path: Path = PROJECT_ROOT / "artifacts" / "results" / "preprocessing_config.json",
) -> Path:
    """
    Save the active preprocessing configuration as a JSON artifact.

    This file documents all preprocessing decisions for reproducibility,
    model evaluation, quantization, and FPGA input formatting.

    Parameters
    ----------
    output_path : Path
        Destination JSON file path.

    Returns
    -------
    Path
        Path of the saved file.
    """
    active_sample_rate = RESAMPLE_TARGET_SR if RESAMPLE_ENABLED else MAFAULDA_SAMPLE_RATE
    window_duration_seconds = WINDOW_SIZE / active_sample_rate

    config = {
        "dataset": {
            "name": "MAFAULDA",
            "original_sample_rate_hz": MAFAULDA_SAMPLE_RATE,
            "channels": EXPECTED_COLUMNS,
            "num_channels": EXPECTED_NUM_COLUMNS,
            "label_map": LABEL_MAP,
            "num_classes": NUM_CLASSES,
        },
        "split": {
            "method": "file_level_stratified",
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": RANDOM_SEED,
        },
        "quality_filtering": {
            "missing_ratio_threshold": MISSING_RATIO_THRESHOLD,
            "missing_fill_method": MISSING_FILL_METHOD,
        },
        "signal_processing": {
            "lowpass_filter_enabled": LOWPASS_FILTER_ENABLED,
            "lowpass_cutoff_hz": LOWPASS_CUTOFF_HZ if LOWPASS_FILTER_ENABLED else None,
            "lowpass_filter_order": LOWPASS_FILTER_ORDER if LOWPASS_FILTER_ENABLED else None,
            "wiener_filter_enabled": WIENER_FILTER_ENABLED,
            "wiener_kernel_size": WIENER_KERNEL_SIZE if WIENER_FILTER_ENABLED else None,
            "resample_enabled": RESAMPLE_ENABLED,
            "resample_target_sr_hz": RESAMPLE_TARGET_SR if RESAMPLE_ENABLED else None,
            "low_energy_filter_enabled": LOW_ENERGY_FILTER_ENABLED,
            "low_energy_channel_idx": LOW_ENERGY_CHANNEL_IDX if LOW_ENERGY_FILTER_ENABLED else None,
            "low_energy_threshold_rms": LOW_ENERGY_THRESHOLD_RMS if LOW_ENERGY_FILTER_ENABLED else None,
            "low_energy_segment_size": LOW_ENERGY_SEGMENT_SIZE if LOW_ENERGY_FILTER_ENABLED else None,
        },
        "normalization": {
            "method": NORMALIZATION_METHOD,
            "scope": "per_file_per_channel",
            "epsilon": EPSILON,
            "note": "Statistics are computed per file; not globally fitted on train split.",
        },
        "windowing": {
            "window_size_samples": WINDOW_SIZE,
            "window_stride_samples": WINDOW_STRIDE,
            "active_sample_rate_hz": active_sample_rate,
            "window_duration_seconds": round(window_duration_seconds, 6),
            "overlap_samples": max(0, WINDOW_SIZE - WINDOW_STRIDE),
            "include_tail_window": INCLUDE_TAIL_WINDOW,
            "tail_window_mode": TAIL_WINDOW_MODE if INCLUDE_TAIL_WINDOW else None,
        },
        "augmentation": {
            "smote_enabled": USE_SMOTE_AUGMENTATION,
            "noise_injection_enabled": USE_NOISE_INJECTION,
            "noise_injection_std": NOISE_INJECTION_STD,
            "scope": "train_split_only",
        },
        "dataloader": {
            "batch_size": BATCH_SIZE,
            "num_workers": DATALOADER_NUM_WORKERS,
        },
        "tensor_shape": {
            "description": "[batch_size, window_size, num_channels]",
            "window_size": WINDOW_SIZE,
            "num_channels": EXPECTED_NUM_COLUMNS,
            "dtype": "float32",
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[INFO] Preprocessing config saved to {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


if __name__ == "__main__":
    save_preprocessing_config()
