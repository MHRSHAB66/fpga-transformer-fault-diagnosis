"""
Archived normalization variants — NOT used by the active pipeline.

The deployed model uses **per-window z-score** only (see DECISION_LOG D005). The
`per_file` and `global_train` normalization scopes were removed from the active
data pipeline (`make_dataloaders.py`, `prepare_dataset.py`) to keep a single,
unambiguous normalization path. Their logic is preserved here so the planned
normalization-comparison study (EXP-5 in docs/future_experiments.md) can be run
without rewriting it.

Nothing in `src/` imports this module during normal training/evaluation. To run
EXP-5, wire these back into `build_X_y_from_file_index` / `build_dataloaders`
(and relax the `NORMALIZATION_SCOPE` validation in config.py).

Contents
--------
- fit_global_normalization_stats : fit per-channel mean/std (or min/max) on the
  TRAIN split only (leakage-free), for the `global_train` scope.
- normalize_array                : per-file (or fixed-stats) per-channel scaling,
  applied to a whole [rows, channels] file array before windowing.
- normalize_windows_array_all_scopes : the original 3-scope window normalizer
  (per_file / global_train / per_window). The active pipeline keeps only the
  per_window path in prepare_dataset.normalize_windows_array.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import NORMALIZATION_METHOD, NORMALIZATION_SCOPE, EPSILON
from src.data.preprocessed_cache import load_or_create_preprocessed_array


def fit_global_normalization_stats(
    index_df,
    method: str = NORMALIZATION_METHOD,
    epsilon: float = EPSILON,
) -> dict:
    """
    Fit per-channel normalization constants from training files only.

    Validation/test files are never inspected here, which keeps
    ``global_train`` leakage-free.
    """
    total_rows = 0
    kept_files = 0
    dropped_files = 0
    sum_values = None
    sum_squares = None
    min_values = None
    max_values = None

    for _, row in index_df.iterrows():
        values = load_or_create_preprocessed_array(row["filepath"])
        if values is None:
            dropped_files += 1
            continue
        if values.size == 0:
            continue

        values64 = values.astype(np.float64, copy=False)
        kept_files += 1
        total_rows += int(values64.shape[0])

        file_sum = values64.sum(axis=0)
        file_sum_squares = np.square(values64).sum(axis=0)
        file_min = values64.min(axis=0)
        file_max = values64.max(axis=0)

        if sum_values is None:
            sum_values = file_sum
            sum_squares = file_sum_squares
            min_values = file_min
            max_values = file_max
        else:
            sum_values += file_sum
            sum_squares += file_sum_squares
            min_values = np.minimum(min_values, file_min)
            max_values = np.maximum(max_values, file_max)

    if total_rows == 0 or sum_values is None:
        raise ValueError("Could not fit global normalization stats: no training rows.")

    if method == "zscore":
        mean = sum_values / total_rows
        variance = np.maximum((sum_squares / total_rows) - np.square(mean), 0.0)
        std = np.sqrt(variance)
        safe_std = np.where(np.abs(std) < epsilon, 1.0, std)
        return {
            "scope": "global_train",
            "method": method,
            "fitted_on": "train",
            "files_used": int(kept_files),
            "files_dropped": int(dropped_files),
            "rows_used": int(total_rows),
            "epsilon": float(epsilon),
            "mean": mean.astype(float).tolist(),
            "std": safe_std.astype(float).tolist(),
        }

    if method == "minmax":
        denom = max_values - min_values
        safe_denom = np.where(np.abs(denom) < epsilon, 1.0, denom)
        return {
            "scope": "global_train",
            "method": method,
            "fitted_on": "train",
            "files_used": int(kept_files),
            "files_dropped": int(dropped_files),
            "rows_used": int(total_rows),
            "epsilon": float(epsilon),
            "min": min_values.astype(float).tolist(),
            "max": max_values.astype(float).tolist(),
            "range": safe_denom.astype(float).tolist(),
        }

    raise ValueError(f"Unsupported normalization method: {method}")


def normalize_array(
    values: np.ndarray,
    method: str = NORMALIZATION_METHOD,
    stats: Optional[dict] = None,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Normalize a 2D time-series array per channel (per-file or fixed-stats).

    ``stats`` is optional for per-file fitting and required when applying
    constants fitted on the training split.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.float32, copy=False)

    if method == "zscore":
        if stats is None:
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else values.std(axis=0)
        else:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        safe_std = np.where(np.abs(std) < epsilon, 1.0, std)
        return ((values - mean) / safe_std).astype(np.float32)

    if method == "minmax":
        if stats is None:
            min_val = values.min(axis=0)
            max_val = values.max(axis=0)
        else:
            min_val = np.asarray(stats["min"], dtype=np.float32)
            max_val = np.asarray(stats["max"], dtype=np.float32)
        denom = max_val - min_val
        safe_denom = np.where(np.abs(denom) < epsilon, 1.0, denom)
        return ((values - min_val) / safe_denom).astype(np.float32)

    raise ValueError(f"Unsupported normalization method: {method}")


def normalize_windows_array_all_scopes(
    windows: np.ndarray,
    method: str = NORMALIZATION_METHOD,
    scope: str = NORMALIZATION_SCOPE,
    global_stats: Optional[dict] = None,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Original 3-scope window normalizer (per_file / global_train / per_window).

    The active pipeline (prepare_dataset.normalize_windows_array) keeps only the
    per_window path; this full version is retained for EXP-5.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.size == 0:
        return windows.astype(np.float32, copy=False)

    if scope == "per_file":
        # File was already scaled by normalize_array before windowing.
        return windows.astype(np.float32, copy=False)

    if scope == "global_train":
        if global_stats is None:
            raise ValueError("global_train normalization requires global_stats.")
        shape = (1, 1, windows.shape[-1])
        if method == "zscore":
            mean = np.asarray(global_stats["mean"], dtype=np.float32).reshape(shape)
            std = np.asarray(global_stats["std"], dtype=np.float32).reshape(shape)
            safe_std = np.where(np.abs(std) < epsilon, 1.0, std)
            return ((windows - mean) / safe_std).astype(np.float32)
        if method == "minmax":
            min_val = np.asarray(global_stats["min"], dtype=np.float32).reshape(shape)
            max_val = np.asarray(global_stats["max"], dtype=np.float32).reshape(shape)
            denom = max_val - min_val
            safe_denom = np.where(np.abs(denom) < epsilon, 1.0, denom)
            return ((windows - min_val) / safe_denom).astype(np.float32)
        raise ValueError(f"Unsupported normalization method: {method}")

    if scope == "per_window":
        if method != "zscore":
            raise ValueError("per_window normalization currently supports zscore only.")
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        safe_std = np.where(np.abs(std) < epsilon, 1.0, std)
        return ((windows - mean) / safe_std).astype(np.float32)

    raise ValueError(f"Unsupported normalization scope: {scope}")
