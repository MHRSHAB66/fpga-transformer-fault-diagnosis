"""
Utilities for building PyTorch DataLoaders from the MAFAULDA dataset.

This module is responsible for:
1. Building a file index from the dataset directory
2. Filtering valid labels
3. Splitting files into train/validation/test sets at file level
4. Reusing disk-cached base-preprocessed files for window creation
5. Materializing split windows from the preprocessed cache
6. Keeping validation and test datasets clean and unaugmented
7. Creating Dataset and DataLoader objects for PyTorch

Important
---------
- Splitting is performed at file level before window creation.
- This reduces leakage between train, validation, and test because
  multiple windows extracted from the same file are highly related.
- Base preprocessing is cached per file on disk and reused across splits.
- Augmentation cache remains train-only.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time

from src.config import (
    DATA_RAW_DIR,
    LABEL_MAP,
    NUM_CLASSES,
    RANDOM_SEED,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    INCLUDE_TAIL_WINDOW,
    TAIL_WINDOW_MODE,
    EXPECTED_NUM_COLUMNS,
    NORMALIZATION_METHOD,
    NORMALIZATION_SCOPE,
    EPSILON,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    DATALOADER_NUM_WORKERS,
)
from src.data.load_data import build_file_index
from src.data.prepare_dataset import (
    create_windows_from_dataframe,
    count_normal_windows,
    normalize_array,
    normalize_windows_array,
)
from src.data.preprocessed_cache import (
    load_or_create_preprocessed_array,
    CACHE_DIR as PREPROCESSED_CACHE_DIR,
)


def deterministic_small_split(index_df, train_size):
    """
    Split very small debug subsets without raising sklearn split errors.

    This is only a fallback path. Normal dataset sizes still use
    train_test_split with the configured random seed.
    """
    shuffled = index_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if len(shuffled) < 2:
        return shuffled.copy(), shuffled.iloc[0:0].copy()

    first_count = int(round(len(shuffled) * train_size))
    first_count = min(max(first_count, 1), len(shuffled) - 1)

    return (
        shuffled.iloc[:first_count].reset_index(drop=True),
        shuffled.iloc[first_count:].reset_index(drop=True),
    )


def safe_train_test_split(index_df, train_size, stratify_labels):
    """
    Wrapper around sklearn's train_test_split with a safe fallback.

    Parameters
    ----------
    index_df : pandas.DataFrame
        File index DataFrame to split.
    train_size : float
        Fraction of samples to place in the first split.

        Note
        ----
        The parameter name is kept as `train_size` to match sklearn's
        `train_test_split` API. In the second split step of this project,
        it effectively represents the size of the validation split relative
        to the temporary subset.
    stratify_labels : pandas.Series or None
        Labels used for stratified splitting when possible.

    Returns
    -------
    tuple
        Two DataFrames produced by train_test_split.

    Notes
    -----
    - `shuffle=True` is required for meaningful random splitting.
    - If stratified splitting fails, the function falls back to a
      non-stratified split and prints a warning.
    """
    try:
        return train_test_split(
            index_df,
            train_size=train_size,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError:
        if stratify_labels is not None:
            print("[WARN] Stratified split failed. Falling back to non-stratified split.")
        try:
            return train_test_split(
                index_df,
                train_size=train_size,
                random_state=RANDOM_SEED,
                shuffle=True,
                stratify=None,
            )
        except ValueError:
            print("[WARN] Split too small for sklearn. Using deterministic debug split.")
            return deterministic_small_split(index_df, train_size)


def split_file_index(index_df):
    """
    Split the file index into train, validation, and test sets.

    Parameters
    ----------
    index_df : pandas.DataFrame
        File index DataFrame containing at least:
        - filepath
        - label

    Returns
    -------
    tuple
        (train_df, val_df, test_df)

    Notes
    -----
    Splitting is done in two stages:
    1. train vs. temporary remainder
    2. validation vs. test from the remainder

    Stratification is used when more than one class is present.
    """
    if index_df.empty:
        return index_df.copy(), index_df.copy(), index_df.copy()

    labels = index_df["label"]
    stratify_labels = labels if labels.nunique() > 1 else None

    train_df, temp_df = safe_train_test_split(index_df, TRAIN_RATIO, stratify_labels)

    temp_stratify = temp_df["label"] if temp_df["label"].nunique() > 1 else None
    val_share_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = safe_train_test_split(temp_df, val_share_of_temp, temp_stratify)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


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


def build_X_y_from_file_index(
    index_df,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    include_tail: bool = INCLUDE_TAIL_WINDOW,
    tail_mode: str = TAIL_WINDOW_MODE,
    normalization_scope: str = NORMALIZATION_SCOPE,
    normalization_method: str = NORMALIZATION_METHOD,
    global_normalization_stats: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, dict]:
    """
    Materialize windowed training arrays from a file index DataFrame.

    Parameters
    ----------
    index_df : pandas.DataFrame
        File index DataFrame containing:
        - filepath
        - label
    window_size : int, default=WINDOW_SIZE
        Number of rows per time window.
    stride : int, default=WINDOW_STRIDE
        Step size between consecutive windows.
    include_tail : bool, default=INCLUDE_TAIL_WINDOW
        Whether to add a backshift tail window per file when trailing samples
        would otherwise be discarded.
    tail_mode : str, default=TAIL_WINDOW_MODE
        Tail strategy passed through to ``create_windows_from_dataframe``.
    normalization_scope : str, default=NORMALIZATION_SCOPE
        "per_file", "per_window", or "global_train".
    normalization_method : str, default=NORMALIZATION_METHOD
        Current supported final method is "zscore".
    global_normalization_stats : dict or None
        Required when normalization_scope is "global_train".

    Returns
    -------
    tuple
        (X, y, kept_files, dropped_files, window_counts)

        - X : np.ndarray
            Windowed input data with dtype float32 and shape
            [num_windows, window_size, num_features]
        - y : np.ndarray
            Integer labels with dtype int64
        - kept_files : int
            Number of files kept after preprocessing
        - dropped_files : int
            Number of files dropped during preprocessing
        - window_counts : dict
            {"normal": int, "tail": int, "total": int} window counts
            aggregated over all kept files in this split.

    Notes
    -----
    - Base preprocessing is loaded from the per-file preprocessing cache
      when available, or created once and then cached on disk.
    - Each file is assigned one label, and all windows extracted from that file
      receive the same label.
    """
    X = []
    y = []

    kept_files = 0
    dropped_files = 0
    normal_windows = 0
    tail_windows = 0

    for _, row in index_df.iterrows():
        file_path = row["filepath"]
        label_name = row["label"]

        # Skip unexpected labels for robustness.
        if label_name not in LABEL_MAP:
            continue

        values = load_or_create_preprocessed_array(file_path)

        if values is None:
            dropped_files += 1
            continue

        kept_files += 1

        if normalization_scope == "per_file":
            values = normalize_array(values, method=normalization_method)

        windows = create_windows_from_dataframe(
            values,
            window_size=window_size,
            stride=stride,
            include_tail=include_tail,
            tail_mode=tail_mode,
        )

        # The backshift policy appends at most one extra window per file, so the
        # tail count for this file is whatever exceeds the standard window count.
        file_normal = count_normal_windows(len(values), window_size, stride)
        normal_windows += file_normal
        tail_windows += max(0, len(windows) - file_normal)

        label_id = LABEL_MAP[label_name]

        # Preserve alignment between inputs and labels:
        # every appended window gets its matching file-level label.
        if windows:
            windows = normalize_windows_array(
                np.asarray(windows, dtype=np.float32),
                method=normalization_method,
                scope=normalization_scope,
                global_stats=global_normalization_stats,
            )

        for window in windows:
            X.append(window)
            y.append(label_id)

    if X:
        X = np.array(X, dtype=np.float32)
    else:
        X = np.empty((0, window_size, EXPECTED_NUM_COLUMNS), dtype=np.float32)

    y = np.array(y, dtype=np.int64)

    window_counts = {
        "normal": int(normal_windows),
        "tail": int(tail_windows),
        "total": int(normal_windows + tail_windows),
    }

    return X, y, kept_files, dropped_files, window_counts


def build_dataloaders(
    data_dir=DATA_RAW_DIR,
    max_files: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    normalization_scope: str = NORMALIZATION_SCOPE,
    normalization_method: str = NORMALIZATION_METHOD,
    num_workers: int = DATALOADER_NUM_WORKERS,
    include_tail: bool = INCLUDE_TAIL_WINDOW,
    tail_mode: str = TAIL_WINDOW_MODE,
):
    """
    Build train, validation, and test DataLoaders.

    Parameters
    ----------
    data_dir : Path-like, default=DATA_RAW_DIR
        Root directory of the dataset.
    max_files : int or None, default=None
        Optional limit on the number of files used. Useful for debugging.
    batch_size : int, default=BATCH_SIZE
        DataLoader batch size.
    window_size : int, default=WINDOW_SIZE
        Number of rows per time window.
    stride : int, default=WINDOW_STRIDE
        Step size between consecutive windows.
    normalization_scope : str, default=NORMALIZATION_SCOPE
        "per_file", "per_window", or "global_train".
    normalization_method : str, default=NORMALIZATION_METHOD
        Normalization method. The final pipeline uses "zscore".
    num_workers : int, default=DATALOADER_NUM_WORKERS
        Number of worker processes used by PyTorch DataLoader.
    include_tail : bool, default=INCLUDE_TAIL_WINDOW
        Whether to add a backshift tail window per file so trailing samples that
        would otherwise be discarded are still used. Applied identically to all
        splits; overlap stays within a file because splitting is file-level.
    tail_mode : str, default=TAIL_WINDOW_MODE
        Tail strategy. Only "backshift" is supported.

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, stats)

        - train_loader : DataLoader
        - val_loader : DataLoader
        - test_loader : DataLoader
        - stats : dict
            Summary information about file counts, sample counts,
            augmentation usage, preprocessing cache usage, and
            DataLoader settings.

    Notes
    -----
    - File-level split happens before window creation.
    - Validation and test remain clean and unaugmented.
    - Base preprocessing is reused through the per-file disk cache.
    - All splits are materialized from the per-file preprocessing cache.
    """
    if normalization_scope not in {"per_file", "per_window", "global_train"}:
        raise ValueError(f"Unsupported normalization_scope: {normalization_scope!r}")
    if normalization_scope == "per_window" and normalization_method != "zscore":
        raise ValueError("per_window normalization currently supports zscore only.")

    index_df = build_file_index(data_dir)
    if index_df.empty:
        raise ValueError(f"No CSV files were found under dataset directory: {data_dir}")

    # Keep only labels that are part of the expected training label map.
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    if index_df.empty:
        raise ValueError(
            "No files remained after filtering labels with LABEL_MAP. "
            "Check dataset structure and label parsing."
        )

    if max_files is not None:
        # Shuffle before truncating so debug subsets are not biased by sorted paths.
        index_df = index_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        index_df = index_df.head(max_files)

    train_index_df, val_index_df, test_index_df = split_file_index(index_df)

    if train_index_df.empty:
        raise ValueError("Train split is empty after file-level splitting.")
    if val_index_df.empty:
        raise ValueError("Validation split is empty after file-level splitting.")
    if test_index_df.empty:
        raise ValueError("Test split is empty after file-level splitting.")

    train_augmentation_cache_used = False
    train_augmentation_cache_path = None
    global_normalization_stats = None
    normalization_fit_seconds = 0.0
    split_materialization_seconds = {}

    if normalization_scope == "global_train":
        fit_start = time.perf_counter()
        global_normalization_stats = fit_global_normalization_stats(
            train_index_df,
            method=normalization_method,
        )
        normalization_fit_seconds = time.perf_counter() - fit_start

    train_materialize_start = time.perf_counter()
    X_train, y_train, train_kept, train_dropped, train_window_counts = (
        build_X_y_from_file_index(
            train_index_df,
            window_size=window_size,
            stride=stride,
            include_tail=include_tail,
            tail_mode=tail_mode,
            normalization_scope=normalization_scope,
            normalization_method=normalization_method,
            global_normalization_stats=global_normalization_stats,
        )
    )
    split_materialization_seconds["train"] = time.perf_counter() - train_materialize_start
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset contains zero samples after preprocessing/windowing."
        )

    augmentation_info = {
        "smote_applied": False,
        "noise_injection_applied": False,
        "original_samples": len(X_train),
        "augmented_samples": len(X_train),
    }

    val_materialize_start = time.perf_counter()
    X_val, y_val, val_kept, val_dropped, val_window_counts = build_X_y_from_file_index(
        val_index_df,
        window_size=window_size,
        stride=stride,
        include_tail=include_tail,
        tail_mode=tail_mode,
        normalization_scope=normalization_scope,
        normalization_method=normalization_method,
        global_normalization_stats=global_normalization_stats,
    )
    split_materialization_seconds["val"] = time.perf_counter() - val_materialize_start
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    test_materialize_start = time.perf_counter()
    X_test, y_test, test_kept, test_dropped, test_window_counts = build_X_y_from_file_index(
        test_index_df,
        window_size=window_size,
        stride=stride,
        include_tail=include_tail,
        tail_mode=tail_mode,
        normalization_scope=normalization_scope,
        normalization_method=normalization_method,
        global_normalization_stats=global_normalization_stats,
    )
    split_materialization_seconds["test"] = time.perf_counter() - test_materialize_start
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset contains zero windows after preprocessing/windowing."
        )
    if len(test_dataset) == 0:
        raise ValueError(
            "Test dataset contains zero windows after preprocessing/windowing."
        )

    # Shuffle training batches for better learning.
    # Validation and test remain ordered for stable evaluation.
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    preprocessing_cache_enabled = True
    preprocessing_cache_path = str(PREPROCESSED_CACHE_DIR)
    train_augmented = False

    # Per-class window counts for distribution reporting.
    label_names_by_id = {v: k for k, v in LABEL_MAP.items()}

    def _class_window_counts(y: np.ndarray) -> dict:
        if len(y) == 0:
            return {label_names_by_id.get(i, f"class_{i}"): 0 for i in range(NUM_CLASSES)}
        raw = np.bincount(y, minlength=NUM_CLASSES)
        return {label_names_by_id.get(i, f"class_{i}"): int(raw[i]) for i in range(NUM_CLASSES)}

    train_class_window_counts = _class_window_counts(y_train)
    val_class_window_counts = _class_window_counts(y_val)
    test_class_window_counts = _class_window_counts(y_test)

    def _estimated_split_bytes(num_windows: int) -> int:
        input_bytes = int(num_windows) * int(window_size) * int(EXPECTED_NUM_COLUMNS) * 4
        label_bytes = int(num_windows) * 8
        return input_bytes + label_bytes

    estimated_ram_bytes = {
        "train": _estimated_split_bytes(len(train_dataset)),
        "val": _estimated_split_bytes(len(val_dataset)),
        "test": _estimated_split_bytes(len(test_dataset)),
    }
    estimated_ram_bytes["total"] = sum(estimated_ram_bytes.values())
    estimated_ram_gib = {
        key: round(value / (1024 ** 3), 3)
        for key, value in estimated_ram_bytes.items()
    }

    stats = {
        "data_dir": str(data_dir),
        "max_files": None if max_files is None else int(max_files),
        "total_files_after_filter": len(index_df),
        "train_files_total": int(len(train_index_df)),
        "val_files_total": int(len(val_index_df)),
        "test_files_total": int(len(test_index_df)),
        "train_files_kept": train_kept,
        "train_files_dropped": train_dropped,
        "val_files_kept": val_kept,
        "val_files_dropped": val_dropped,
        "test_files_kept": test_kept,
        "test_files_dropped": test_dropped,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "window_shape": [int(window_size), EXPECTED_NUM_COLUMNS],
        "window_size": int(window_size),
        "window_stride": int(stride),
        "batch_size": int(batch_size),
        "normalization": {
            "scope": str(normalization_scope),
            "method": str(normalization_method),
            "epsilon": float(EPSILON),
            "global_stats": global_normalization_stats,
            "global_stats_fitted_on": (
                "train" if normalization_scope == "global_train" else None
            ),
        },
        "tail_window_info": {
            "enabled": bool(include_tail),
            "mode": tail_mode if include_tail else None,
            "train": train_window_counts,
            "val": val_window_counts,
            "test": test_window_counts,
            "total_normal": (
                train_window_counts["normal"]
                + val_window_counts["normal"]
                + test_window_counts["normal"]
            ),
            "total_tail": (
                train_window_counts["tail"]
                + val_window_counts["tail"]
                + test_window_counts["tail"]
            ),
        },
        "train_class_window_counts": train_class_window_counts,
        "val_class_window_counts": val_class_window_counts,
        "test_class_window_counts": test_class_window_counts,
        "train_augmented": train_augmented,
        "augmentation": augmentation_info,
        "train_augmentation_cache_used": train_augmentation_cache_used,
        "train_augmentation_cache_path": train_augmentation_cache_path,
        "preprocessing_cache_enabled": preprocessing_cache_enabled,
        "preprocessing_cache_path": preprocessing_cache_path,
        "preprocessing_timing_seconds": {
            "global_normalization_fit": float(normalization_fit_seconds),
            "split_materialization": {
                key: float(value)
                for key, value in split_materialization_seconds.items()
            },
            "total_split_materialization": float(sum(split_materialization_seconds.values())),
        },
        "estimated_materialized_ram_bytes": estimated_ram_bytes,
        "estimated_materialized_ram_gib": estimated_ram_gib,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
    }

    return train_loader, val_loader, test_loader, stats
