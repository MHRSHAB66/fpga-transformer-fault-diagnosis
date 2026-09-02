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
    NORMALIZATION_METHOD,
    NORMALIZATION_SCOPE,
    RANDOM_SEED,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    EXPECTED_NUM_COLUMNS,
    EPSILON,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    DATALOADER_NUM_WORKERS,
)
from src.data.load_data import build_file_index, load_one_csv
from src.data.prepare_dataset import (
    create_windows_from_dataframe,
    normalize_windows_array,
    preprocess_dataframe,
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


def build_X_y_from_file_index(
    index_df,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
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
            {"normal": int} window counts aggregated over all kept files.

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

    for _, row in index_df.iterrows():
        file_path = row["filepath"]
        label_name = row["label"]

        # Skip unexpected labels for robustness.
        if label_name not in LABEL_MAP:
            continue

        df = load_one_csv(file_path)
        df = preprocess_dataframe(df)
        if df is None:
            dropped_files += 1
            continue
        values = df.to_numpy(copy=True).astype(np.float32, copy=False)

        kept_files += 1

        windows = create_windows_from_dataframe(
            values,
            window_size=window_size,
            stride=stride,
        )

        label_id = LABEL_MAP[label_name]

        if windows:
            windows = normalize_windows_array(np.asarray(windows, dtype=np.float32))
            normal_windows += len(windows)

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
    }

    return X, y, kept_files, dropped_files, window_counts


def build_dataloaders(
    data_dir=DATA_RAW_DIR,
    max_files: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    num_workers: int = DATALOADER_NUM_WORKERS,
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
    num_workers : int, default=DATALOADER_NUM_WORKERS
        Number of worker processes used by PyTorch DataLoader.

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

    split_materialization_seconds = {}

    train_materialize_start = time.perf_counter()
    X_train, y_train, train_kept, train_dropped, train_window_counts = (
        build_X_y_from_file_index(
            train_index_df,
            window_size=window_size,
            stride=stride,
        )
    )
    split_materialization_seconds["train"] = time.perf_counter() - train_materialize_start
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    val_materialize_start = time.perf_counter()
    X_val, y_val, val_kept, val_dropped, val_window_counts = build_X_y_from_file_index(
        val_index_df,
        window_size=window_size,
        stride=stride,
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
    )
    split_materialization_seconds["test"] = time.perf_counter() - test_materialize_start
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset contains zero samples after preprocessing/windowing."
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
            "scope": NORMALIZATION_SCOPE,
            "method": NORMALIZATION_METHOD,
            "epsilon": float(EPSILON),
        },
        "window_counts": {
            "train": train_window_counts,
            "val": val_window_counts,
            "test": test_window_counts,
            "total_normal": (
                train_window_counts["normal"]
                + val_window_counts["normal"]
                + test_window_counts["normal"]
            ),
        },
        "train_class_window_counts": train_class_window_counts,
        "val_class_window_counts": val_class_window_counts,
        "test_class_window_counts": test_class_window_counts,
        "preprocessing_timing_seconds": {
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
