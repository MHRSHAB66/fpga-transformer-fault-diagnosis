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
    PROJECT_ROOT,
    LABEL_MAP,
    NUM_CLASSES,
    RANDOM_SEED,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    INCLUDE_TAIL_WINDOW,
    TAIL_WINDOW_MODE,
    EXPECTED_NUM_COLUMNS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    DATALOADER_NUM_WORKERS,
    USE_SMOTE_AUGMENTATION,
    USE_NOISE_INJECTION,
    NOISE_INJECTION_STD,
)
from src.data.augmentation import augment_training_data
from src.data.cached_augmentation import (
    build_train_augmentation_cache_key,
    get_train_augmentation_cache_path,
    load_cached_augmented_train_data,
    save_augmented_train_data_cache,
)
from src.data.load_data import build_file_index
from src.data.prepare_dataset import create_windows_from_dataframe, count_normal_windows
from src.data.preprocessed_cache import (
    load_or_create_preprocessed_array,
    CACHE_DIR as PREPROCESSED_CACHE_DIR,
)


def _save_bar_plot(
    values: dict,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    display_labels: Optional[dict] = None,
    figure_size=(6, 4),
    rotation: int = 0,
) -> None:
    """
    Save a simple bar chart for the provided mapping.

    This helper is used only from the direct-run path so normal library usage
    remains unchanged.
    """
    if not values:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(values.keys())
    numeric_values = list(values.values())
    tick_labels = [display_labels.get(label, label) for label in labels] if display_labels else labels

    fig, ax = plt.subplots(figsize=figure_size)
    bars = ax.bar(labels, numeric_values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=rotation, ha="right" if rotation else "center")

    for label, bar, value in zip(labels, bars, numeric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            (
                f"{value:.6f}"
                if label == "second_train_batch_seconds" and isinstance(value, float)
                else f"{value:.3f}" if isinstance(value, float) else str(value)
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_direct_run_plots(stats: dict, timings: dict) -> None:
    """
    Save lightweight summary plots for direct execution only.
    """
    output_dir = PROJECT_ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_bar_plot(
        {
            "train": stats["train_files_kept"],
            "val": stats["val_files_kept"],
            "test": stats["test_files_kept"],
        },
        output_dir / "dataloader_file_counts.png",
        "Kept Files Per Split",
        "Files",
    )

    _save_bar_plot(
        {
            "train": stats["train_samples"],
            "val": stats["val_samples"],
            "test": stats["test_samples"],
        },
        output_dir / "dataloader_sample_counts.png",
        "Samples Per Split",
        "Samples",
    )

    available_timings = {
        key: value
        for key, value in timings.items()
        if value is not None
    }
    if available_timings:
        _save_bar_plot(
            available_timings,
            output_dir / "dataloader_timing_seconds.png",
            "DataLoader Build And Batch Timing",
            "Seconds",
            display_labels={
                "build_dataloaders_seconds": "build_dataloaders",
                "first_train_batch_seconds": "first_train_batch",
                "second_train_batch_seconds": "second_train_batch",
            },
            figure_size=(7.5, 4),
            rotation=0,
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
    include_tail: bool = INCLUDE_TAIL_WINDOW,
    tail_mode: str = TAIL_WINDOW_MODE,
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
    use_smote: bool = USE_SMOTE_AUGMENTATION,
    use_noise_injection: bool = USE_NOISE_INJECTION,
    noise_std: float = NOISE_INJECTION_STD,
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
    use_smote : bool, default=USE_SMOTE_AUGMENTATION
        Whether to apply SMOTE-based augmentation to training data.
    use_noise_injection : bool, default=USE_NOISE_INJECTION
        Whether to apply noise injection to training data.
    noise_std : float, default=NOISE_INJECTION_STD
        Standard deviation used for noise injection.
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

    if use_smote or use_noise_injection:
        # SMOTE and noise injection operate on arrays, so training is materialized
        # before augmentation. Validation and test remain clean and unaugmented.
        cache_key = build_train_augmentation_cache_key(
            train_index_df,
            use_smote=use_smote,
            use_noise_injection=use_noise_injection,
            noise_std=noise_std,
            include_tail=include_tail,
            tail_mode=tail_mode,
        )
        train_augmentation_cache_path = str(get_train_augmentation_cache_path(cache_key))
        cached_train_data = load_cached_augmented_train_data(cache_key)

        if cached_train_data is not None:
            X_train, y_train, cache_metadata = cached_train_data
            augmentation_info = cache_metadata["augmentation_info"]
            train_kept = int(cache_metadata["train_kept"])
            train_dropped = int(cache_metadata["train_dropped"])
            # Pre-augmentation window counts are stored in the cache metadata.
            # Older caches predate this field; fall back to zeros if absent.
            train_window_counts = cache_metadata.get(
                "window_counts", {"normal": 0, "tail": 0, "total": 0}
            )
            train_augmentation_cache_used = True
        else:
            X_train, y_train, train_kept, train_dropped, train_window_counts = (
                build_X_y_from_file_index(
                    train_index_df, include_tail=include_tail, tail_mode=tail_mode
                )
            )
            X_train, y_train, augmentation_info = augment_training_data(
                X_train,
                y_train,
                use_smote=use_smote,
                use_noise_injection=use_noise_injection,
                noise_std=noise_std,
            )
            save_augmented_train_data_cache(
                cache_key,
                X_train,
                y_train,
                {
                    "augmentation_info": augmentation_info,
                    "train_kept": int(train_kept),
                    "train_dropped": int(train_dropped),
                    "window_counts": train_window_counts,
                },
            )

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )

        if len(train_dataset) == 0:
            raise ValueError(
                "Training dataset contains zero samples after materialization/augmentation."
            )
    else:
        X_train, y_train, train_kept, train_dropped, train_window_counts = (
            build_X_y_from_file_index(
                train_index_df, include_tail=include_tail, tail_mode=tail_mode
            )
        )
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
            "noise_std": float(noise_std),
            "original_samples": len(X_train),
            "augmented_samples": len(X_train),
        }

    X_val, y_val, val_kept, val_dropped, val_window_counts = build_X_y_from_file_index(
        val_index_df, include_tail=include_tail, tail_mode=tail_mode
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    X_test, y_test, test_kept, test_dropped, test_window_counts = build_X_y_from_file_index(
        test_index_df, include_tail=include_tail, tail_mode=tail_mode
    )
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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    preprocessing_cache_enabled = True
    preprocessing_cache_path = str(PREPROCESSED_CACHE_DIR)
    train_augmented = bool(use_smote or use_noise_injection)

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

    stats = {
        "data_dir": str(data_dir),
        "max_files": None if max_files is None else int(max_files),
        "total_files_after_filter": len(index_df),
        "train_files_kept": train_kept,
        "train_files_dropped": train_dropped,
        "val_files_kept": val_kept,
        "val_files_dropped": val_dropped,
        "test_files_kept": test_kept,
        "test_files_dropped": test_dropped,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "window_shape": [WINDOW_SIZE, EXPECTED_NUM_COLUMNS],
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
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
    }

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    t0 = time.perf_counter()

    train_loader, val_loader, test_loader, stats = build_dataloaders()

    t1 = time.perf_counter()

    train_iter = iter(train_loader)
    x_batch, y_batch = next(train_iter)

    t2 = time.perf_counter()

    try:
        _ = next(train_iter)
        t3 = time.perf_counter()
    except StopIteration:
        t3 = t2

    print("Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nTiming:")
    print(f"build_dataloaders_seconds: {t1 - t0:.3f}")
    print(f"first_train_batch_seconds: {t2 - t1:.3f}")
    print(f"second_train_batch_seconds: {t3 - t2:.10f}")
    print(f"total_seconds: {t3 - t0:.3f}")

    _save_direct_run_plots(
        stats,
        {
            "build_dataloaders_seconds": t1 - t0,
            "first_train_batch_seconds": t2 - t1,
            "second_train_batch_seconds": t3 - t2,
        },
    )

    print("\nNumber of batches:")
    print("train:", len(train_loader))
    print("val:", len(val_loader))
    print("test:", len(test_loader))

    print("\nBatch X shape:", x_batch.shape)
    print("Batch y shape:", y_batch.shape)
    print("Batch labels:", y_batch.tolist())
