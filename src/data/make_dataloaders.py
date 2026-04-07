"""
Utilities for building PyTorch DataLoaders from the MAFAULDA dataset.

This module is responsible for:
1. Building a file index from the dataset directory
2. Filtering valid labels
3. Splitting files into train/validation/test sets at file level
4. Converting each file into preprocessed time windows
5. Applying augmentation to training data only
6. Creating TensorDataset and DataLoader objects for PyTorch

Important
---------
Splitting is performed at file level before window creation.
This is necessary to reduce data leakage between train, validation, and test
sets, since multiple windows extracted from the same file are highly related.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DATA_RAW_DIR,
    LABEL_MAP,
    RANDOM_SEED,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    USE_SMOTE_AUGMENTATION,
    USE_NOISE_INJECTION,
    NOISE_INJECTION_STD,
)
from src.data.augmentation import augment_training_data
from src.data.load_data import build_file_index, load_one_csv
from src.data.prepare_dataset import preprocess_dataframe, create_windows_from_dataframe


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
        print("[WARN] Stratified split failed. Falling back to non-stratified split.")
        return train_test_split(
            index_df,
            train_size=train_size,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=None,
        )


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
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build X and y arrays from a file index DataFrame.

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
        (X, y, kept_files, dropped_files)

        - X : np.ndarray
            Windowed input data with dtype float32
        - y : np.ndarray
            Integer labels with dtype int64
        - kept_files : int
            Number of files kept after preprocessing
        - dropped_files : int
            Number of files dropped during preprocessing

    Notes
    -----
    Each file is assigned one label, and all windows extracted from that file
    receive the same label.
    """
    X = []
    y = []

    kept_files = 0
    dropped_files = 0

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

        kept_files += 1

        windows = create_windows_from_dataframe(
            df,
            window_size=window_size,
            stride=stride,
        )

        label_id = LABEL_MAP[label_name]

        # Preserve alignment between inputs and labels:
        # every appended window gets its matching file-level label.
        for window in windows:
            X.append(window)
            y.append(label_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, kept_files, dropped_files


def build_dataloaders(
    data_dir=DATA_RAW_DIR,
    max_files: Optional[int] = None,
    use_smote: bool = USE_SMOTE_AUGMENTATION,
    use_noise_injection: bool = USE_NOISE_INJECTION,
    noise_std: float = NOISE_INJECTION_STD,
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

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, stats)

        - train_loader : DataLoader
        - val_loader : DataLoader
        - test_loader : DataLoader
        - stats : dict
            Summary information about file counts, sample counts,
            and augmentation.
    """
    index_df = build_file_index(data_dir)

    # Keep only labels that are part of the expected training label map.
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()

    '''Inessntial SHUFFLE - already done in safe_train_test_split with shuffle=True.'''
    # Shuffle file order before splitting so path ordering does not bias the split.
    # index_df = index_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if max_files is not None:
        index_df = index_df.head(max_files)

    train_index_df, val_index_df, test_index_df = split_file_index(index_df)

    X_train, y_train, train_kept, train_dropped = build_X_y_from_file_index(train_index_df)
    X_val, y_val, val_kept, val_dropped = build_X_y_from_file_index(val_index_df)
    X_test, y_test, test_kept, test_dropped = build_X_y_from_file_index(test_index_df)

    # Apply augmentation only to training data.
    X_train, y_train, augmentation_info = augment_training_data(
        X_train,
        y_train,
        use_smote=use_smote,
        use_noise_injection=use_noise_injection,
        noise_std=noise_std,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    # Shuffle training batches for better learning.
    # Validation and test remain ordered for stable evaluation.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    stats = {
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
        "augmentation": augmentation_info,
    }

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    train_loader, val_loader, test_loader, stats = build_dataloaders()

    print("Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nNumber of batches:")
    print("train:", len(train_loader))
    print("val:", len(val_loader))
    print("test:", len(test_loader))

    x_batch, y_batch = next(iter(train_loader))

    print("\nBatch X shape:", x_batch.shape)
    print("Batch y shape:", y_batch.shape)
    print("Batch labels:", y_batch.tolist())