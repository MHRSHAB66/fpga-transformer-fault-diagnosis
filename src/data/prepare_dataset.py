"""
Dataset preprocessing utilities for the MAFAULDA pipeline.

This module provides reusable helper functions for:
1. Converting raw CSV DataFrames to numeric values
2. Dropping files with too many missing values
3. Filling missing values when needed
4. Normalizing each feature column
5. Creating fixed-length windows from a preprocessed DataFrame

Important
---------
The main training pipeline should use `src.data.make_dataloaders.py`, which:
- performs file-level train/validation/test splitting
- prevents data leakage between splits
- builds the final PyTorch DataLoaders

This module is intentionally limited to reusable preprocessing and windowing
helpers.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import (
    WINDOW_SIZE,
    WINDOW_STRIDE,
    MISSING_RATIO_THRESHOLD,
    MISSING_FILL_METHOD,
    NORMALIZATION_METHOD,
    EPSILON,
)


def create_windows_from_dataframe(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> List[np.ndarray]:
    """
    Split a preprocessed DataFrame into fixed-length windows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with shape [num_rows, num_features].
    window_size : int, default=WINDOW_SIZE
        Number of rows per window.
    stride : int, default=WINDOW_STRIDE
        Step size between consecutive windows.

    Returns
    -------
    list[np.ndarray]
        List of windows, where each window has shape [window_size, num_features].

    Notes
    -----
    - If the DataFrame has fewer rows than `window_size`, the returned list is empty.
    - With `stride == window_size`, the windows do not overlap.
    - With `stride < window_size`, the windows overlap.
    """
    values = df.to_numpy(copy=True)
    windows: List[np.ndarray] = []

    num_rows = len(values)
    if num_rows < window_size:
        return windows

    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        window = values[start:end]
        windows.append(window)

    return windows


def convert_dataframe_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all DataFrame columns to numeric values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame read from a raw CSV file.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame where each column has been converted
        to numeric values when possible.

    Notes
    -----
    Non-numeric values are coerced to NaN using `errors="coerce"`.
    This makes malformed values explicit so they can be handled later.
    """
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def should_drop_file(
    df: pd.DataFrame,
    missing_ratio_threshold: float = MISSING_RATIO_THRESHOLD,
) -> bool:
    """
    Decide whether a file should be dropped based on its missing-value ratio.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    missing_ratio_threshold : float, default=MISSING_RATIO_THRESHOLD
        Maximum allowed ratio of missing cells before dropping the file.

    Returns
    -------
    bool
        True if the file should be dropped, otherwise False.
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    if total_cells == 0:
        return True

    missing_ratio = missing_cells / total_cells
    return missing_ratio > missing_ratio_threshold


def fill_missing_values(
    df: pd.DataFrame,
    method: str = MISSING_FILL_METHOD,
) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame that may contain NaN values.
    method : str, default=MISSING_FILL_METHOD
        Missing-value handling strategy. Supported values:
        - "median"
        - "mean"
        - "ffill"
        - "bfill"

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values filled.

    Raises
    ------
    ValueError
        If the method is not supported.
    """
    df = df.copy()

    if method == "median":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    elif method == "mean":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    elif method == "ffill":
        df = df.ffill()

    elif method == "bfill":
        df = df.bfill()

    else:
        raise ValueError(f"Unsupported missing fill method: {method}")

    return df


def normalize_dataframe(
    df: pd.DataFrame,
    method: str = NORMALIZATION_METHOD,
) -> pd.DataFrame:
    """
    Normalize each column of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    method : str, default=NORMALIZATION_METHOD
        Normalization strategy. Supported values:
        - "zscore"
        - "minmax"

    Returns
    -------
    pandas.DataFrame
        Normalized DataFrame.

    Raises
    ------
    ValueError
        If the method is not supported.

    Notes
    -----
    EPSILON is added to denominators to avoid division by zero.
    """
    df = df.copy()

    if method == "zscore":
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + EPSILON)

    elif method == "minmax":
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + EPSILON)

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return df


def preprocess_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Run the preprocessing pipeline on a single raw DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw DataFrame read from a CSV file.

    Returns
    -------
    pandas.DataFrame or None
        Preprocessed DataFrame if the file is valid.
        Returns None if the file should be dropped.

    Pipeline
    --------
    1. Convert all values to numeric
    2. Drop the file if the missing-value ratio is too high
    3. Fill missing values only if NaN values exist
    4. Normalize each feature column
    """
    df = convert_dataframe_to_numeric(df)

    if should_drop_file(df):
        return None

    # Avoid unnecessary fill operations when the DataFrame is already clean.
    if df.isna().any().any():
        df = fill_missing_values(df)

    df = normalize_dataframe(df)

    return df