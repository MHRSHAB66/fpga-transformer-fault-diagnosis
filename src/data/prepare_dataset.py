"""
Dataset preprocessing utilities for the MAFAULDA pipeline.

This module provides reusable helper functions for:
1. Converting raw CSV DataFrames to numeric values
2. Dropping files with too many missing values
3. Filling missing values when needed
4. Creating fixed-length windows from a preprocessed DataFrame or ndarray
5. Per-window z-score normalization of a stack of windows

Important
---------
The main training pipeline should use `src.data.make_dataloaders.py`, which:
- performs file-level train/validation/test splitting
- prevents data leakage between splits
- builds the final PyTorch DataLoaders

This module is intentionally limited to reusable preprocessing and windowing
helpers.
"""

from numbers import Real
from typing import List, Optional, Union

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
    df: Union[pd.DataFrame, np.ndarray],
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> List[np.ndarray]:
    """
    Split a preprocessed DataFrame or ndarray into fixed-length windows.

    Trailing samples shorter than window_size are discarded.

    Input must have shape (samples, channels). window_size and stride must
    be positive integers (booleans are not accepted).

    Raises
    ------
    TypeError
        If window_size or stride is not an integer.
    ValueError
        If window_size or stride is not positive, or input is not two-dimensional.
    """
    for name, value in (("window_size", window_size), ("stride", stride)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a positive integer; got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value!r}")

    if hasattr(df, "to_numpy"):
        values = df.to_numpy(copy=True)
    else:
        values = np.asarray(df, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError(
            f"Input must be two-dimensional (samples, channels); got shape {values.shape}"
        )

    windows: List[np.ndarray] = []
    num_rows = len(values)
    if num_rows < window_size:
        return windows

    for start in range(0, num_rows - window_size + 1, stride):
        windows.append(values[start:start + window_size])

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
        Maximum allowed ratio of missing cells, finite and within [0, 1].

    Returns
    -------
    bool
        True if the file should be dropped, otherwise False.
    """
    if (
        isinstance(missing_ratio_threshold, (bool, np.bool_))
        or not isinstance(missing_ratio_threshold, Real)
        or not np.isfinite(missing_ratio_threshold)
        or not 0 <= missing_ratio_threshold <= 1
    ):
        raise ValueError("missing_ratio_threshold must be a finite number in [0, 1]")

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
        DataFrame after applying the selected fill method. A column containing
        only missing values may still contain NaNs after the operation.

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


def normalize_windows_array(
    windows: np.ndarray,
    epsilon: float = EPSILON,
    method: str = NORMALIZATION_METHOD,
) -> np.ndarray:
    """
    Apply per-window z-score normalization to a stack of windows.

    Input shape is (windows, samples, channels). Each channel in each window
    is normalized independently over its time samples; no dataset-wide
    statistics are needed. An empty batch is allowed, but samples and channels
    must be nonempty. Only method='zscore' is supported.

    Raises ValueError for invalid shape, non-finite values, non-positive or
    non-finite epsilon, unsupported methods, or float32 arithmetic overflow.
    """
    if method != "zscore":
        raise ValueError(
            "normalize_windows_array supports only method='zscore'; "
            f"got {method!r}"
        )

    if (
        isinstance(epsilon, (bool, np.bool_))
        or not isinstance(epsilon, Real)
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be a finite positive number")

    with np.errstate(over="ignore", invalid="ignore"):
        windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1] == 0 or windows.shape[2] == 0:
        raise ValueError(
            "windows must have shape (windows, samples, channels) with nonempty "
            f"samples and channels; got {windows.shape}"
        )
    if not np.isfinite(windows).all():
        raise ValueError("windows must contain only finite float32 values")
    if windows.size == 0:
        return windows.astype(np.float32, copy=False)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        safe_std = np.where((std == 0) | (np.abs(std) < epsilon), 1.0, std)
        result = ((windows - mean) / safe_std).astype(np.float32)
    if not (np.isfinite(mean).all() and np.isfinite(std).all()
            and np.isfinite(result).all()):
        raise ValueError("float32 overflow during window normalization")
    return result


def preprocess_dataframe(df: pd.DataFrame, normalize: bool = False) -> Optional[pd.DataFrame]:
    """
    Run the preprocessing pipeline on a single raw DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw DataFrame read from a CSV file.

    Returns
    -------
    pandas.DataFrame or None
        Preprocessed DataFrame if the file is valid. Returns None if the file
        exceeds the missing threshold, cannot be fully filled, or contains infinity.

    Pipeline
    --------
    1. Convert all values to numeric
    2. Drop the file if the missing-value ratio is too high
    3. Fill missing values only if NaN values exist
    4. Drop the file if missing values still remain after filling

    Normalization is intentionally NOT done here. The active pipeline applies
    per-window z-score normalization later, in make_dataloaders.py (via
    normalize_windows_array). The ``normalize`` argument is a compatibility
    breadcrumb: passing normalize=True raises, pointing to the removed
    file-level normalization path.
    """
    if normalize:
        raise NotImplementedError(
            "normalize_dataframe was removed (2026-09-01). "
            "Active pipeline uses per-window z-score in make_dataloaders.py. "
            "See archive/preprocess_data.py for the old file-level normalization path."
        )

    df = convert_dataframe_to_numeric(df)

    # Infinity is not missing data: reject it rather than invent replacements.
    if df.isin([np.inf, -np.inf]).any().any():
        return None

    if should_drop_file(df):
        return None

    # Avoid unnecessary fill operations when the DataFrame is already clean.
    if df.isna().any().any():
        df = fill_missing_values(df)

    # Median/mean cannot fill a column whose every value is missing. Reject
    # such files instead of silently feeding NaNs or invented zeros downstream.
    if df.isna().any().any() or df.isin([np.inf, -np.inf]).any().any():
        return None

    return df
