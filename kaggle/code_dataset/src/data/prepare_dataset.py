"""
Dataset preprocessing utilities for the MAFAULDA pipeline.

This module provides reusable helper functions for:
1. Converting raw CSV DataFrames to numeric values
2. Dropping files with too many missing values
3. Filling missing values when needed
4. Normalizing each feature column
5. Creating fixed-length windows from a preprocessed DataFrame or ndarray

Important
---------
The main training pipeline should use `src.data.make_dataloaders.py`, which:
- performs file-level train/validation/test splitting
- prevents data leakage between splits
- builds the final PyTorch DataLoaders

This module is intentionally limited to reusable preprocessing and windowing
helpers.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.config import (
    WINDOW_SIZE,
    WINDOW_STRIDE,
    INCLUDE_TAIL_WINDOW,
    TAIL_WINDOW_MODE,
    MISSING_RATIO_THRESHOLD,
    MISSING_FILL_METHOD,
    NORMALIZATION_METHOD,
    NORMALIZATION_SCOPE,
    EPSILON,
    MAFAULDA_SAMPLE_RATE,
    RESAMPLE_ENABLED,
    RESAMPLE_TARGET_SR,
    LOWPASS_FILTER_ENABLED,
    LOWPASS_CUTOFF_HZ,
    LOWPASS_FILTER_ORDER,
    WIENER_FILTER_ENABLED,
    WIENER_KERNEL_SIZE,
    LOW_ENERGY_FILTER_ENABLED,
    LOW_ENERGY_CHANNEL_IDX,
    LOW_ENERGY_THRESHOLD_RMS,
    LOW_ENERGY_SEGMENT_SIZE,
)


def count_normal_windows(num_rows: int, window_size: int, stride: int) -> int:
    """
    Number of full non-overlapping/striding windows produced before any tail.

    Parameters
    ----------
    num_rows : int
        Number of rows available in the file.
    window_size : int
        Window length in samples.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    int
        Count of standard windows (excluding any backshift tail window).
        0 when the file is shorter than one window.
    """
    if num_rows < window_size:
        return 0
    return (num_rows - window_size) // stride + 1


def create_windows_from_dataframe(
    df: Union[pd.DataFrame, np.ndarray],
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    include_tail: bool = INCLUDE_TAIL_WINDOW,
    tail_mode: str = TAIL_WINDOW_MODE,
) -> List[np.ndarray]:
    """
    Split a preprocessed DataFrame or ndarray into fixed-length windows.

    Parameters
    ----------
    df : pandas.DataFrame or np.ndarray
        Input tabular time-series data with shape [num_rows, num_features].
        The input may be either:
        - a preprocessed pandas DataFrame
        - a preprocessed NumPy array
    window_size : int, default=WINDOW_SIZE
        Number of rows per window.
    stride : int, default=WINDOW_STRIDE
        Step size between consecutive windows.
    include_tail : bool, default=INCLUDE_TAIL_WINDOW
        When True, add one extra full-size window aligned to the end of the file
        if the standard windowing left trailing samples shorter than a window.
    tail_mode : str, default=TAIL_WINDOW_MODE
        Tail strategy. Only "backshift" is supported: the extra window starts at
        ``num_rows - window_size`` so it ends exactly at the last row, using only
        real samples (no padding).

    Returns
    -------
    list[np.ndarray]
        List of windows, where each window has shape
        [window_size, num_features].

    Notes
    -----
    - If the input has fewer rows than `window_size`, the returned list is empty.
    - With `stride == window_size`, the standard windows do not overlap.
    - With `stride < window_size`, the standard windows overlap.
    - The function preserves the original row order of the input.
    - The backshift tail window is added only when the standard windows do not
      already reach the last row, i.e. when ``(num_rows - window_size)`` is
      strictly greater than the last standard window's start index. That single
      condition simultaneously guarantees a non-zero remainder exists and that
      the tail window is not a duplicate of the last standard window.
    - Supporting both DataFrame and ndarray inputs allows reuse with
      disk-cached preprocessed arrays without changing the windowing logic.
    """
    if hasattr(df, "to_numpy"):
        values = df.to_numpy(copy=True)
    else:
        values = np.asarray(df, dtype=np.float32)

    windows: List[np.ndarray] = []

    num_rows = len(values)
    if num_rows < window_size:
        return windows

    last_start = 0
    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])
        last_start = start

    if include_tail:
        if tail_mode != "backshift":
            raise ValueError(
                f"Unsupported tail_mode: {tail_mode!r}. Only 'backshift' is supported."
            )
        # tail_start > last_start means the last standard window does not reach
        # the end of the file (a non-zero remainder exists) AND the tail window
        # starts at a different index, so it cannot duplicate the last window.
        tail_start = num_rows - window_size
        if tail_start > last_start:
            windows.append(values[tail_start:num_rows])

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


def normalize_array(
    values: np.ndarray,
    method: str = NORMALIZATION_METHOD,
    stats: Optional[dict] = None,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Normalize a 2D time-series array per channel.

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


def normalize_windows_array(
    windows: np.ndarray,
    method: str = NORMALIZATION_METHOD,
    scope: str = NORMALIZATION_SCOPE,
    global_stats: Optional[dict] = None,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Normalize a stack of windows according to the selected scope.

    The final candidate is ``per_window`` z-score: mean/std are computed per
    channel inside each individual window, which is reproducible for FPGA
    inference from the 1024-sample input buffer.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.size == 0:
        return windows.astype(np.float32, copy=False)

    if scope == "per_file":
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


def apply_lowpass_filter(
    values: np.ndarray,
    cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    sample_rate: int = MAFAULDA_SAMPLE_RATE,
    order: int = LOWPASS_FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth lowpass filter to each column.

    Parameters
    ----------
    values : np.ndarray
        Shape [num_rows, num_features].
    cutoff_hz : float
        Filter cutoff frequency in Hz.
    sample_rate : int
        Signal sample rate in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered array, same shape as input.

    Notes
    -----
    filtfilt is non-causal (zero-phase). It cannot be replicated exactly in
    real-time hardware; a causal alternative would be needed for the FPGA path.
    """
    from scipy.signal import butter, filtfilt

    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1.0:
        return values

    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    result = values.copy()
    min_len = order * 3 + 1
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        if len(col) > min_len:
            result[:, col_idx] = filtfilt(b, a, col).astype(np.float32)
    return result.astype(np.float32)


def apply_wiener_filter(
    values: np.ndarray,
    kernel_size: int = WIENER_KERNEL_SIZE,
) -> np.ndarray:
    """
    Apply Wiener denoising to each column independently.

    Parameters
    ----------
    values : np.ndarray
        Shape [num_rows, num_features].
    kernel_size : int
        Wiener filter window size.

    Returns
    -------
    np.ndarray
        Denoised array, same shape as input.
    """
    from scipy.signal import wiener

    result = values.copy()
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        if len(col) >= kernel_size:
            result[:, col_idx] = np.asarray(wiener(col, mysize=kernel_size), dtype=np.float32)
    return result.astype(np.float32)


def apply_resample_array(
    values: np.ndarray,
    original_sr: int = MAFAULDA_SAMPLE_RATE,
    target_sr: int = RESAMPLE_TARGET_SR,
) -> np.ndarray:
    """
    Resample the time axis of a multichannel array using FFT-based resampling.

    Parameters
    ----------
    values : np.ndarray
        Shape [num_rows, num_features].
    original_sr : int
        Original sample rate in Hz.
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    np.ndarray
        Resampled array with shape [new_num_rows, num_features].

    Notes
    -----
    WINDOW_SIZE stays in samples. After resampling, each window covers
    (WINDOW_SIZE / target_sr) seconds instead of (WINDOW_SIZE / original_sr).
    The FPGA input path must use target_sr when resampling is enabled.
    """
    from scipy.signal import resample as scipy_resample

    if original_sr == target_sr or values.shape[0] == 0:
        return values
    target_len = max(1, int(round(values.shape[0] * target_sr / original_sr)))
    return np.asarray(scipy_resample(values, target_len, axis=0), dtype=np.float32)


def apply_low_energy_filter(
    values: np.ndarray,
    channel_idx: int = LOW_ENERGY_CHANNEL_IDX,
    threshold_rms: float = LOW_ENERGY_THRESHOLD_RMS,
    segment_size: int = LOW_ENERGY_SEGMENT_SIZE,
) -> np.ndarray:
    """
    Remove low-energy segments from a multichannel array.

    RMS energy is measured on a single channel over non-overlapping segments.
    Segments below the threshold are discarded across all channels.

    Parameters
    ----------
    values : np.ndarray
        Shape [num_rows, num_features].
    channel_idx : int
        Column used to measure energy.
    threshold_rms : float
        RMS threshold; segments below this are removed.
    segment_size : int
        Number of samples per segment.

    Returns
    -------
    np.ndarray
        Array with low-energy rows removed.
        Returns the original array unchanged if channel_idx is out of range
        or if filtering would leave an empty array.

    Notes
    -----
    This filter operates on already-normalized values, so for z-scored data
    a near-zero RMS means a nearly-constant (low-information) segment.
    """
    if channel_idx < 0 or channel_idx >= values.shape[1] or values.shape[0] == 0:
        return values

    channel = values[:, channel_idx]
    num_rows = values.shape[0]
    keep = np.ones(num_rows, dtype=bool)

    for start in range(0, num_rows, segment_size):
        end = min(start + segment_size, num_rows)
        chunk = channel[start:end]
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        if rms < threshold_rms:
            keep[start:end] = False

    filtered = values[keep]
    if filtered.shape[0] == 0:
        return values
    return filtered.astype(np.float32)


def preprocess_dataframe(df: pd.DataFrame, normalize: bool = True) -> Optional[pd.DataFrame]:
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
    4. Optionally normalize each feature column
    """
    df = convert_dataframe_to_numeric(df)

    if should_drop_file(df):
        return None

    # Avoid unnecessary fill operations when the DataFrame is already clean.
    if df.isna().any().any():
        df = fill_missing_values(df)

    if normalize:
        df = normalize_dataframe(df)

    return df
