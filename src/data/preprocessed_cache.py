"""
Disk cache helpers for base per-file preprocessing.

This module caches the output of the shared preprocessing path:
raw CSV -> load_one_csv -> preprocess_dataframe -> float32 ndarray

The cache is reused by both:
- materialized training array construction
- lazy window datasets for validation/test and non-augmented training

Important
---------
- This cache is different from the train-only augmentation cache.
- The cache key depends on the source file identity, file modification info,
  preprocessing-relevant config, and an explicit cache version.
"""

from pathlib import Path
from typing import Optional
import hashlib
import json

import numpy as np

from src.config import (
    EPSILON,
    EXPECTED_NUM_COLUMNS,
    MISSING_FILL_METHOD,
    MISSING_RATIO_THRESHOLD,
    NORMALIZATION_METHOD,
    PROJECT_ROOT,
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
from src.data.load_data import load_one_csv
from src.data.prepare_dataset import (
    preprocess_dataframe,
    apply_lowpass_filter,
    apply_wiener_filter,
    apply_resample_array,
    apply_low_energy_filter,
)


CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "preprocessed_files"
CACHE_VERSION = 1


def _build_preprocessed_cache_key(file_path) -> str:
    """
    Build a stable cache key for one raw input file.

    Parameters
    ----------
    file_path : str or Path-like
        Path to one raw CSV file.

    Returns
    -------
    str
        SHA256 fingerprint representing the source file state and
        preprocessing-relevant configuration.

    Notes
    -----
    The key depends on:
    - resolved file path
    - file size
    - file modification time
    - preprocessing-related config values
    - explicit cache version
    """
    file_path = Path(file_path)
    stat = file_path.stat()

    payload = {
        "cache_version": CACHE_VERSION,
        "filepath": str(file_path.resolve()),
        "file_size": int(stat.st_size),
        "file_mtime_ns": int(
            getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
        ),
        "expected_num_columns": int(EXPECTED_NUM_COLUMNS),
        "missing_ratio_threshold": float(MISSING_RATIO_THRESHOLD),
        "missing_fill_method": str(MISSING_FILL_METHOD),
        "normalization_method": str(NORMALIZATION_METHOD),
        "epsilon": float(EPSILON),
    }

    # Optional signal processing flags are only added to the payload when they
    # are enabled. This keeps the hash identical to the original when all flags
    # are False, so existing cache entries remain valid.
    if LOWPASS_FILTER_ENABLED:
        payload["lowpass_filter_enabled"] = True
        payload["lowpass_cutoff_hz"] = float(LOWPASS_CUTOFF_HZ)
        payload["lowpass_filter_order"] = int(LOWPASS_FILTER_ORDER)
        payload["mafaulda_sample_rate"] = int(MAFAULDA_SAMPLE_RATE)
    if WIENER_FILTER_ENABLED:
        payload["wiener_filter_enabled"] = True
        payload["wiener_kernel_size"] = int(WIENER_KERNEL_SIZE)
    if RESAMPLE_ENABLED:
        payload["resample_enabled"] = True
        payload["mafaulda_sample_rate"] = int(MAFAULDA_SAMPLE_RATE)
        payload["resample_target_sr"] = int(RESAMPLE_TARGET_SR)
    if LOW_ENERGY_FILTER_ENABLED:
        payload["low_energy_filter_enabled"] = True
        payload["low_energy_channel_idx"] = int(LOW_ENERGY_CHANNEL_IDX)
        payload["low_energy_threshold_rms"] = float(LOW_ENERGY_THRESHOLD_RMS)
        payload["low_energy_segment_size"] = int(LOW_ENERGY_SEGMENT_SIZE)

    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def get_preprocessed_cache_path(file_path) -> Path:
    """
    Return the cache file path for one raw input file.

    Parameters
    ----------
    file_path : str or Path-like
        Path to one raw CSV file.

    Returns
    -------
    pathlib.Path
        Cache file path under the preprocessing cache directory.
    """
    cache_key = _build_preprocessed_cache_key(file_path)
    return CACHE_DIR / f"{cache_key}.npz"


def load_cached_preprocessed_array(file_path) -> Optional[np.ndarray]:
    """
    Load a cached preprocessed array for one raw input file.

    Parameters
    ----------
    file_path : str or Path-like
        Path to one raw CSV file.

    Returns
    -------
    np.ndarray or None
        Cached float32 array if available and readable, otherwise None.
    """
    cache_path = get_preprocessed_cache_path(file_path)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            values = cached["values"].astype(np.float32, copy=False)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[WARN] Ignoring unreadable preprocessing cache {cache_path}: {exc}")
        return None

    return values


def save_preprocessed_array_cache(file_path, values: np.ndarray) -> Path:
    """
    Save one preprocessed float32 array to the preprocessing cache.

    Parameters
    ----------
    file_path : str or Path-like
        Path to one raw CSV file.
    values : np.ndarray
        Preprocessed array with shape [num_rows, num_features].

    Returns
    -------
    pathlib.Path
        Final cache file path.

    Notes
    -----
    The file is first written to a temporary path and then atomically
    replaced to reduce the chance of leaving a partial cache file.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = get_preprocessed_cache_path(file_path)
    tmp_path = cache_path.with_suffix(".tmp")

    values = np.asarray(values, dtype=np.float32)

    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            values=values,
        )

    tmp_path.replace(cache_path)
    return cache_path


def load_or_create_preprocessed_array(file_path) -> Optional[np.ndarray]:
    """
    Load a cached preprocessed array if it exists; otherwise create it.

    Parameters
    ----------
    file_path : str or Path-like
        Path to one raw CSV file.

    Returns
    -------
    np.ndarray or None
        Float32 preprocessed array, or None if preprocessing drops the file.
    """
    cached = load_cached_preprocessed_array(file_path)
    if cached is not None:
        return cached

    df = load_one_csv(file_path)
    df = preprocess_dataframe(df)

    if df is None:
        return None

    values = df.to_numpy(copy=True).astype(np.float32, copy=False)

    # Optional signal processing steps, applied after per-file normalization.
    # All disabled by default; enable via config flags.
    # Order: lowpass → Wiener → resample → low-energy filter.
    if LOWPASS_FILTER_ENABLED:
        values = apply_lowpass_filter(values, LOWPASS_CUTOFF_HZ, MAFAULDA_SAMPLE_RATE, LOWPASS_FILTER_ORDER)
    if WIENER_FILTER_ENABLED:
        values = apply_wiener_filter(values, WIENER_KERNEL_SIZE)
    if RESAMPLE_ENABLED:
        values = apply_resample_array(values, MAFAULDA_SAMPLE_RATE, RESAMPLE_TARGET_SR)
    if LOW_ENERGY_FILTER_ENABLED:
        values = apply_low_energy_filter(values, LOW_ENERGY_CHANNEL_IDX, LOW_ENERGY_THRESHOLD_RMS, LOW_ENERGY_SEGMENT_SIZE)

    save_preprocessed_array_cache(file_path, values)
    return values