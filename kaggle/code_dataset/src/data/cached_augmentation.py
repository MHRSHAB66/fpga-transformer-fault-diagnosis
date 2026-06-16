"""
Disk cache helpers for materialized training augmentation.

The cache is intentionally limited to augmented training arrays. Validation and
test datasets remain lazy and never use this module.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.config import (
    EPSILON,
    EXPECTED_NUM_COLUMNS,
    INCLUDE_TAIL_WINDOW,
    LABEL_MAP,
    MISSING_FILL_METHOD,
    MISSING_RATIO_THRESHOLD,
    NORMALIZATION_METHOD,
    NORMALIZATION_SCOPE,
    PROJECT_ROOT,
    RANDOM_SEED,
    TAIL_WINDOW_MODE,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)


CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "train_augmentation"


def build_train_augmentation_cache_key(
    train_index_df,
    use_smote: bool,
    use_noise_injection: bool,
    noise_std: float,
    window_size: int = WINDOW_SIZE,
    window_stride: int = WINDOW_STRIDE,
    normalization_scope: str = NORMALIZATION_SCOPE,
    normalization_method: str = NORMALIZATION_METHOD,
    include_tail: bool = INCLUDE_TAIL_WINDOW,
    tail_mode: str = TAIL_WINDOW_MODE,
) -> str:
    """
    Build a stable fingerprint for augmented training arrays.

    The training split identity is represented by the ordered list of
    filepaths and labels in ``train_index_df``.
    """
    train_files = [
        {
            "filepath": str(row["filepath"]),
            "label": str(row["label"]),
        }
        for _, row in train_index_df.reset_index(drop=True).iterrows()
    ]
    payload = {
        "cache_version": 2,
        "train_files": train_files,
        "random_seed": int(RANDOM_SEED),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "expected_num_columns": int(EXPECTED_NUM_COLUMNS),
        "missing_ratio_threshold": float(MISSING_RATIO_THRESHOLD),
        "missing_fill_method": str(MISSING_FILL_METHOD),
        "normalization_method": str(normalization_method),
        "normalization_scope": str(normalization_scope),
        "epsilon": float(EPSILON),
        "label_map": LABEL_MAP,
        "use_smote": bool(use_smote),
        "use_noise_injection": bool(use_noise_injection),
        "noise_std": float(noise_std),
    }
    # Only add tail-window keys when enabled so existing caches built before this
    # option remain valid when the tail window is off (the default).
    if include_tail:
        payload["include_tail_window"] = True
        payload["tail_window_mode"] = str(tail_mode)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def get_train_augmentation_cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.npz"


def load_cached_augmented_train_data(cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    cache_path = get_train_augmentation_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            X_train = cached["X_train"].astype(np.float32, copy=False)
            y_train = cached["y_train"].astype(np.int64, copy=False)
            metadata = json.loads(str(cached["metadata"].item()))
            required_metadata = {"augmentation_info", "train_kept", "train_dropped"}
            if not required_metadata.issubset(metadata):
                missing = required_metadata.difference(metadata)
                raise ValueError(f"missing metadata keys: {sorted(missing)}")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"[WARN] Ignoring unreadable train augmentation cache {cache_path}: {exc}")
        return None

    return X_train, y_train, metadata


def save_augmented_train_data_cache(
    cache_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    metadata: Dict[str, Any],
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_train_augmentation_cache_path(cache_key)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp")

    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            X_train=np.asarray(X_train, dtype=np.float32),
            y_train=np.asarray(y_train, dtype=np.int64),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )

    tmp_path.replace(cache_path)
    return cache_path
