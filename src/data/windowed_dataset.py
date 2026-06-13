"""
Lazy PyTorch Dataset for MAFAULDA file-level windowing.

The dataset stores lightweight window metadata only. Per-file base-preprocessed
arrays are loaded from the disk cache on demand and sliced into a single window
inside ``__getitem__``.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import LABEL_MAP, WINDOW_SIZE, WINDOW_STRIDE
from src.data.preprocessed_cache import load_or_create_preprocessed_array


class WindowedFileDataset(Dataset):
    """
    Lazily expose fixed-size time windows from a file-level MAFAULDA index.

    Parameters
    ----------
    index_df : pandas.DataFrame
        File index with ``filepath`` and ``label`` columns.
    window_size : int, default=WINDOW_SIZE
        Number of rows per returned time window.
    stride : int, default=WINDOW_STRIDE
        Step size between consecutive windows.

    Notes
    -----
    - File-level splitting must happen before creating this dataset.
    - Only lightweight metadata is stored in memory:
      file path, integer label, and window start offset.
    - Base preprocessing is reused through the per-file disk cache.
    - A one-file in-memory cache avoids repeatedly loading the same
      cached preprocessed array when consecutive samples come from
      the same file.
    """

    def __init__(
        self,
        index_df,
        window_size: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
    ):
        self.index_df = index_df.reset_index(drop=True).copy()
        self.window_size = int(window_size)
        self.stride = int(stride)

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        self.window_index: List[Tuple[str, int, int]] = []
        self.targets: List[int] = []
        self.kept_files = 0
        self.dropped_files = 0

        self._cache_filepath: Optional[str] = None
        self._cache_values: Optional[np.ndarray] = None

        self._build_window_index()

    def _build_window_index(self) -> None:
        """
        Build the in-memory window metadata index.

        For each file in the input split:
        - load or create the cached preprocessed array
        - count valid fixed-length windows
        - store lightweight metadata tuples of
          (file_path, label_id, window_start)
        """
        for _, row in self.index_df.iterrows():
            file_path = row["filepath"]
            label_name = row["label"]

            if label_name not in LABEL_MAP:
                continue

            values = load_or_create_preprocessed_array(file_path)

            if values is None:
                self.dropped_files += 1
                continue

            self.kept_files += 1

            num_rows = len(values)
            if num_rows < self.window_size:
                continue

            label_id = int(LABEL_MAP[label_name])
            for start in range(0, num_rows - self.window_size + 1, self.stride):
                self.window_index.append((str(file_path), label_id, start))
                self.targets.append(label_id)

    def _load_preprocessed_values(self, file_path: str) -> np.ndarray:
        """
        Load one cached preprocessed file array, with a one-file memory cache.

        Parameters
        ----------
        file_path : str
            Path to the raw source file whose preprocessed cache entry should
            be loaded.

        Returns
        -------
        np.ndarray
            Float32 array with shape [num_rows, num_features].
        """
        if self._cache_filepath == file_path and self._cache_values is not None:
            return self._cache_values

        values = load_or_create_preprocessed_array(file_path)
        if values is None:
            raise RuntimeError(
                f"Cached window metadata points to a dropped file: {file_path}"
            )

        self._cache_filepath = file_path
        self._cache_values = values
        return values

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int):
        """
        Return one window tensor and its label tensor.

        Parameters
        ----------
        idx : int
            Window index inside the lightweight metadata list.

        Returns
        -------
        tuple
            (window_tensor, label_tensor)
        """
        file_path, label_id, start = self.window_index[idx]
        values = self._load_preprocessed_values(file_path)

        end = start + self.window_size
        window = values[start:end]

        if window.shape[0] != self.window_size:
            raise RuntimeError(f"Invalid window shape for {file_path}: {window.shape}")

        window_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        return window_tensor, label_tensor