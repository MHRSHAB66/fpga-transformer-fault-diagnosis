"""
Lazy PyTorch Dataset for MAFAULDA file-level windowing.

The dataset stores lightweight window metadata only. Raw CSV data is loaded,
preprocessed, and sliced into a single window on demand in ``__getitem__``.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import LABEL_MAP, WINDOW_SIZE, WINDOW_STRIDE
from src.data.load_data import load_one_csv
from src.data.prepare_dataset import preprocess_dataframe


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
    - Only lightweight metadata is stored: file path, integer label, and window
      start offset.
    - A one-file cache avoids reloading and preprocessing the same file when
      consecutive samples come from that file.
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
        for _, row in self.index_df.iterrows():
            file_path = row["filepath"]
            label_name = row["label"]

            if label_name not in LABEL_MAP:
                continue

            df = load_one_csv(file_path)
            df = preprocess_dataframe(df)

            if df is None:
                self.dropped_files += 1
                continue

            self.kept_files += 1

            num_rows = len(df)
            if num_rows < self.window_size:
                continue

            label_id = int(LABEL_MAP[label_name])
            for start in range(0, num_rows - self.window_size + 1, self.stride):
                self.window_index.append((str(file_path), label_id, start))
                self.targets.append(label_id)

    def _load_preprocessed_values(self, file_path: str) -> np.ndarray:
        if self._cache_filepath == file_path and self._cache_values is not None:
            return self._cache_values

        df = load_one_csv(file_path)
        df = preprocess_dataframe(df)
        if df is None:
            raise RuntimeError(f"Cached window metadata points to a dropped file: {file_path}")

        values = df.to_numpy(copy=True).astype(np.float32, copy=False)
        self._cache_filepath = file_path
        self._cache_values = values
        return values

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int):
        file_path, label_id, start = self.window_index[idx]
        values = self._load_preprocessed_values(file_path)

        end = start + self.window_size
        window = values[start:end]

        if window.shape[0] != self.window_size:
            raise RuntimeError(f"Invalid window shape for {file_path}: {window.shape}")

        window_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        return window_tensor, label_tensor
