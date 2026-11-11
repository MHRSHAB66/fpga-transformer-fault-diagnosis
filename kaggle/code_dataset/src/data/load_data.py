"""
Utilities for discovering MAFAULDA CSV files, building a file index,
and loading individual CSV files as pandas DataFrames.

This module is responsible for:
1. Finding all CSV files under a given dataset directory.
2. Inferring a grouped fault label from the top-level dataset directory.
3. Building a file index as a pandas DataFrame with:
   - filepath: path relative to PROJECT_ROOT
   - label: inferred class label
4. Loading a single CSV file into a pandas DataFrame.

Notes
-----
- The MAFAULDA CSV files used in this project do not contain a header row,
  so `header=None` is used when reading them.
- File paths are stored relative to PROJECT_ROOT to keep the index portable
  inside the project.
"""

from pathlib import Path
from typing import List, Union

import pandas as pd

from src.config import DATA_RAW_DIR, PROJECT_ROOT


def project_relative_or_absolute(path_like: Union[str, Path]) -> Path:
    """
    Convert an absolute or relative path to a path relative to PROJECT_ROOT
    when possible.

    Parameters
    ----------
    path_like : str or Path
        Input path. Relative paths are resolved from the current working directory.

    Returns
    -------
    Path
        Path relative to PROJECT_ROOT if the input path is inside the project.
        Otherwise, returns the resolved absolute path.
    """
    path = Path(path_like).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return path


def list_csv_files(directory: Path) -> List[Path]:
    """
    Recursively find all CSV files inside the given directory.

    Parameters
    ----------
    directory : Path
        Root directory to search. Relative paths use the current working directory.

    Returns
    -------
    list[Path]
        Sorted list of CSV file paths. Returns an empty list if the directory
        does not exist.

    Raises
    ------
    NotADirectoryError
        If the search path exists but is not a directory.
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        print(f"[WARN] Directory does not exist: {project_relative_or_absolute(directory)}")
        return []

    if not directory.is_dir():
        raise NotADirectoryError(f"CSV search path is not a directory: {directory}")

    files = sorted(path for path in directory.rglob("*.csv") if path.is_file())
    print(
        f"[INFO] Found {len(files)} CSV files in "
        f"{project_relative_or_absolute(directory)}"
    )
    return files


def infer_label_from_path(file_path: Union[str, Path]) -> str:
    """
    Infer the grouped class label from the dataset's top-level class directory.

    Match the exact directory name directly below DATA_RAW_DIR (case-insensitive):
    - horizontal-misalignment / vertical-misalignment -> misalignment
    - overhang / underhang -> bearing_fault
    - imbalance -> imbalance
    - normal -> normal

    Parameters
    ----------
    file_path : str or Path
        Absolute CSV path or a path relative to PROJECT_ROOT, as stored in the index.

    Returns
    -------
    str
        One of:
        - "normal"
        - "imbalance"
        - "misalignment"
        - "bearing_fault"
        - "unknown" if no rule matches

    Notes
    -----
    File names, deeper subdirectories, and partial matches such as "abnormal"
    do not determine the label. A file directly in DATA_RAW_DIR has no class
    directory and returns "unknown". Paths outside DATA_RAW_DIR raise ValueError,
    including when called directly rather than through build_file_index.
    """
    try:
        relative_path = (PROJECT_ROOT / file_path).resolve().relative_to(
            DATA_RAW_DIR.resolve()
        )
    except ValueError as exc:
        raise ValueError(
            f"CSV path must be inside DATA_RAW_DIR ({DATA_RAW_DIR}); "
            f"got: {file_path}"
        ) from exc
    if len(relative_path.parts) < 2:
        return "unknown"

    class_directory = relative_path.parts[0].lower()
    directory_labels = {
        "horizontal-misalignment": "misalignment",
        "vertical-misalignment": "misalignment",
        "overhang": "bearing_fault",
        "underhang": "bearing_fault",
        "imbalance": "imbalance",
        "normal": "normal",
    }
    return directory_labels.get(class_directory, "unknown")


def build_file_index(directory: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """
    Build a file index for all CSV files under the given dataset directory.

    Each row in the returned DataFrame corresponds to one CSV file and contains:
    - filepath: file path relative to PROJECT_ROOT
    - label: inferred label from the file path

    Parameters
    ----------
    directory : Path, default=DATA_RAW_DIR
        DATA_RAW_DIR or one of its subdirectories. Relative paths use the
        current working directory.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - filepath
        - label

    Raises
    ------
    ValueError
        If the directory or a discovered file resolves outside DATA_RAW_DIR.
    NotADirectoryError
        If the search path exists but is not a directory.
    """
    directory = Path(directory).resolve()
    try:
        directory.relative_to(DATA_RAW_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Dataset directory must be inside DATA_RAW_DIR ({DATA_RAW_DIR}); "
            f"got: {directory}"
        ) from exc

    files = list_csv_files(directory)

    rows = []
    for file_path in files:
        rows.append(
            {
                "filepath": str(project_relative_or_absolute(file_path)),
                "label": infer_label_from_path(file_path),
            }
        )

    return pd.DataFrame(rows, columns=["filepath", "label"])


def load_one_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single CSV file as a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Absolute CSV path or a path relative to PROJECT_ROOT. The resolved path
        must be inside DATA_RAW_DIR.

    Returns
    -------
    pandas.DataFrame
        Raw CSV contents as a DataFrame with integer column indices because
        the source files do not include a header row.

    Raises
    ------
    ValueError
        If the resolved path is outside DATA_RAW_DIR.

    File-access and CSV-parsing errors propagate to the caller. Data-quality
    checks and preprocessing are performed by the downstream pipeline.
    """
    path = (PROJECT_ROOT / file_path).resolve()
    try:
        path.relative_to(DATA_RAW_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"CSV path must be inside DATA_RAW_DIR ({DATA_RAW_DIR}); got: {path}"
        ) from exc
    return pd.read_csv(path, header=None)


if __name__ == "__main__":
    # Build and preview the dataset file index for quick inspection.
    index_df = build_file_index()

    print(index_df.head())
    print("\nLabel counts:")

    if len(index_df) > 0:
        print(index_df["label"].value_counts(dropna=False))
    else:
        print("No files found.")
