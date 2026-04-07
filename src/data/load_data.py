"""
Utilities for discovering MAFAULDA CSV files, building a file index,
and loading individual CSV files as pandas DataFrames.

This module is responsible for:
1. Finding all CSV files under a given dataset directory.
2. Inferring a coarse fault label from each file path.
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


def to_project_relative(path_like: Union[str, Path]) -> Path:
    """
    Convert an absolute or relative path to a path relative to PROJECT_ROOT
    when possible.

    Parameters
    ----------
    path_like : str or Path
        Input path.

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
        Root directory to search.

    Returns
    -------
    list[Path]
        Sorted list of CSV file paths. Returns an empty list if the directory
        does not exist.
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        print(f"[WARN] Directory does not exist: {to_project_relative(directory)}")
        return []

    files = sorted(directory.rglob("*.csv"))
    print(f"[INFO] Found {len(files)} CSV files in {to_project_relative(directory)}")
    return files


def infer_label_from_path(file_path: Union[str, Path]) -> str:
    """
    Infer the coarse class label from a file path.

    The mapping is based on directory/file naming conventions in MAFAULDA:
    - horizontal-misalignment / vertical-misalignment -> misalignment
    - overhang / underhang -> bearing_fault
    - imbalance -> imbalance
    - normal -> normal

    Parameters
    ----------
    file_path : str or Path
        Path to a CSV file.

    Returns
    -------
    str
        One of:
        - "normal"
        - "imbalance"
        - "misalignment"
        - "bearing_fault"
        - "unknown" if no rule matches
    """
    path_str = str(file_path).lower().replace("\\", "/")

    if "horizontal-misalignment" in path_str or "vertical-misalignment" in path_str:
        return "misalignment"
    if "overhang" in path_str or "underhang" in path_str:
        return "bearing_fault"
    if "imbalance" in path_str:
        return "imbalance"
    if "normal" in path_str:
        return "normal"

    return "unknown"


def build_file_index(directory: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """
    Build a file index for all CSV files under the given dataset directory.

    Each row in the returned DataFrame corresponds to one CSV file and contains:
    - filepath: file path relative to PROJECT_ROOT
    - label: inferred label from the file path

    Parameters
    ----------
    directory : Path, default=DATA_RAW_DIR
        Dataset root directory.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - filepath
        - label
    """
    files = list_csv_files(directory)

    rows = []
    for file_path in files:
        rows.append(
            {
                "filepath": str(to_project_relative(file_path)),
                "label": infer_label_from_path(file_path),
            }
        )

    return pd.DataFrame(rows)


def load_one_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single CSV file as a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        File path relative to PROJECT_ROOT, or a Path compatible with PROJECT_ROOT.

    Returns
    -------
    pandas.DataFrame
        Raw CSV contents as a DataFrame with integer column indices because
        the source files do not include a header row.
    """
    return pd.read_csv(PROJECT_ROOT / file_path, header=None)


if __name__ == "__main__":
    # Build and preview the dataset file index for quick inspection.
    index_df = build_file_index()

    print(index_df.head())
    print("\nLabel counts:")

    if len(index_df) > 0:
        print(index_df["label"].value_counts(dropna=False))
    else:
        print("No files found.")