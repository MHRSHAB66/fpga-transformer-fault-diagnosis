from pathlib import Path
import pandas as pd

from src.config import DATA_RAW_SYNTHETIC_DIR, PROJECT_ROOT


def to_project_relative(path_like):
    path = Path(path_like).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return path


def list_csv_files(directory: Path):
    directory = Path(directory).resolve()

    if not directory.exists():
        print(f"[WARN] Directory does not exist: {to_project_relative(directory)}")
        return []

    files = sorted(directory.rglob("*.csv"))
    print(f"[INFO] Found {len(files)} CSV files in {to_project_relative(directory)}")
    return files


def infer_label_from_path(file_path: str) -> str:
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


def build_file_index(directory: Path = DATA_RAW_SYNTHETIC_DIR):
    files = list_csv_files(directory)

    rows = []
    for f in files:
        rows.append({
            "filepath": str(to_project_relative(f)),
            "label": infer_label_from_path(f),
        })

    df = pd.DataFrame(rows)
    return df


def load_one_csv(file_path: str,):
    return pd.read_csv(PROJECT_ROOT / file_path, header = None)


if __name__ == "__main__":
    df = build_file_index()
    print(df.head())
    print("\nLabel counts:")
    if len(df) > 0:
        print(df["label"].value_counts(dropna=False))
    else:
        print("No files found.")