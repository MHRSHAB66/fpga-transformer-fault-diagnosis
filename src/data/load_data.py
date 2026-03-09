from pathlib import Path
import pandas as pd

from src.config import DATA_RAW_DIR


def list_csv_files(directory: Path):
    if not directory.exists():
        print(f"[WARN] Directory does not exist: {directory}")
        return []

    files = sorted(directory.rglob("*.csv"))
    print(f"[INFO] Found {len(files)} CSV files in {directory}")
    return files


def infer_label_from_path(file_path: Path):
    p = str(file_path).lower()

    if "normal" in p:
        return "normal"
    if "imbalance" in p:
        return "imbalance"
    if "misalignment" in p:
        return "misalignment"
    if "bearing" in p:
        return "bearing_fault"

    return "unknown"


def build_file_index(directory: Path = DATA_RAW_DIR):
    files = list_csv_files(directory)

    rows = []
    for f in files:
        rows.append({
            "filepath": str(f),
            "label": infer_label_from_path(f),
        })

    df = pd.DataFrame(rows)
    return df


def load_one_csv(file_path: str):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    df = build_file_index()
    print(df.head())
    print("\nLabel counts:")
    if len(df) > 0:
        print(df["label"].value_counts(dropna=False))
    else:
        print("No files found.")