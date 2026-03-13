from pathlib import Path
import pandas as pd

from src.config import DATA_RAW_SYNTHETIC_DIR, DATA_SAMPLE_DIR, PROJECT_ROOT
from src.data.load_data import build_file_index, load_one_csv


def inspect_directory(directory, name="directory"):
    path = Path(directory).resolve()
    try:
        display_path = path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        display_path = path
    print(f"\n[INFO] Inspecting {name}: {display_path}")
    index_df = build_file_index(directory)

    if len(index_df) == 0:
        print("[INFO] No CSV files found.")
        return

    print("\n[INFO] Label distribution:")
    print(index_df["label"].value_counts(dropna=False))

    first_file = index_df.iloc[0]["filepath"]
    print(f"\n[INFO] Inspecting first file: {first_file}")

    df = load_one_csv(first_file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescribe:")
    print(df.describe().T)


if __name__ == "__main__":
    inspect_directory(DATA_SAMPLE_DIR, "sample directory")
    inspect_directory(DATA_RAW_SYNTHETIC_DIR, "raw directory")