from pathlib import Path
import pandas as pd

from src.config import (
    DATA_RAW_SYNTHETIC_DIR,
    EXPECTED_COLUMNS,
    EXPECTED_NUM_COLUMNS,
    MIN_ROWS_PER_FILE,
    PROJECT_ROOT,
)
from src.data.load_data import build_file_index, load_one_csv


def check_one_file(file_path):
    result = {
        "filepath": str(Path(file_path).resolve().relative_to(PROJECT_ROOT)),
        "can_read": False,
        "num_rows": None,
        "num_columns": None,
        "missing_total": None,
        "non_numeric_columns": [],
        "issues": [],
    }

    try:
        df = load_one_csv(file_path)
    except Exception as e:
        result["issues"].append(f"read_error: {e}")
        return result

    result["can_read"] = True
    result["num_rows"] = int(df.shape[0])
    result["num_columns"] = int(df.shape[1])

    if df.shape[1] != EXPECTED_NUM_COLUMNS:
        result["issues"].append(
            f"expected_{EXPECTED_NUM_COLUMNS}_columns_got_{df.shape[1]}"
        )

    if list(df.columns) != EXPECTED_COLUMNS:
        result["issues"].append("column_names_mismatch")

    non_numeric_columns = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().sum() > df[col].isna().sum():
            non_numeric_columns.append(col)

    result["non_numeric_columns"] = non_numeric_columns

    if non_numeric_columns:
        result["issues"].append("non_numeric_values_detected")

    missing_per_column = df.isna().sum()
    result["missing_total"] = int(missing_per_column.sum())

    if result["missing_total"] > 0:
        result["issues"].append("missing_values_detected")

    if df.shape[0] < MIN_ROWS_PER_FILE:
        result["issues"].append(f"too_few_rows_below_{MIN_ROWS_PER_FILE}")

    return result


def check_dataset_quality(data_dir=DATA_RAW_SYNTHETIC_DIR):
    index_df = build_file_index(data_dir)

    results = []
    for _, row in index_df.iterrows():
        file_path = row["filepath"]
        results.append(check_one_file(file_path))

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_df = check_dataset_quality()

    print("Total files checked:", len(results_df))

    if len(results_df) == 0:
        print("No files found.")
    else:
        issues_df = results_df[results_df["issues"].apply(len) > 0]

        print("\nFiles with issues:")
        if len(issues_df) == 0:
            print("No issues found.")
        else:
            print(
                issues_df[
                    ["filepath", "issues", "missing_total", "non_numeric_columns"]
                ].to_string(index=False)
            )

        print("\nSummary:")
        print("Readable files:", int(results_df["can_read"].sum()))
        print("Unreadable files:", int((~results_df["can_read"]).sum()))
        print("Files with missing values:", int((results_df["missing_total"].fillna(0) > 0).sum()))
        print("Files with non-numeric values:", int(results_df["non_numeric_columns"].apply(len).gt(0).sum()))