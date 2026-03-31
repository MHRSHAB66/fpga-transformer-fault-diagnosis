import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    LABEL_MAP,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    MISSING_RATIO_THRESHOLD,
    MISSING_FILL_METHOD,
    NORMALIZATION_METHOD,
    EPSILON,
)
from src.data.load_data import build_file_index, load_one_csv


def create_windows_from_dataframe(df, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    values = df.values
    windows = []

    num_rows = len(values)

    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        window = values[start:end]
        windows.append(window)

    return windows


def convert_dataframe_to_numeric(df):
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def should_drop_file(df, missing_ratio_threshold=MISSING_RATIO_THRESHOLD):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    if total_cells == 0:
        return True

    missing_ratio = missing_cells / total_cells
    return missing_ratio > missing_ratio_threshold


def fill_missing_values(df, method=MISSING_FILL_METHOD):
    df = df.copy()

    if method == "median":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    elif method == "mean":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    elif method == "ffill":
        df = df.ffill()

    elif method == "bfill":
        df = df.bfill()

    else:
        raise ValueError(f"Unsupported missing fill method: {method}")

    return df


def normalize_dataframe(df, method=NORMALIZATION_METHOD):
    df = df.copy()

    if method == "zscore":
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + EPSILON)

    elif method == "minmax":
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + EPSILON)

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return df


def preprocess_dataframe(df):
    df = convert_dataframe_to_numeric(df)

    if should_drop_file(df):
        return None

    df = fill_missing_values(df)
    df = normalize_dataframe(df)

    return df


def prepare_dataset(data_dir=DATA_RAW_DIR, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    index_df = build_file_index(data_dir)

    X = []
    y = []

    kept_files = 0
    dropped_files = 0

    for _, row in index_df.iterrows():
        file_path = row["filepath"]
        label_name = row["label"]

        if label_name not in LABEL_MAP:
            continue

        df = load_one_csv(file_path)
        df = preprocess_dataframe(df)

        if df is None:
            dropped_files += 1
            continue

        kept_files += 1

        file_windows = create_windows_from_dataframe(
            df,
            window_size=window_size,
            stride=stride
        )

        label_id = LABEL_MAP[label_name]

        for window in file_windows:
            X.append(window)
            y.append(label_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, kept_files, dropped_files


if __name__ == "__main__":
    X, y, kept_files, dropped_files = prepare_dataset()

    print("Kept files:", kept_files)
    print("Dropped files:", dropped_files)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) > 0:
        print("\nFirst sample shape:", X[0].shape)
        print("First sample label:", y[0])

    unique, counts = np.unique(y, return_counts=True)
    print("\nLabel counts:")
    for label_id, count in zip(unique, counts):
        print(f"Label {label_id}: {count}")