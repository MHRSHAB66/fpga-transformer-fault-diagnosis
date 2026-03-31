import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.config import (
    DATA_RAW_DIR,
    LABEL_MAP,
    RANDOM_SEED,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
)
from src.data.load_data import build_file_index, load_one_csv
from src.data.prepare_dataset import preprocess_dataframe, create_windows_from_dataframe


def split_file_index(index_df):
    total_files = len(index_df)
    train_size = int(TRAIN_RATIO * total_files)
    val_size = int(VAL_RATIO * total_files)

    train_df = index_df.iloc[:train_size].reset_index(drop=True)
    val_df = index_df.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_df = index_df.iloc[train_size + val_size:].reset_index(drop=True)

    return train_df, val_df, test_df


def build_X_y_from_file_index(index_df, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    X = []
    y = []

    kept_files = 0
    dropped_files = 0

    for _, row in index_df.iterrows():
        file_path = row["filepath"]
        label_name = row["label"]

        df = load_one_csv(file_path)
        df = preprocess_dataframe(df)

        if df is None:
            dropped_files += 1
            continue

        kept_files += 1

        windows = create_windows_from_dataframe(
            df,
            window_size=window_size,
            stride=stride,
        )

        label_id = LABEL_MAP[label_name]

        for window in windows:
            X.append(window)
            y.append(label_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, kept_files, dropped_files


def build_dataloaders(data_dir=DATA_RAW_DIR, max_files=1000):
    index_df = build_file_index(data_dir)

    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    index_df = index_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if max_files is not None:
        index_df = index_df.head(max_files).reset_index(drop=True)

    train_index_df, val_index_df, test_index_df = split_file_index(index_df)

    X_train, y_train, train_kept, train_dropped = build_X_y_from_file_index(train_index_df)
    X_val, y_val, val_kept, val_dropped = build_X_y_from_file_index(val_index_df)
    X_test, y_test, test_kept, test_dropped = build_X_y_from_file_index(test_index_df)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    stats = {
        "total_files_after_filter": len(index_df),
        "train_files_kept": train_kept,
        "train_files_dropped": train_dropped,
        "val_files_kept": val_kept,
        "val_files_dropped": val_dropped,
        "test_files_kept": test_kept,
        "test_files_dropped": test_dropped,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    train_loader, val_loader, test_loader, stats = build_dataloaders()

    print("Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nNumber of batches:")
    print("train:", len(train_loader))
    print("val:", len(val_loader))
    print("test:", len(test_loader))

    x_batch, y_batch = next(iter(train_loader))

    print("\nBatch X shape:", x_batch.shape)
    print("Batch y shape:", y_batch.shape)
    print("Batch labels:", y_batch.tolist())