import numpy as np

from src.config import DATA_RAW_DIR, LABEL_MAP, WINDOW_SIZE, WINDOW_STRIDE
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


def prepare_dataset(data_dir=DATA_RAW_DIR, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    index_df = build_file_index(data_dir)

    X = []
    y = []

    for _, row in index_df.iterrows():
        file_path = row["filepath"]
        label_name = row["label"]

        if label_name not in LABEL_MAP:
            continue

        df = load_one_csv(file_path)

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

    return X, y


if __name__ == "__main__":
    X, y = prepare_dataset()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) > 0:
        print("\nFirst sample shape:", X[0].shape)
        print("First sample label:", y[0])

    unique, counts = np.unique(y, return_counts=True)
    print("\nLabel counts:")
    for label_id, count in zip(unique, counts):
        print(f"Label {label_id}: {count}")