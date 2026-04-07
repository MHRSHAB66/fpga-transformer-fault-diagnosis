from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.data.load_data import build_file_index, load_one_csv
from src.data.prepare_dataset import preprocess_dataframe
from tools.signal_preprocessing import preprocess_signal


MICROPHONE_COLUMN = 7
VIBRATION_COLUMNS = [1, 2, 3, 4, 5, 6]
SAMPLE_RATE = 5000
TARGET_SAMPLE_RATE = 1000


def align_signal_lengths(signals: list[np.ndarray]) -> np.ndarray | None:
    non_empty = [np.asarray(signal, dtype=np.float32).reshape(-1) for signal in signals if np.asarray(signal).size > 0]
    if not non_empty:
        return None

    target_length = min(signal.shape[0] for signal in non_empty)
    if target_length == 0:
        return None

    return np.stack([signal[:target_length] for signal in non_empty], axis=0).astype(np.float32)


def preprocess_dataset(data_dir: Path = DATA_RAW_DIR, output_dir: Path = DATA_PROCESSED_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_dir = output_dir / "preprocessed_signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    index_df = build_file_index(data_dir)
    index_df = index_df[index_df["label"] != "unknown"].reset_index(drop=True)

    manifest = []
    skipped_files = 0

    for idx, row in index_df.iterrows():
        file_path = row["filepath"]
        label = row["label"]

        df = load_one_csv(file_path)
        df = preprocess_dataframe(df)
        if df is None or df.shape[1] <= MICROPHONE_COLUMN:
            skipped_files += 1
            continue

        values = df.to_numpy(dtype=np.float32)
        audio_signal = preprocess_signal(
            values[:, MICROPHONE_COLUMN],
            sample_rate=SAMPLE_RATE,
            target_sr=TARGET_SAMPLE_RATE,
        )

        vibration_signals = align_signal_lengths(
            [
                preprocess_signal(
                    values[:, column_idx],
                    sample_rate=SAMPLE_RATE,
                    target_sr=TARGET_SAMPLE_RATE,
                )
                for column_idx in VIBRATION_COLUMNS
            ]
        )

        if audio_signal.size == 0 or vibration_signals is None:
            skipped_files += 1
            continue

        output_path = signals_dir / f"sample_{idx:05d}.npz"
        np.savez_compressed(
            output_path,
            audio=audio_signal.astype(np.float32),
            vibration=vibration_signals,
            label=label,
            source_filepath=file_path,
        )

        manifest.append(
            {
                "sample_id": idx,
                "label": label,
                "source_filepath": file_path,
                "output_path": str(output_path.relative_to(output_dir)),
                "audio_length": int(audio_signal.shape[0]),
                "vibration_shape": list(vibration_signals.shape),
            }
        )

    manifest_path = output_dir / "preprocessing_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "processed_samples": len(manifest),
        "skipped_files": skipped_files,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
    }
    return summary


def main():
    summary = preprocess_dataset()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
