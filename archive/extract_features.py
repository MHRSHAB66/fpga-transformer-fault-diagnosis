"""
SUPERSEDED — archived 2026-08-19. Old MAFAULDA feature-extraction path.

Builds ``data/processed/features_dataset.npz`` from the old preprocessing cache.
Not used by the main pipeline. Kept for reference only; it depends on
``preprocess_data`` (also archived here) — the ``from src.tools.preprocess_data``
import reflects the original location under ``src/tools/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import DATA_PROCESSED_DIR
from src.tools.preprocess_data import preprocess_dataset
from src.tools.signal_preprocessing import prepare_feature_dataset


def extract_feature_dataset(output_dir: Path = DATA_PROCESSED_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_dir = output_dir / "preprocessed_signals"
    manifest_path = output_dir / "preprocessing_manifest.json"

    if not signals_dir.exists() or not manifest_path.exists():
        preprocess_dataset(output_dir=output_dir)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    audio_signals = []
    vibration_signals = []
    labels = []
    source_files = []

    for item in manifest:
        sample_path = output_dir / item["output_path"]
        sample = np.load(sample_path, allow_pickle=True)
        audio_signals.append(sample["audio"])
        vibration_signals.append(sample["vibration"])
        labels.append(str(sample["label"]))
        source_files.append(str(sample["source_filepath"]))

    X = prepare_feature_dataset(
        audio_signals=audio_signals,
        vibration_signals=vibration_signals,
        sample_rate=1000,
    )
    y = np.asarray(labels)

    features_path = output_dir / "features_dataset.npz"
    np.savez_compressed(
        features_path,
        X=X.astype(np.float32),
        y=y,
        source_files=np.asarray(source_files),
    )

    summary = {
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]) if X.ndim == 2 and X.shape[0] > 0 else 0,
        "features_path": str(features_path),
    }

    summary_path = output_dir / "feature_extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    summary = extract_feature_dataset()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
