"""
Reproducibility helpers: build and export the experiment configuration snapshot.

All values are read from ``src.config`` (and the active loss decision at runtime),
so the saved snapshot always matches the actual run. Nothing here is hard-coded.

The snapshot has three sections:
- model_config         : Transformer architecture (matches TransformerBaseline defaults)
- training_config      : optimizer / schedule / seed
- preprocessing_config : windowing, normalization, and signal-processing flags

These are embedded inside the checkpoint by the training script and also written
as standalone JSON files for human inspection and downstream stages
(quantization, golden vectors, FPGA constants).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config import (
    BATCH_SIZE,
    CHANNEL_SELECTION,
    INCLUDE_TAIL_WINDOW,
    INPUT_CHANNELS,
    INPUT_LENGTH,
    LEARNING_RATE,
    LOWPASS_FILTER_ENABLED,
    MISSING_FILL_METHOD,
    MISSING_RATIO_THRESHOLD,
    NORMALIZATION_METHOD,
    NUM_CLASSES,
    NUM_EPOCHS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESAMPLE_ENABLED,
    TAIL_WINDOW_MODE,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    WIENER_FILTER_ENABLED,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)


def build_model_config() -> dict:
    """Transformer architecture, mirroring ``TransformerBaseline`` defaults."""
    return {
        "input_dim": INPUT_CHANNELS,
        "seq_len": INPUT_LENGTH,
        "num_classes": NUM_CLASSES,
        "d_model": TRANSFORMER_D_MODEL,
        "nhead": TRANSFORMER_NHEAD,
        "num_layers": TRANSFORMER_NUM_LAYERS,
        "dim_feedforward": TRANSFORMER_DIM_FEEDFORWARD,
        "dropout": TRANSFORMER_DROPOUT,
    }


def build_training_config(loss: str = "weighted_cross_entropy") -> dict:
    """
    Optimizer / schedule / seed.

    Parameters
    ----------
    loss : str
        The loss actually used for the run. Pass "cross_entropy" or
        "weighted_cross_entropy" based on whether class weighting was applied.
    """
    return {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "optimizer": "Adam",
        "loss": loss,
        "random_seed": RANDOM_SEED,
    }


def build_preprocessing_config() -> dict:
    """Windowing, normalization, and optional signal-processing flags."""
    return {
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "normalization": NORMALIZATION_METHOD,
        "missing_ratio_threshold": MISSING_RATIO_THRESHOLD,
        "fill_method": MISSING_FILL_METHOD,
        "include_tail_window": INCLUDE_TAIL_WINDOW,
        "tail_window_mode": TAIL_WINDOW_MODE if INCLUDE_TAIL_WINDOW else None,
        "resample_enabled": RESAMPLE_ENABLED,
        "lowpass_enabled": LOWPASS_FILTER_ENABLED,
        "wiener_enabled": WIENER_FILTER_ENABLED,
    }


def build_experiment_config(loss: str = "weighted_cross_entropy") -> dict:
    """Full experiment snapshot: model + training + preprocessing."""
    return {
        "model_config": build_model_config(),
        "training_config": build_training_config(loss=loss),
        "preprocessing_config": build_preprocessing_config(),
        "channel_selection": CHANNEL_SELECTION,
    }


def save_experiment_config_files(
    loss: str = "weighted_cross_entropy",
    checkpoints_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Write the standalone reproducibility JSON files.

    Returns
    -------
    (model_config_path, experiment_config_path)
        - artifacts/checkpoints/transformer_baseline_model_config.json
        - artifacts/results/transformer_experiment_config.json
    """
    checkpoints_dir = checkpoints_dir or (PROJECT_ROOT / "artifacts" / "checkpoints")
    results_dir = results_dir or (PROJECT_ROOT / "artifacts" / "results")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_config_path = checkpoints_dir / "transformer_baseline_model_config.json"
    experiment_config_path = results_dir / "transformer_experiment_config.json"

    model_config_path.write_text(
        json.dumps(build_model_config(), indent=2), encoding="utf-8"
    )
    experiment_config_path.write_text(
        json.dumps(build_experiment_config(loss=loss), indent=2), encoding="utf-8"
    )
    return model_config_path, experiment_config_path


if __name__ == "__main__":
    # Generate standalone config files for the current config.py state without
    # retraining. Useful to (re)create reproducibility files for an existing
    # checkpoint whose architecture matches the current config.
    mc_path, ec_path = save_experiment_config_files()
    print(f"[INFO] Wrote {mc_path.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Wrote {ec_path.relative_to(PROJECT_ROOT)}")
    print("[INFO] Saved model_config and experiment_config for reproducibility.")
