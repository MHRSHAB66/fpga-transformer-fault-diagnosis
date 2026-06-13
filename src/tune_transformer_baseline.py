"""
Random-search tuning script for the Transformer baseline.

This module is responsible for:
1. Building DataLoaders once through the current project pipeline
2. Sampling valid Transformer hyperparameter configurations
3. Training one model per trial
4. Selecting the best trial by validation loss
5. Saving tuning results to disk

Notes
-----
- Validation is used for model selection.
- Test evaluation is intentionally disabled during tuning to avoid
  repeatedly touching the test split.
"""

import json
import random
import time

import torch

from src.config import (
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
)
from src.data.make_dataloaders import build_dataloaders
from src.models.transformer_baseline import TransformerBaseline
from src.train_transformer_baseline import run_training, set_seed


SEARCH_SPACE = {
    "d_model": [32, 64],
    "nhead": [2, 4, 8],
    "num_layers": [2, 3, 4],
    "dim_feedforward": [64, 128, 256],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [1e-4, 5e-4, 1e-3],
}

DEFAULT_NUM_TRIALS = 6


def _save_trial_metric_plot(trials, metric_key, output_path, title, ylabel):
    """
    Save a line plot for a per-trial scalar metric, skipping missing entries.
    """
    trial_indices = []
    metric_values = []

    for trial in trials:
        trial_index = trial.get("trial")
        metric_value = trial.get(metric_key)
        if trial_index is None or metric_value is None:
            continue

        try:
            numeric_value = float(metric_value)
        except (TypeError, ValueError):
            continue

        trial_indices.append(int(trial_index))
        metric_values.append(numeric_value)

    if not metric_values:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trial_indices, metric_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_top_trials_plot(trials, output_path, top_k=5):
    """
    Save a bar chart for the best trials by validation loss.
    """
    valid_trials = []
    for trial in trials:
        best_val_loss = trial.get("best_val_loss")
        trial_index = trial.get("trial")
        if best_val_loss is None or trial_index is None:
            continue

        try:
            numeric_loss = float(best_val_loss)
        except (TypeError, ValueError):
            continue

        valid_trials.append((int(trial_index), numeric_loss))

    if not valid_trials:
        return

    top_trials = sorted(valid_trials, key=lambda item: item[1])[: max(1, min(top_k, len(valid_trials)))]
    labels = [f"trial_{trial_index}" for trial_index, _ in top_trials]
    values = [loss for _, loss in top_trials]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title("Top Trials By Best Validation Loss")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Validation Loss")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_trial_scatter_plot(trials, x_key, y_key, output_path, title, xlabel, ylabel):
    """
    Save a scatter plot for two per-trial scalar metrics, skipping missing entries.
    """
    x_values = []
    y_values = []

    for trial in trials:
        x_value = trial.get(x_key)
        y_value = trial.get(y_key)
        if x_value is None or y_value is None:
            continue

        try:
            numeric_x = float(x_value)
            numeric_y = float(y_value)
        except (TypeError, ValueError):
            continue

        x_values.append(numeric_x)
        y_values.append(numeric_y)

    if not x_values:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_values, y_values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_tuning_plots(trials):
    """
    Save tuning summary plots after all trials are complete.
    """
    if not trials:
        return

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _save_trial_metric_plot(
        trials,
        "best_val_loss",
        results_dir / "transformer_tuning_val_loss_by_trial.png",
        "Best Validation Loss By Trial",
        "Best Validation Loss",
    )
    _save_trial_metric_plot(
        trials,
        "val_acc_at_best_val_loss",
        results_dir / "transformer_tuning_val_acc_by_trial.png",
        "Validation Accuracy At Best Validation Loss By Trial",
        "Validation Accuracy",
    )
    _save_trial_metric_plot(
        trials,
        "trial_seconds",
        results_dir / "transformer_tuning_trial_seconds.png",
        "Trial Duration By Trial",
        "Seconds",
    )
    _save_top_trials_plot(
        trials,
        results_dir / "transformer_tuning_top_trials_by_val_loss.png",
        top_k=min(10, max(5, len(trials))) if trials else 5,
    )
    _save_trial_scatter_plot(
        trials,
        "trial_seconds",
        "best_val_loss",
        results_dir / "transformer_tuning_seconds_vs_val_loss.png",
        "Trial Duration vs Best Validation Loss",
        "Trial Duration (Seconds)",
        "Best Validation Loss",
    )


def sample_hyperparameters(rng):
    """
    Sample one valid hyperparameter configuration.

    Ensures that d_model is divisible by nhead.

    Parameters
    ----------
    rng : random.Random
        Random generator used for reproducible sampling.

    Returns
    -------
    dict
        Valid sampled hyperparameters.
    """
    while True:
        config = {
            "d_model": rng.choice(SEARCH_SPACE["d_model"]),
            "nhead": rng.choice(SEARCH_SPACE["nhead"]),
            "num_layers": rng.choice(SEARCH_SPACE["num_layers"]),
            "dim_feedforward": rng.choice(SEARCH_SPACE["dim_feedforward"]),
            "dropout": rng.choice(SEARCH_SPACE["dropout"]),
            "learning_rate": rng.choice(SEARCH_SPACE["learning_rate"]),
        }

        if config["d_model"] % config["nhead"] == 0:
            return config


def tune(num_trials=DEFAULT_NUM_TRIALS):
    """
    Run random-search tuning for the Transformer baseline.

    Parameters
    ----------
    num_trials : int, default=DEFAULT_NUM_TRIALS
        Number of random trials to run.
    """
    set_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader_build_start = time.perf_counter()
    train_loader, val_loader, _, stats = build_dataloaders()
    dataloader_build_seconds = time.perf_counter() - dataloader_build_start

    print(f"[INFO] build_dataloaders_seconds={dataloader_build_seconds:.3f}")

    if stats.get("max_files") is not None:
        print(
            f"[WARN] build_dataloaders is running with max_files={stats['max_files']}. "
            "This is a debug subset, not a full run."
        )

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = results_dir / "transformer_random_search.json"

    trials = []
    best_trial = None

    for trial_idx in range(num_trials):
        trial_start = time.perf_counter()
        params = sample_hyperparameters(rng)

        print(f"\nTrial {trial_idx + 1}/{num_trials}: {params}")

        model = TransformerBaseline(
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            num_classes=NUM_CLASSES,
        ).to(device)

        _, trial_results = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            device=device,
            learning_rate=params["learning_rate"],
            num_epochs=NUM_EPOCHS,
            num_classes=NUM_CLASSES,
            checkpoint_path=None,
        )

        trial_seconds = time.perf_counter() - trial_start

        trial_summary = {
            "trial": trial_idx + 1,
            "params": params,
            "best_epoch": trial_results["best_epoch"],
            "best_val_loss": trial_results["best_val_loss"],
            "val_acc_at_best_val_loss": trial_results["val_acc_at_best_val_loss"],
            "imbalance_info": trial_results["imbalance_info"],
            "trial_seconds": trial_seconds,
            "timing_info": trial_results["timing_info"],
        }
        trials.append(trial_summary)

        if best_trial is None or trial_summary["best_val_loss"] < best_trial["best_val_loss"]:
            best_trial = trial_summary

    payload = {
        "search_type": "random_search",
        "num_trials": num_trials,
        "dataloader_build_seconds": dataloader_build_seconds,
        "default_reference": {
            "d_model": TRANSFORMER_D_MODEL,
            "nhead": TRANSFORMER_NHEAD,
            "num_layers": TRANSFORMER_NUM_LAYERS,
            "dim_feedforward": TRANSFORMER_DIM_FEEDFORWARD,
            "dropout": TRANSFORMER_DROPOUT,
            "learning_rate": LEARNING_RATE,
        },
        "data_stats": stats,
        "best_trial": best_trial,
        "trials": trials,
    }

    _save_tuning_plots(trials)

    with open(tuning_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nBest trial:")
    print(json.dumps(best_trial, indent=2))
    print(f"\nSaved tuning results to {tuning_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    tune()
