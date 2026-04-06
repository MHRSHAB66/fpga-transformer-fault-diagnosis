import json
import random

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


def sample_hyperparameters(rng):
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
    set_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, stats = build_dataloaders()

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = results_dir / "transformer_random_search.json"

    trials = []
    best_trial = None

    for trial_idx in range(num_trials):
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
            test_loader=test_loader,
            device=device,
            learning_rate=params["learning_rate"],
            num_epochs=NUM_EPOCHS,
            num_classes=NUM_CLASSES,
            checkpoint_path=None,
        )

        trial_summary = {
            "trial": trial_idx + 1,
            "params": params,
            "best_epoch": trial_results["best_epoch"],
            "best_val_loss": trial_results["best_val_loss"],
            "best_val_acc": trial_results["best_val_acc"],
            "test_acc": trial_results["test_acc"],
            "test_loss": trial_results["test_loss"],
            "imbalance_info": trial_results["imbalance_info"],
        }
        trials.append(trial_summary)

        if best_trial is None or trial_summary["best_val_loss"] < best_trial["best_val_loss"]:
            best_trial = trial_summary

    payload = {
        "search_type": "random_search",
        "num_trials": num_trials,
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

    with open(tuning_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nBest trial:")
    print(json.dumps(best_trial, indent=2))
    print(f"\nSaved tuning results to {tuning_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    tune()
