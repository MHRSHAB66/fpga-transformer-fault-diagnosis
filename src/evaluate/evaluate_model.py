from __future__ import annotations

import json

import numpy as np
import torch

from src.config import NUM_CLASSES, PROJECT_ROOT, RANDOM_SEED
from src.data.make_dataloaders import build_dataloaders
from src.models.transformer_baseline import TransformerBaseline
from src.train_transformer_baseline import build_loss_function, evaluate_with_details, set_seed


def evaluate_saved_model():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "transformer_baseline_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run `python -m src.train.train_model` first."
        )

    train_loader, val_loader, test_loader, stats = build_dataloaders()
    criterion, imbalance_info = build_loss_function(train_loader, NUM_CLASSES, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerBaseline().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, val_confusion, val_per_class_accuracy = evaluate_with_details(
        model, val_loader, criterion, device, NUM_CLASSES
    )
    test_loss, test_acc, test_confusion, test_per_class_accuracy = evaluate_with_details(
        model, test_loader, criterion, device, NUM_CLASSES
    )

    results = {
        "checkpoint_path": str(checkpoint_path),
        "data_stats": stats,
        "imbalance_info": imbalance_info,
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_val_loss_from_checkpoint": float(checkpoint.get("val_loss", 0.0)),
        "best_val_acc_from_checkpoint": float(checkpoint.get("val_acc", 0.0)),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_confusion_matrix": val_confusion.tolist(),
        "val_per_class_accuracy": val_per_class_accuracy,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_confusion_matrix": test_confusion.tolist(),
        "test_per_class_accuracy": test_per_class_accuracy,
    }

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "transformer_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("\nValidation confusion matrix:")
    print(np.array(val_confusion))
    print("\nTest confusion matrix:")
    print(np.array(test_confusion))

    return results


if __name__ == "__main__":
    evaluate_saved_model()
