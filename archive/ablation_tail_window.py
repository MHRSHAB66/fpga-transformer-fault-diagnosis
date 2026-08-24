"""
Ablation: backshift tail window disabled vs enabled.

Trains the Transformer baseline twice under identical settings, changing only
INCLUDE_TAIL_WINDOW (off = baseline, on = tail enabled), and compares:
- train / val / test accuracy
- macro F1 and weighted F1 (computed from the confusion matrix)
- val / test confusion matrices
- window counts per split (normal vs added tail windows)

Both runs use the same random seed, model hyperparameters, and data split, so
any metric difference is attributable to the tail-window samples alone. The
checkpoint is kept in memory only (checkpoint_path=None) so the production
checkpoint at artifacts/checkpoints/ is never overwritten.

Requirements
------------
This script trains real models and therefore needs torch (and a GPU for a
full-dataset run). The original full run takes hours on CPU; use --max-files
for a quick subset comparison, or --epochs to shorten each run.

Usage
-----
    # Full ablation (slow; needs the training environment / GPU):
    python -m src.tools.ablation_tail_window

    # Fast subset smoke comparison:
    python -m src.tools.ablation_tail_window --max-files 60 --epochs 3

Output
------
    artifacts/results/ablation_tail_window.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from src.config import (
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    PROJECT_ROOT,
    RANDOM_SEED,
)


def metrics_from_confusion(confusion) -> dict:
    """
    Derive accuracy and macro/weighted F1 from a confusion matrix.

    Parameters
    ----------
    confusion : array-like, shape [num_classes, num_classes]
        Rows are true labels, columns are predicted labels.

    Returns
    -------
    dict
        {"accuracy", "macro_f1", "weighted_f1", "per_class_f1", "support"}
        macro_f1 averages F1 over classes that have at least one true sample.
    """
    cm = np.asarray(confusion, dtype=np.float64)
    num_classes = cm.shape[0]

    support = cm.sum(axis=1)          # true samples per class
    predicted = cm.sum(axis=0)        # predicted samples per class
    total = cm.sum()

    per_class_f1 = np.zeros(num_classes, dtype=np.float64)
    for i in range(num_classes):
        tp = cm[i, i]
        precision = tp / predicted[i] if predicted[i] > 0 else 0.0
        recall = tp / support[i] if support[i] > 0 else 0.0
        if precision + recall > 0:
            per_class_f1[i] = 2.0 * precision * recall / (precision + recall)

    present = support > 0
    macro_f1 = float(per_class_f1[present].mean()) if present.any() else 0.0
    weighted_f1 = float((per_class_f1 * support).sum() / total) if total > 0 else 0.0
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1.tolist(),
        "support": support.astype(int).tolist(),
    }


def run_one(include_tail: bool, max_files: Optional[int], num_epochs: int) -> dict:
    """
    Build dataloaders and train one configuration; return its metrics.
    """
    # Imported lazily so the module can be inspected without torch installed.
    import torch

    from src.data.make_dataloaders import build_dataloaders
    from src.models.transformer_baseline import TransformerBaseline
    from src.train_transformer_baseline import run_training, set_seed

    label = "tail_on" if include_tail else "tail_off"
    print("\n" + "=" * 60)
    print(f"Run: {label} (include_tail={include_tail})")
    print("=" * 60)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, stats = build_dataloaders(
        max_files=max_files,
        include_tail=include_tail,
    )

    tail_info = stats["tail_window_info"]
    print(f"tail_window_info: {json.dumps(tail_info)}")
    print(
        f"samples -> train={stats['train_samples']} "
        f"val={stats['val_samples']} test={stats['test_samples']}"
    )

    model = TransformerBaseline().to(device)
    _, results = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=num_epochs,
        num_classes=NUM_CLASSES,
        checkpoint_path=None,  # keep in memory; never overwrite production checkpoint
    )

    best_epoch = results["best_epoch"]
    train_acc_history = results["history"].get("train_acc", [])
    train_acc_at_best = (
        float(train_acc_history[best_epoch - 1])
        if 1 <= best_epoch <= len(train_acc_history)
        else None
    )

    val_metrics = metrics_from_confusion(results["val_confusion_matrix"])
    test_metrics = (
        metrics_from_confusion(results["test_confusion_matrix"])
        if results["test_confusion_matrix"] is not None
        else None
    )

    return {
        "include_tail": include_tail,
        "tail_window_info": tail_info,
        "sample_counts": {
            "train": stats["train_samples"],
            "val": stats["val_samples"],
            "test": stats["test_samples"],
        },
        "best_epoch": best_epoch,
        "train_acc_at_best_epoch": train_acc_at_best,
        "val": val_metrics,
        "val_confusion_matrix": results["val_confusion_matrix"],
        "test": test_metrics,
        "test_confusion_matrix": results["test_confusion_matrix"],
    }


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _print_comparison(off: dict, on: dict) -> None:
    print("\n" + "=" * 60)
    print("Ablation comparison (tail_off  ->  tail_on)")
    print("=" * 60)

    rows = [
        ("train_acc", off["train_acc_at_best_epoch"], on["train_acc_at_best_epoch"]),
        ("val_acc", off["val"]["accuracy"], on["val"]["accuracy"]),
        ("val_macro_f1", off["val"]["macro_f1"], on["val"]["macro_f1"]),
        ("val_weighted_f1", off["val"]["weighted_f1"], on["val"]["weighted_f1"]),
    ]
    if off["test"] is not None and on["test"] is not None:
        rows += [
            ("test_acc", off["test"]["accuracy"], on["test"]["accuracy"]),
            ("test_macro_f1", off["test"]["macro_f1"], on["test"]["macro_f1"]),
            ("test_weighted_f1", off["test"]["weighted_f1"], on["test"]["weighted_f1"]),
        ]

    print(f"{'metric':<20}{'tail_off':>12}{'tail_on':>12}{'delta':>12}")
    for name, a, b in rows:
        delta = (b - a) if isinstance(a, float) and isinstance(b, float) else None
        print(f"{name:<20}{_fmt(a):>12}{_fmt(b):>12}{(_fmt(delta) if delta is not None else 'n/a'):>12}")

    print("\nWindow counts (total / tail):")
    for name, run in [("tail_off", off), ("tail_on", on)]:
        info = run["tail_window_info"]
        print(
            f"  {name}: train={info['train']['total']}(+{info['train']['tail']}) "
            f"val={info['val']['total']}(+{info['val']['tail']}) "
            f"test={info['test']['total']}(+{info['test']['tail']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tail-window ablation.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit files for a fast subset run (default: all).")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Epochs per run (default: NUM_EPOCHS={NUM_EPOCHS}).")
    args = parser.parse_args()

    off = run_one(include_tail=False, max_files=args.max_files, num_epochs=args.epochs)
    on = run_one(include_tail=True, max_files=args.max_files, num_epochs=args.epochs)

    _print_comparison(off, on)

    report = {
        "description": "Backshift tail-window ablation (disabled vs enabled).",
        "settings": {
            "max_files": args.max_files,
            "epochs": args.epochs,
            "random_seed": RANDOM_SEED,
            "num_classes": NUM_CLASSES,
        },
        "tail_off": off,
        "tail_on": on,
    }

    output_dir = PROJECT_ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_tail_window.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[INFO] Ablation report saved to {output_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
