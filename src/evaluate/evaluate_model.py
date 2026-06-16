from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support

from src.config import (
    NUM_CLASSES,
    PROJECT_ROOT,
    RANDOM_SEED,
    LABEL_MAP,
    INPUT_CHANNELS,
    INPUT_LENGTH,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    RUNS_DIR,
)
from src.data.make_dataloaders import build_dataloaders
from src.models.transformer_baseline import TransformerBaseline
from src.experiment_config import build_model_config
from src.evaluation_metrics import (
    fault_vs_normal_summary,
    precision_recall_fbeta_from_confusion,
)
from src.train_transformer_baseline import build_loss_function, evaluate_with_details, set_seed

# Ordered label names matching LABEL_MAP integer keys
_CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]


def _confusion_to_f1(confusion: np.ndarray, class_names: list) -> dict:
    """
    Derive per-class and macro/weighted F1, Precision, Recall from a confusion matrix.

    Returns a dict ready for JSON serialization.
    """
    n = confusion.shape[0]
    per_class: dict = {}
    for i in range(n):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        support = int(confusion[i, :].sum())
        per_class[class_names[i]] = {
            "precision": round(float(precision), 4),
            "recall":    round(float(recall), 4),
            "f1":        round(float(f1), 4),
            "support":   support,
        }

    # Macro averages (unweighted)
    precisions = [per_class[c]["precision"] for c in class_names]
    recalls    = [per_class[c]["recall"]    for c in class_names]
    f1s        = [per_class[c]["f1"]        for c in class_names]
    supports   = [per_class[c]["support"]   for c in class_names]
    total      = sum(supports)

    macro_p = sum(precisions) / n
    macro_r = sum(recalls) / n
    macro_f1 = sum(f1s) / n

    # Weighted averages
    weighted_p  = sum(p * s for p, s in zip(precisions, supports)) / total
    weighted_r  = sum(r * s for r, s in zip(recalls,    supports)) / total
    weighted_f1 = sum(f * s for f, s in zip(f1s,        supports)) / total

    return {
        "per_class": per_class,
        "macro":    {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "weighted": {"precision": round(weighted_p, 4), "recall": round(weighted_r, 4), "f1": round(weighted_f1, 4)},
    }


def _build_model_from_config(model_config: dict, device) -> TransformerBaseline:
    """
    Instantiate TransformerBaseline using a checkpoint's model_config.

    Any missing key falls back to the corresponding config.py default, so this
    is robust to partial configs from older checkpoints.
    """
    return TransformerBaseline(
        input_dim=model_config.get("input_dim", INPUT_CHANNELS),
        seq_len=model_config.get("seq_len", INPUT_LENGTH),
        num_classes=model_config.get("num_classes", NUM_CLASSES),
        d_model=model_config.get("d_model", TRANSFORMER_D_MODEL),
        nhead=model_config.get("nhead", TRANSFORMER_NHEAD),
        num_layers=model_config.get("num_layers", TRANSFORMER_NUM_LAYERS),
        dim_feedforward=model_config.get("dim_feedforward", TRANSFORMER_DIM_FEEDFORWARD),
        dropout=model_config.get("dropout", TRANSFORMER_DROPOUT),
    ).to(device)


def evaluate_saved_model(checkpoint_path=None, output_dir=None, max_files=None):
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
        PROJECT_ROOT / "artifacts" / "checkpoints" / "transformer_baseline_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run `python -m src.train_transformer_baseline` first."
        )

    train_loader, val_loader, test_loader, stats = build_dataloaders(max_files=max_files)
    criterion, imbalance_info = build_loss_function(train_loader, NUM_CLASSES, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reproducibility: prefer the architecture recorded in the checkpoint.
    # Backward-compatible: older checkpoints without model_config still load.
    model_config = checkpoint.get("model_config")
    if model_config:
        print("[INFO] Found model_config in checkpoint:")
        print(json.dumps(model_config, indent=2))
        model = _build_model_from_config(model_config, device)
    else:
        print(
            "[WARN] model_config not found in checkpoint. "
            "Falling back to the current config.py architecture. "
            "Retrain to embed model_config for full reproducibility."
        )
        model = TransformerBaseline().to(device)
        model_config = {}

    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, val_confusion, val_per_class_accuracy = evaluate_with_details(
        model, val_loader, criterion, device, NUM_CLASSES
    )
    test_loss, test_acc, test_confusion, test_per_class_accuracy = evaluate_with_details(
        model, test_loader, criterion, device, NUM_CLASSES
    )

    val_metrics  = _confusion_to_f1(val_confusion,  _CLASS_NAMES)
    test_metrics = _confusion_to_f1(test_confusion, _CLASS_NAMES)
    val_f2_metrics = precision_recall_fbeta_from_confusion(
        val_confusion, _CLASS_NAMES, beta=2.0
    )
    test_f2_metrics = precision_recall_fbeta_from_confusion(
        test_confusion, _CLASS_NAMES, beta=2.0
    )
    val_fault_summary = fault_vs_normal_summary(val_confusion, normal_class_idx=0)
    test_fault_summary = fault_vs_normal_summary(test_confusion, normal_class_idx=0)

    results = {
        "checkpoint_path": str(checkpoint_path),
        "model_config": model_config,
        "data_stats": stats,
        "imbalance_info": imbalance_info,
        "best_epoch": int(checkpoint.get("epoch", -1)),
        # Validation
        "val_loss":               float(val_loss),
        "val_acc":                float(val_acc),
        "val_confusion_matrix":   val_confusion.tolist(),
        "val_per_class_accuracy": val_per_class_accuracy,
        "val_per_class_f1":       val_metrics["per_class"],
        "val_macro_f1":           val_metrics["macro"]["f1"],
        "val_weighted_f1":        val_metrics["weighted"]["f1"],
        "val_macro_precision":    val_metrics["macro"]["precision"],
        "val_macro_recall":       val_metrics["macro"]["recall"],
        "val_f2_metrics":         val_f2_metrics,
        "val_macro_f2":           val_f2_metrics["macro"]["f2"],
        "val_weighted_f2":        val_f2_metrics["weighted"]["f2"],
        "val_fault_vs_normal":    val_fault_summary,
        # Test
        "test_loss":               float(test_loss),
        "test_acc":                float(test_acc),
        "test_confusion_matrix":   test_confusion.tolist(),
        "test_per_class_accuracy": test_per_class_accuracy,
        "test_per_class_f1":       test_metrics["per_class"],
        "test_macro_f1":           test_metrics["macro"]["f1"],
        "test_weighted_f1":        test_metrics["weighted"]["f1"],
        "test_macro_precision":    test_metrics["macro"]["precision"],
        "test_macro_recall":       test_metrics["macro"]["recall"],
        "test_f2_metrics":         test_f2_metrics,
        "test_macro_f2":           test_f2_metrics["macro"]["f2"],
        "test_weighted_f2":        test_f2_metrics["weighted"]["f2"],
        "test_fault_vs_normal":    test_fault_summary,
    }

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = RUNS_DIR / f"{timestamp}_evaluation" / "results"
    else:
        results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save model_config beside checkpoint. If the checkpoint embedded one, use it
    # verbatim; otherwise reconstruct from config.py and flag it as reconstructed
    # so it is not mistaken for the exact recorded architecture.
    if model_config:
        config_to_save = model_config
    else:
        config_to_save = {
            **build_model_config(),
            "_warning": (
                "Reconstructed from config.py; not embedded in this checkpoint. "
                "Retrain to embed model_config in the checkpoint."
            ),
        }
    config_path = results_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    # Console summary
    print("\n" + "=" * 55)
    print("TRANSFORMER EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Checkpoint epoch : {results['best_epoch']}")
    print(f"Val  accuracy    : {val_acc:.4f}")
    print(f"Test accuracy    : {test_acc:.4f}")
    print(f"Val  macro F1    : {val_metrics['macro']['f1']:.4f}")
    print(f"Test macro F1    : {test_metrics['macro']['f1']:.4f}")
    print(f"Test weighted F1 : {test_metrics['weighted']['f1']:.4f}")
    print(f"Test macro F2    : {test_f2_metrics['macro']['f2']:.4f}")
    print(f"Test weighted F2 : {test_f2_metrics['weighted']['f2']:.4f}")
    print(f"Missed faults    : {test_fault_summary['missed_faults']}")
    print(f"False alarms     : {test_fault_summary['false_alarms']}")
    print()
    print("Per-class test metrics:")
    print(f"  {'Class':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "-" * 55)
    for cls in _CLASS_NAMES:
        m = test_metrics["per_class"][cls]
        print(f"  {cls:<16} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>9,}")
    print()
    print(f"Results  -> {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Config   -> {config_path.relative_to(PROJECT_ROOT)}")
    print("=" * 55)

    return results


if __name__ == "__main__":
    evaluate_saved_model()
