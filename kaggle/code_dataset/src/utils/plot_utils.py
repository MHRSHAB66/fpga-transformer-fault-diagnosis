"""
Plotting utilities for training and evaluation results.
"""

import numpy as np

from src.config import LABEL_MAP


def _get_class_labels(num_classes):
    labels_by_idx = {class_idx: f"class_{class_idx}" for class_idx in range(num_classes)}
    for label_name, class_idx in LABEL_MAP.items():
        if 0 <= class_idx < num_classes:
            labels_by_idx[class_idx] = label_name
    return [labels_by_idx[class_idx] for class_idx in range(num_classes)]


def _save_history_plots(history, output_dir):
    if not history:
        return

    epochs = history.get("epoch")
    train_loss = history.get("train_loss")
    val_loss = history.get("val_loss")
    train_acc = history.get("train_acc")
    val_acc = history.get("val_acc")

    if not epochs:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    if train_loss is not None and val_loss is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, train_loss, label="train_loss")
        ax.plot(epochs, val_loss, label="val_loss")
        ax.set_title("Transformer Baseline Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "transformer_loss_curve.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    if train_acc is not None and val_acc is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, train_acc, label="train_acc")
        ax.plot(epochs, val_acc, label="val_acc")
        ax.set_title("Transformer Baseline Accuracy Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "transformer_accuracy_curve.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_confusion_matrix_plot(confusion_matrix, output_path, title, class_labels):
    if confusion_matrix is None:
        return

    matrix = np.asarray(confusion_matrix)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    x_labels = class_labels[: matrix.shape[1]]
    y_labels = class_labels[: matrix.shape[0]]
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(y_labels)

    max_value = float(matrix.max()) if matrix.size > 0 else 0.0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                str(int(value)),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_per_class_accuracy_plot(per_class_accuracy, output_path, title, class_labels):
    if not per_class_accuracy:
        return

    labels = []
    values = []

    for class_idx in sorted(per_class_accuracy):
        value = per_class_accuracy[class_idx]
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        label = class_labels[class_idx] if 0 <= class_idx < len(class_labels) else f"class_{class_idx}"
        labels.append(label)
        values.append(numeric_value)

    if not values:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_result_plots(results, num_classes, output_dir):
    if not results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    class_labels = _get_class_labels(num_classes)

    _save_history_plots(results.get("history"), output_dir)
    _save_confusion_matrix_plot(
        results.get("val_confusion_matrix"),
        output_dir / "transformer_val_confusion_matrix.png",
        "Transformer Baseline Validation Confusion Matrix",
        class_labels,
    )
    _save_confusion_matrix_plot(
        results.get("test_confusion_matrix"),
        output_dir / "transformer_test_confusion_matrix.png",
        "Transformer Baseline Test Confusion Matrix",
        class_labels,
    )
    _save_per_class_accuracy_plot(
        results.get("val_per_class_accuracy"),
        output_dir / "transformer_val_per_class_accuracy.png",
        "Transformer Baseline Validation Per-Class Accuracy",
        class_labels,
    )
    _save_per_class_accuracy_plot(
        results.get("test_per_class_accuracy"),
        output_dir / "transformer_test_per_class_accuracy.png",
        "Transformer Baseline Test Per-Class Accuracy",
        class_labels,
    )
