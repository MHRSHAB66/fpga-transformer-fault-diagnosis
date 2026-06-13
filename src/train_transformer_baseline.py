"""
Training script for the Transformer baseline model.

This module is responsible for:
1. Building DataLoaders through the current project pipeline
2. Handling optional class-weighted loss for class imbalance
3. Training the Transformer baseline
4. Saving the best checkpoint based on validation loss
5. Evaluating on validation and test sets
6. Writing training history and final metrics to disk

Notes
-----
- The training split may be either:
    * a TensorDataset (materialized / augmented path), or
    * a lazy WindowedFileDataset
- Validation and test evaluation always operate on the DataLoaders
  returned by build_dataloaders().
- The best model is selected using validation loss.
"""

import copy
import json
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from src.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_SEED,
    PROJECT_ROOT,
    NUM_CLASSES,
    LABEL_MAP,
)
from src.data.make_dataloaders import build_dataloaders
from src.models.transformer_baseline import TransformerBaseline


def _get_class_labels(num_classes):
    """
    Build readable class labels from LABEL_MAP when available.
    """
    labels_by_idx = {class_idx: f"class_{class_idx}" for class_idx in range(num_classes)}
    for label_name, class_idx in LABEL_MAP.items():
        if 0 <= class_idx < num_classes:
            labels_by_idx[class_idx] = label_name
    return [labels_by_idx[class_idx] for class_idx in range(num_classes)]


def _save_history_plots(history):
    """
    Save training history plots if the expected arrays are available.
    """
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

    output_dir = PROJECT_ROOT / "artifacts" / "results"
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
            output_dir / "transformer_baseline_loss_curve.png",
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
            output_dir / "transformer_baseline_accuracy_curve.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_confusion_matrix_plot(confusion_matrix, output_path, title, class_labels):
    """
    Save a confusion matrix plot if matrix data is available and valid.
    """
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
    """
    Save a per-class accuracy bar plot, skipping missing values safely.
    """
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


def _save_result_plots(results, num_classes):
    """
    Save all supported result plots, skipping unavailable data safely.
    """
    if not results:
        return

    output_dir = PROJECT_ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = _get_class_labels(num_classes)

    _save_history_plots(results.get("history"))
    _save_confusion_matrix_plot(
        results.get("val_confusion_matrix"),
        output_dir / "transformer_baseline_val_confusion_matrix.png",
        "Transformer Baseline Validation Confusion Matrix",
        class_labels,
    )
    _save_confusion_matrix_plot(
        results.get("test_confusion_matrix"),
        output_dir / "transformer_baseline_test_confusion_matrix.png",
        "Transformer Baseline Test Confusion Matrix",
        class_labels,
    )
    _save_per_class_accuracy_plot(
        results.get("val_per_class_accuracy"),
        output_dir / "transformer_baseline_val_per_class_accuracy.png",
        "Transformer Baseline Validation Per-Class Accuracy",
        class_labels,
    )
    _save_per_class_accuracy_plot(
        results.get("test_per_class_accuracy"),
        output_dir / "transformer_baseline_test_per_class_accuracy.png",
        "Transformer Baseline Test Per-Class Accuracy",
        class_labels,
    )


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate a model on a DataLoader and return average loss and accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    dataloader : torch.utils.data.DataLoader
        Evaluation DataLoader.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[float, float]
        (average_loss, accuracy)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += X_batch.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def compute_class_weights(targets, num_classes):
    """
    Compute inverse-frequency class weights.

    Parameters
    ----------
    targets : np.ndarray
        Integer class labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        (class_weights, class_counts)
    """
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    nonzero = counts[counts > 0]

    if nonzero.size == 0:
        return torch.ones(num_classes, dtype=torch.float32), counts

    weights = np.zeros(num_classes, dtype=np.float32)
    total = counts.sum()

    for class_idx, count in enumerate(counts):
        if count > 0:
            weights[class_idx] = total / (num_classes * count)

    return torch.tensor(weights, dtype=torch.float32), counts


def get_dataset_targets(dataset):
    """
    Extract targets from either a TensorDataset or a lazy dataset.

    Supported cases
    ---------------
    - TensorDataset: expects labels in tensors[1]
    - Lazy dataset: expects a `targets` attribute

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset attached to the training DataLoader.

    Returns
    -------
    np.ndarray
        Integer class labels.

    Raises
    ------
    AttributeError
        If the dataset does not expose labels in one of the supported ways.
    """
    if hasattr(dataset, "tensors"):
        return dataset.tensors[1].cpu().numpy()

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)

    raise AttributeError(
        "Dataset must expose either tensors[1] or targets for class-weight computation."
    )


def build_loss_function(train_loader, num_classes, device):
    """
    Build the training loss function.

    Class weights are applied only when the observed imbalance ratio
    is sufficiently high.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training DataLoader.
    num_classes : int
        Number of classes.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[torch.nn.Module, dict]
        (criterion, imbalance_info)
    """
    try:
        train_targets = get_dataset_targets(train_loader.dataset)
    except AttributeError as exc:
        raise AttributeError(
            "Could not extract training labels from train_loader.dataset. "
            "Expected either TensorDataset.tensors[1] or dataset.targets. "
            "This is required because build_dataloaders() may return either "
            "a materialized TensorDataset or a lazy WindowedFileDataset."
        ) from exc

    class_weights, class_counts = compute_class_weights(train_targets, num_classes)

    imbalance_ratio = 1.0
    nonzero_counts = class_counts[class_counts > 0]
    if nonzero_counts.size > 0:
        imbalance_ratio = float(nonzero_counts.max() / nonzero_counts.min())

    use_class_weights = imbalance_ratio > 1.5
    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    imbalance_info = {
        "class_counts": class_counts.astype(int).tolist(),
        "imbalance_ratio": imbalance_ratio,
        "class_weights_applied": use_class_weights,
        "class_weights": class_weights.tolist(),
    }

    return criterion, imbalance_info


def evaluate_with_details(model, dataloader, criterion, device, num_classes):
    """
    Evaluate a model and also return confusion matrix and per-class accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    dataloader : torch.utils.data.DataLoader
        Evaluation DataLoader.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        Target device.
    num_classes : int
        Number of classes.

    Returns
    -------
    tuple
        (
            average_loss,
            accuracy,
            confusion_matrix,
            per_class_accuracy_dict
        )
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += X_batch.size(0)

            y_true = y_batch.cpu().numpy()
            y_pred = preds.cpu().numpy()

            for t, p in zip(y_true, y_pred):
                confusion[t, p] += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    per_class_accuracy = {}
    for class_idx in range(num_classes):
        class_total = confusion[class_idx].sum()
        if class_total == 0:
            per_class_accuracy[class_idx] = None
        else:
            per_class_accuracy[class_idx] = confusion[class_idx, class_idx] / class_total

    return avg_loss, accuracy, confusion, per_class_accuracy


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def run_training(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate,
    num_epochs,
    num_classes,
    test_loader=None,
    checkpoint_path=None,
):
    """
    Train the model and evaluate it on validation / optional test data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : torch.utils.data.DataLoader
        Training DataLoader.
    val_loader : torch.utils.data.DataLoader
        Validation DataLoader.
    device : torch.device
        Target device.
    learning_rate : float
        Optimizer learning rate.
    num_epochs : int
        Number of training epochs.
    num_classes : int
        Number of classes.
    test_loader : torch.utils.data.DataLoader or None, optional
        Test DataLoader. If None, test evaluation is skipped.
    checkpoint_path : pathlib.Path or None, optional
        Where to save the best checkpoint.

    Returns
    -------
    tuple[torch.nn.Module, dict]
        (trained_model, results_dict)
    """
    criterion, imbalance_info = build_loss_function(train_loader, num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_acc_at_best_val_loss = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    timing_info = {
        # This measures the full first optimization step of epoch 1:
        # first batch retrieval from the DataLoader + forward/backward/update.
        "first_train_step_seconds": None,
        "val_seconds_history": [],
        "total_training_seconds": None,
    }

    training_start = time.perf_counter()

    def train_one_batch(X_batch, y_batch):
        """
        Run one optimization step and return basic batch statistics.
        """
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_correct = (preds == y_batch).sum().item()
        batch_size = X_batch.size(0)

        return loss.item(), batch_correct, batch_size

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        train_iter = iter(train_loader)

        if epoch == 0:
            try:
                # This timing intentionally includes more than data loading alone.
                # It covers first batch retrieval plus the first training step.
                first_batch_start = time.perf_counter()
                X_batch, y_batch = next(train_iter)
                loss_value, batch_correct, batch_size = train_one_batch(X_batch, y_batch)
                timing_info["first_train_step_seconds"] = (
                    time.perf_counter() - first_batch_start
                )

                total_train_loss += loss_value * batch_size
                total_train_correct += batch_correct
                total_train_samples += batch_size
            except StopIteration as exc:
                raise ValueError("Training DataLoader produced no batches.") from exc

        for X_batch, y_batch in train_iter:
            loss_value, batch_correct, batch_size = train_one_batch(X_batch, y_batch)

            total_train_loss += loss_value * batch_size
            total_train_correct += batch_correct
            total_train_samples += batch_size

        train_loss = total_train_loss / total_train_samples
        train_acc = total_train_correct / total_train_samples

        val_start = time.perf_counter()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_seconds = time.perf_counter() - val_start
        timing_info["val_seconds_history"].append(val_seconds)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_acc_at_best_val_loss = val_acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": best_model_state,
                        "val_loss": best_val_loss,
                        "val_acc_at_best_val_loss": val_acc_at_best_val_loss,
                    },
                    checkpoint_path,
                )
                print(
                    f"[INFO] Saved new best model at epoch {best_epoch} "
                    f"(val_loss={best_val_loss:.4f}, "
                    f"val_acc={val_acc_at_best_val_loss:.4f}) "
                    f"-> {checkpoint_path.relative_to(PROJECT_ROOT)}"
                )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_seconds={val_seconds:.3f}"
        )

    timing_info["total_training_seconds"] = time.perf_counter() - training_start

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\n[INFO] Loaded best model from epoch {best_epoch} "
            f"with val_loss={best_val_loss:.4f} "
            f"and val_acc={val_acc_at_best_val_loss:.4f}"
        )

    val_loss, val_acc, val_confusion, val_per_class_accuracy = evaluate_with_details(
        model, val_loader, criterion, device, num_classes
    )

    if test_loader is not None:
        test_loss, test_acc, test_confusion, test_per_class_accuracy = evaluate_with_details(
            model, test_loader, criterion, device, num_classes
        )
        test_confusion_matrix = test_confusion.tolist()
    else:
        test_loss = None
        test_acc = None
        test_confusion_matrix = None
        test_per_class_accuracy = None

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_acc_at_best_val_loss": val_acc_at_best_val_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_confusion_matrix": val_confusion.tolist(),
        "val_per_class_accuracy": val_per_class_accuracy,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_confusion_matrix": test_confusion_matrix,
        "test_per_class_accuracy": test_per_class_accuracy,
        "imbalance_info": imbalance_info,
        "history": history,
        "timing_info": timing_info,
    }

    return model, results


def train():
    """
    Entry point for full Transformer baseline training.
    """
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader_build_start = time.perf_counter()
    train_loader, val_loader, test_loader, stats = build_dataloaders()
    dataloader_build_seconds = time.perf_counter() - dataloader_build_start

    print(f"[INFO] build_dataloaders_seconds={dataloader_build_seconds:.3f}")

    if stats.get("max_files") is not None:
        print(
            f"[WARN] build_dataloaders is running with max_files={stats['max_files']}. "
            "This is a debug subset, not a full run."
        )

    print("\nData stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    model = TransformerBaseline().to(device)

    checkpoints_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "transformer_baseline_best.pt"

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "transformer_baseline_history.json"
    metrics_path = results_dir / "transformer_baseline_metrics.json"

    _, results = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        num_classes=NUM_CLASSES,
        checkpoint_path=best_model_path,
    )

    _save_result_plots(results, NUM_CLASSES)

    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "data_stats": stats,
                "dataloader_build_seconds": dataloader_build_seconds,
                "timing_info": results["timing_info"],
                "imbalance_info": results["imbalance_info"],
                "best_epoch": results["best_epoch"],
                "best_val_loss": results["best_val_loss"],
                "val_acc_at_best_val_loss": results["val_acc_at_best_val_loss"],
                "val_loss": results["val_loss"],
                "val_acc": results["val_acc"],
                "val_confusion_matrix": results["val_confusion_matrix"],
                "val_per_class_accuracy": results["val_per_class_accuracy"],
                "test_loss": results["test_loss"],
                "test_acc": results["test_acc"],
                "test_confusion_matrix": results["test_confusion_matrix"],
                "test_per_class_accuracy": results["test_per_class_accuracy"],
            },
            f,
            indent=2,
        )

    print("\nFinal evaluation results:")
    print(f"best_epoch={results['best_epoch']}")
    print(f"best_val_loss={results['best_val_loss']:.4f}")
    print(f"val_acc_at_best_val_loss={results['val_acc_at_best_val_loss']:.4f}")
    print(f"val_loss={results['val_loss']:.4f}")
    print(f"val_acc={results['val_acc']:.4f}")
    print(f"test_loss={results['test_loss']:.4f}")
    print(f"test_acc={results['test_acc']:.4f}")
    print(
        f"first_train_step_seconds="
        f"{results['timing_info']['first_train_step_seconds']:.3f}"
    )
    print(
        f"total_training_seconds="
        f"{results['timing_info']['total_training_seconds']:.3f}"
    )
    print(f"best_model_path={best_model_path.relative_to(PROJECT_ROOT)}")
    print(f"history_path={history_path.relative_to(PROJECT_ROOT)}")
    print(f"metrics_path={metrics_path.relative_to(PROJECT_ROOT)}")

    print("\nClass imbalance handling:")
    print(json.dumps(results["imbalance_info"], indent=2))

    print("\nValidation confusion matrix:")
    print(np.array(results["val_confusion_matrix"]))

    print("\nTest confusion matrix:")
    print(np.array(results["test_confusion_matrix"]))

    print("\nValidation per-class accuracy:")
    for class_idx, acc in results["val_per_class_accuracy"].items():
        if acc is None:
            print(f"class_{class_idx}: no samples")
        else:
            print(f"class_{class_idx}: {acc:.4f}")

    print("\nTest per-class accuracy:")
    for class_idx, acc in results["test_per_class_accuracy"].items():
        if acc is None:
            print(f"class_{class_idx}: no samples")
        else:
            print(f"class_{class_idx}: {acc:.4f}")


if __name__ == "__main__":
    train()
