import copy
import json
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from src.config import LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED, PROJECT_ROOT, NUM_CLASSES
from src.data.make_dataloaders import build_dataloaders
from src.models.transformer_baseline import TransformerBaseline


def evaluate(model, dataloader, criterion, device):
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
    if hasattr(dataset, "tensors"):
        return dataset.tensors[1].cpu().numpy()
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    raise AttributeError("Dataset must expose either tensors[1] or targets for class-weight computation.")


def build_loss_function(train_loader, num_classes, device):
    train_targets = get_dataset_targets(train_loader.dataset)
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
    test_loader,
    device,
    learning_rate,
    num_epochs,
    num_classes,
    checkpoint_path=None,
):
    criterion, imbalance_info = build_loss_function(train_loader, num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1.0
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

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            total_train_correct += (preds == y_batch).sum().item()
            total_train_samples += X_batch.size(0)

        train_loss = total_train_loss / total_train_samples
        train_acc = total_train_correct / total_train_samples

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": best_model_state,
                        "val_loss": best_val_loss,
                        "val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )
                print(
                    f"[INFO] Saved new best model at epoch {best_epoch} "
                    f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}) "
                    f"-> {checkpoint_path.relative_to(PROJECT_ROOT)}"
                )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\n[INFO] Loaded best model from epoch {best_epoch} "
            f"with val_loss={best_val_loss:.4f} and val_acc={best_val_acc:.4f}"
        )

    val_loss, val_acc, val_confusion, val_per_class_accuracy = evaluate_with_details(
        model, val_loader, criterion, device, num_classes
    )
    test_loss, test_acc, confusion, per_class_accuracy = evaluate_with_details(
        model, test_loader, criterion, device, num_classes
    )

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_confusion_matrix": val_confusion.tolist(),
        "val_per_class_accuracy": val_per_class_accuracy,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_confusion_matrix": confusion.tolist(),
        "test_per_class_accuracy": per_class_accuracy,
        "imbalance_info": imbalance_info,
        "history": history,
    }

    return model, results


def train():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, stats = build_dataloaders()

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

    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "data_stats": stats,
                "imbalance_info": results["imbalance_info"],
                "best_epoch": results["best_epoch"],
                "best_val_loss": results["best_val_loss"],
                "best_val_acc": results["best_val_acc"],
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

    print("\nFinal test results:")
    print(f"best_epoch={results['best_epoch']}")
    print(f"best_val_loss={results['best_val_loss']:.4f}")
    print(f"best_val_acc={results['best_val_acc']:.4f}")
    print(f"val_loss={results['val_loss']:.4f}")
    print(f"val_acc={results['val_acc']:.4f}")
    print(f"test_loss={results['test_loss']:.4f}")
    print(f"test_acc={results['test_acc']:.4f}")
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
