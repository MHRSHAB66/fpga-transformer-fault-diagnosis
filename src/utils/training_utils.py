"""
Shared training and evaluation utilities used by both training and evaluation scripts.
"""

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


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

    return total_loss / total_samples, total_correct / total_samples


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

            for t, p in zip(y_batch.cpu().numpy(), preds.cpu().numpy()):
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


def get_dataset_targets(dataset):
    if hasattr(dataset, "tensors"):
        return dataset.tensors[1].cpu().numpy()
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    raise AttributeError(
        "Dataset must expose either tensors[1] or targets for class-weight computation."
    )


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


def build_loss_function(train_loader, num_classes, device):
    try:
        train_targets = get_dataset_targets(train_loader.dataset)
    except AttributeError as exc:
        raise AttributeError(
            "Could not extract training labels from train_loader.dataset. "
            "Expected either TensorDataset.tensors[1] or dataset.targets."
        ) from exc

    class_weights, class_counts = compute_class_weights(train_targets, num_classes)

    imbalance_ratio = 1.0
    nonzero_counts = class_counts[class_counts > 0]
    if nonzero_counts.size > 0:
        imbalance_ratio = float(nonzero_counts.max() / nonzero_counts.min())

    use_class_weights = imbalance_ratio > 1.5
    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if use_class_weights
        else nn.CrossEntropyLoss()
    )

    return criterion, {
        "class_counts": class_counts.astype(int).tolist(),
        "imbalance_ratio": imbalance_ratio,
        "class_weights_applied": use_class_weights,
        "class_weights": class_weights.tolist(),
    }
