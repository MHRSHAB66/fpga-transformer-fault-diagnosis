from pathlib import Path
import copy
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import json

from src.config import LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED, PROJECT_ROOT
from src.data.make_dataloaders import build_dataloaders
from src.models.mlp_baseline import MLPBaseline


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def train():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, stats = build_dataloaders()

    print("\nData stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    model = MLPBaseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = -1.0
    best_epoch = -1
    best_model_state = None

    checkpoints_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "mlp_baseline_best.pt"

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "mlp_baseline_history.json"

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(NUM_EPOCHS):
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, best_model_path)
            print(f"[INFO] Saved new best model at epoch {best_epoch} -> {best_model_path.relative_to(PROJECT_ROOT)}")

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[INFO] Loaded best model from epoch {best_epoch} with val_acc={best_val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\nFinal test results:")
    print(f"best_epoch={best_epoch}")
    print(f"best_val_acc={best_val_acc:.4f}")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    print(f"best_model_path={best_model_path.relative_to(PROJECT_ROOT)}")
    print(f"history_path={history_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    train()