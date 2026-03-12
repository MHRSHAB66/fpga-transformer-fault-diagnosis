import torch
import torch.nn as nn
import torch.optim as optim

from src.config import LEARNING_RATE, NUM_EPOCHS
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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, stats = build_dataloaders()

    print("\nData stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    model = MLPBaseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\nFinal test results:")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")


if __name__ == "__main__":
    train()