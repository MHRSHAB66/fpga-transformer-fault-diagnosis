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
from datetime import datetime
import json
import time

import numpy as np
import torch
import torch.optim as optim

from src.config import (
    BATCH_SIZE,
    DEFAULT_RUN_NAME,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    NORMALIZATION_METHOD,
    NORMALIZATION_SCOPE,
    NUM_EPOCHS,
    RANDOM_SEED,
    PROJECT_ROOT,
    RUNS_DIR,
    NUM_CLASSES,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from src.data.make_dataloaders import build_dataloaders
from src.evaluation_metrics import (
    fault_vs_normal_summary,
    precision_recall_fbeta_from_confusion,
)
from src.models.transformer_baseline import TransformerBaseline
from src.experiment_config import (
    build_model_config,
    build_training_config,
    build_preprocessing_config,
    save_experiment_config_files,
)
from src.utils.plot_utils import _get_class_labels, _save_result_plots
from src.utils.training_utils import (
    set_seed,
    evaluate,
    evaluate_with_details,
    build_loss_function,
)


def _create_run_dir(run_name: str) -> dict:
    """
    Create a timestamped run directory without touching legacy artifacts.
    """
    safe_run_name = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name
    ).strip("_") or "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{safe_run_name}"
    paths = {
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "results_dir": run_dir / "results",
        "plots_dir": run_dir / "plots",
        "logs_dir": run_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _display_path(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


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
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    model_config_overrides=None,
    training_config_overrides=None,
    preprocessing_config_overrides=None,
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
    early_stopping_patience : int or None
        Number of epochs without validation-loss improvement before stopping.

    Returns
    -------
    tuple[torch.nn.Module, dict]
        (trained_model, results_dict)
    """
    criterion, imbalance_info = build_loss_function(train_loader, num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Record the loss actually used so the saved config matches the run.
    loss_name = (
        "weighted_cross_entropy"
        if imbalance_info.get("class_weights_applied")
        else "cross_entropy"
    )

    val_acc_at_best_val_loss = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    stopped_epoch = None
    early_stopped = False
    epochs_without_improvement = 0

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
        "train_seconds_history": [],
        "total_training_seconds": None,
        "evaluation_seconds": {},
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
        epoch_train_start = time.perf_counter()
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
        train_seconds = time.perf_counter() - epoch_train_start

        val_start = time.perf_counter()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_seconds = time.perf_counter() - val_start
        timing_info["val_seconds_history"].append(val_seconds)
        timing_info["train_seconds_history"].append(train_seconds)

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
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "best_val_loss": best_val_loss,
                        "val_acc_at_best_val_loss": val_acc_at_best_val_loss,
                        # Full reproducibility snapshot, all read from config.py.
                        "model_config": build_model_config(model_config_overrides),
                        "training_config": build_training_config(
                            loss=loss_name,
                            overrides=training_config_overrides,
                        ),
                        "preprocessing_config": build_preprocessing_config(
                            preprocessing_config_overrides
                        ),
                    },
                    checkpoint_path,
                )
                print(
                    f"[INFO] Saved new best model at epoch {best_epoch} "
                    f"(val_loss={best_val_loss:.4f}, "
                    f"val_acc={val_acc_at_best_val_loss:.4f}) "
                    f"-> {_display_path(checkpoint_path)}"
                )

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"train_seconds={train_seconds:.3f} | val_seconds={val_seconds:.3f}"
        )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_epoch = epoch + 1
            early_stopped = True
            print(
                f"[INFO] Early stopping at epoch {stopped_epoch}; "
                f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}"
            )
            break

    timing_info["total_training_seconds"] = time.perf_counter() - training_start
    if stopped_epoch is None:
        stopped_epoch = history["epoch"][-1] if history["epoch"] else 0

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\n[INFO] Loaded best model from epoch {best_epoch} "
            f"with val_loss={best_val_loss:.4f} "
            f"and val_acc={val_acc_at_best_val_loss:.4f}"
        )

    val_eval_start = time.perf_counter()
    val_loss, val_acc, val_confusion, val_per_class_accuracy = evaluate_with_details(
        model, val_loader, criterion, device, num_classes
    )
    timing_info["evaluation_seconds"]["val"] = time.perf_counter() - val_eval_start

    if test_loader is not None:
        test_eval_start = time.perf_counter()
        test_loss, test_acc, test_confusion, test_per_class_accuracy = evaluate_with_details(
            model, test_loader, criterion, device, num_classes
        )
        timing_info["evaluation_seconds"]["test"] = time.perf_counter() - test_eval_start
        test_confusion_matrix = test_confusion.tolist()
    else:
        test_loss = None
        test_acc = None
        test_confusion_matrix = None
        test_per_class_accuracy = None
        test_confusion = None

    class_labels = _get_class_labels(num_classes)
    val_f2_metrics = precision_recall_fbeta_from_confusion(
        val_confusion,
        class_labels,
        beta=2.0,
    )
    val_fault_summary = fault_vs_normal_summary(val_confusion, normal_class_idx=0)
    if test_confusion is not None:
        test_f2_metrics = precision_recall_fbeta_from_confusion(
            test_confusion,
            class_labels,
            beta=2.0,
        )
        test_fault_summary = fault_vs_normal_summary(test_confusion, normal_class_idx=0)
    else:
        test_f2_metrics = None
        test_fault_summary = None

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopped_epoch": int(stopped_epoch),
        "early_stopped": bool(early_stopped),
        "early_stopping_patience": early_stopping_patience,
        "val_acc_at_best_val_loss": val_acc_at_best_val_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_confusion_matrix": val_confusion.tolist(),
        "val_per_class_accuracy": val_per_class_accuracy,
        "val_f2_metrics": val_f2_metrics,
        "val_macro_f2": val_f2_metrics["macro"]["f2"],
        "val_weighted_f2": val_f2_metrics["weighted"]["f2"],
        "val_fault_vs_normal": val_fault_summary,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_confusion_matrix": test_confusion_matrix,
        "test_per_class_accuracy": test_per_class_accuracy,
        "test_f2_metrics": test_f2_metrics,
        "test_macro_f2": (
            test_f2_metrics["macro"]["f2"] if test_f2_metrics is not None else None
        ),
        "test_weighted_f2": (
            test_f2_metrics["weighted"]["f2"] if test_f2_metrics is not None else None
        ),
        "test_fault_vs_normal": test_fault_summary,
        "imbalance_info": imbalance_info,
        "history": history,
        "timing_info": timing_info,
    }

    return model, results


def train(
    *,
    data_dir=None,
    max_files=None,
    run_name: str = DEFAULT_RUN_NAME,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    normalization_scope: str = NORMALIZATION_SCOPE,
    normalization_method: str = NORMALIZATION_METHOD,
):
    """
    Entry point for full Transformer baseline training.
    """
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    paths = _create_run_dir(run_name)
    run_dir = paths["run_dir"]
    checkpoints_dir = paths["checkpoints_dir"]
    results_dir = paths["results_dir"]
    plots_dir = paths["plots_dir"]
    logs_dir = paths["logs_dir"]
    print(f"[INFO] run_dir={run_dir.relative_to(PROJECT_ROOT)}")

    dataloader_build_start = time.perf_counter()
    build_kwargs = {
        "max_files": max_files,
        "batch_size": batch_size,
        "window_size": window_size,
        "stride": stride,
        "normalization_scope": normalization_scope,
        "normalization_method": normalization_method,
    }
    if data_dir is not None:
        build_kwargs["data_dir"] = data_dir
    train_loader, val_loader, test_loader, stats = build_dataloaders(**build_kwargs)
    dataloader_build_seconds = time.perf_counter() - dataloader_build_start

    print(f"[INFO] build_dataloaders_seconds={dataloader_build_seconds:.3f}")
    print(f"[INFO] normalization_scope={normalization_scope}")
    print(f"[INFO] batch_size={batch_size}")

    if stats.get("max_files") is not None:
        print(
            f"[WARN] build_dataloaders is running with max_files={stats['max_files']}. "
            "This is a debug subset, not a full run."
        )

    print("\nData stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    model = TransformerBaseline(seq_len=window_size).to(device)

    best_model_path = checkpoints_dir / "transformer_best.pt"
    history_path = results_dir / "history.json"
    metrics_path = results_dir / "metrics.json"
    normalization_constants_path = results_dir / "normalization_constants.json"

    model_overrides = {"seq_len": int(window_size)}
    training_overrides = {
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
    }
    preprocessing_overrides = {
        "window_size": int(window_size),
        "window_stride": int(stride),
        "normalization": normalization_method,
        "normalization_scope": normalization_scope,
    }
    runtime_info = {
        "run_dir": str(run_dir),
        "cuda_available": bool(cuda_available),
        "gpu_name": gpu_name,
        "device": str(device),
    }

    _, results = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=num_epochs,
        num_classes=NUM_CLASSES,
        checkpoint_path=best_model_path,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        model_config_overrides=model_overrides,
        training_config_overrides=training_overrides,
        preprocessing_config_overrides=preprocessing_overrides,
    )

    _save_result_plots(results, NUM_CLASSES, plots_dir)

    # Reproducibility: export standalone config snapshots next to the
    # checkpoint/results. The same dicts are also embedded in the checkpoint.
    loss_name = (
        "weighted_cross_entropy"
        if results["imbalance_info"].get("class_weights_applied")
        else "cross_entropy"
    )
    model_config_path, experiment_config_path = save_experiment_config_files(
        loss=loss_name,
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        model_overrides=model_overrides,
        training_overrides=training_overrides,
        preprocessing_overrides=preprocessing_overrides,
        runtime_info=runtime_info,
    )
    print("[INFO] Saved model_config and experiment_config for reproducibility.")
    print(f"model_config_path={model_config_path.relative_to(PROJECT_ROOT)}")
    print(f"experiment_config_path={experiment_config_path.relative_to(PROJECT_ROOT)}")

    with open(history_path, "w") as f:
        json.dump(results["history"], f, indent=2)

    normalization_info = stats.get("normalization", {})
    with open(normalization_constants_path, "w") as f:
        json.dump(normalization_info, f, indent=2)

    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated()) if cuda_available else 0
    )
    peak_gpu_memory_gib = peak_gpu_memory_bytes / (1024 ** 3)
    resource_info = {
        "cuda_available": bool(cuda_available),
        "gpu_name": gpu_name,
        "device": str(device),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "peak_gpu_memory_gib": peak_gpu_memory_gib,
        "batch_size": int(batch_size),
        "normalization_scope": normalization_scope,
    }

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "data_stats": stats,
                "dataloader_build_seconds": dataloader_build_seconds,
                "resource_info": resource_info,
                "timing_info": results["timing_info"],
                "imbalance_info": results["imbalance_info"],
                "best_epoch": results["best_epoch"],
                "best_val_loss": results["best_val_loss"],
                "stopped_epoch": results["stopped_epoch"],
                "early_stopped": results["early_stopped"],
                "early_stopping_patience": results["early_stopping_patience"],
                "val_acc_at_best_val_loss": results["val_acc_at_best_val_loss"],
                "val_loss": results["val_loss"],
                "val_acc": results["val_acc"],
                "val_confusion_matrix": results["val_confusion_matrix"],
                "val_per_class_accuracy": results["val_per_class_accuracy"],
                "val_f2_metrics": results["val_f2_metrics"],
                "val_macro_f2": results["val_macro_f2"],
                "val_weighted_f2": results["val_weighted_f2"],
                "val_fault_vs_normal": results["val_fault_vs_normal"],
                "test_loss": results["test_loss"],
                "test_acc": results["test_acc"],
                "test_confusion_matrix": results["test_confusion_matrix"],
                "test_per_class_accuracy": results["test_per_class_accuracy"],
                "test_f2_metrics": results["test_f2_metrics"],
                "test_macro_f2": results["test_macro_f2"],
                "test_weighted_f2": results["test_weighted_f2"],
                "test_fault_vs_normal": results["test_fault_vs_normal"],
            },
            f,
            indent=2,
        )

    run_log_path = logs_dir / "run_summary.json"
    with open(run_log_path, "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "history_path": str(history_path),
                "metrics_path": str(metrics_path),
                "checkpoint_path": str(best_model_path),
                "normalization_constants_path": str(normalization_constants_path),
                "plots_dir": str(plots_dir),
                "resource_info": resource_info,
            },
            f,
            indent=2,
        )

    print("\nFinal evaluation results:")
    print(f"best_epoch={results['best_epoch']}")
    print(f"best_val_loss={results['best_val_loss']:.4f}")
    print(f"stopped_epoch={results['stopped_epoch']}")
    print(f"early_stopped={results['early_stopped']}")
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
    print(f"normalization_constants_path={normalization_constants_path.relative_to(PROJECT_ROOT)}")
    print(f"peak_gpu_memory_gib={peak_gpu_memory_gib:.4f}")

    print("\nClass imbalance handling:")
    print(json.dumps(results["imbalance_info"], indent=2))

    print("\nValidation confusion matrix:")
    print(np.array(results["val_confusion_matrix"]))

    print("\nTest confusion matrix:")
    print(np.array(results["test_confusion_matrix"]))

    print("\nTest safety metrics:")
    print(json.dumps(results["test_fault_vs_normal"], indent=2))
    print(f"test_macro_f2={results['test_macro_f2']}")
    print(f"test_weighted_f2={results['test_weighted_f2']}")

    print("\nValidation per-class accuracy:")
    for class_idx, acc in results["val_per_class_accuracy"].items():
        if acc is None:
            print(f"class_{class_idx}: no samples")
        else:
            print(f"class_{class_idx}: {acc:.4f}")

    if results["test_per_class_accuracy"] is not None:
        print("\nTest per-class accuracy:")
        for class_idx, acc in results["test_per_class_accuracy"].items():
            if acc is None:
                print(f"class_{class_idx}: no samples")
            else:
                print(f"class_{class_idx}: {acc:.4f}")

    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(best_model_path),
        "normalization_constants_path": str(normalization_constants_path),
    }


if __name__ == "__main__":
    train()
