"""
Kaggle smoke test for the FPGA fault-diagnosis Transformer pipeline.

PURPOSE
-------
Verify that the project runs correctly on Kaggle BEFORE any final training run.
This is a SMOKE TEST only:
  * a small subset of files
  * 1 epoch maximum
  * verifies imports, CUDA/GPU, DataLoader, model forward pass,
    a tiny training loop, evaluation, and artifact saving.

It does NOT change the model architecture, the split logic, or any
preprocessing decision. It reuses the real project code unchanged:
  build_dataloaders(...)  ->  TransformerBaseline()  ->  run_training(...)

Attached inputs expected:
  * dataset  vuxuancu/mafaulda-full        (the MAFAULDA CSV files)
  * dataset  mehrdadsheikhabbasi/fpga-fault-src  (the project `src/` package)

Outputs (under /kaggle/working):
  * smoke_report.json                      machine-readable result summary
  * artifacts/checkpoints/...              tiny checkpoint from the 1-epoch run
  * artifacts/results/...                  metrics/plots from the 1-epoch run
"""

import os
import sys
import glob
import json
import time
import traceback

# ---------------------------------------------------------------------------
# Smoke-test knobs (intentionally tiny; NOT a final training configuration)
# ---------------------------------------------------------------------------
SMOKE_MAX_FILES = 24     # small subset of raw files (kept modest for CPU fallback)
SMOKE_NUM_EPOCHS = 1     # 1 epoch maximum
WORKING_DIR = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"

report = {
    "passed": False,
    "stage": "init",
    "max_files": SMOKE_MAX_FILES,
    "num_epochs": SMOKE_NUM_EPOCHS,
    "errors": [],
}


def log(msg):
    print(msg, flush=True)


def fail(stage, exc):
    report["stage"] = stage
    report["errors"].append(f"{stage}: {type(exc).__name__}: {exc}")
    log(f"\n[SMOKE][FAIL] stage={stage}: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    _save_report()
    sys.exit(1)


def _save_report():
    try:
        os.makedirs(WORKING_DIR, exist_ok=True)
        with open(os.path.join(WORKING_DIR, "smoke_report.json"), "w") as f:
            json.dump(report, f, indent=2)
    except Exception as exc:  # pragma: no cover - best effort
        log(f"[SMOKE][WARN] could not write smoke_report.json: {exc}")


# ---------------------------------------------------------------------------
# 1) Locate the project source package and put PROJECT_ROOT in a writable place
# ---------------------------------------------------------------------------
try:
    report["stage"] = "locate_source"

    # Diagnostic: show what actually mounted under /kaggle/input.
    top_level = sorted(glob.glob(os.path.join(INPUT_ROOT, "*")))
    log(f"[SMOKE] /kaggle/input entries = {[os.path.basename(p) for p in top_level]}")

    # Robustly find the project source: any '<...>/src/config.py' under the
    # input mounts. `code_root` is the directory that should go on sys.path
    # (the parent of the `src` package).
    code_root = None
    config_hits = glob.glob(os.path.join(INPUT_ROOT, "**", "src", "config.py"),
                            recursive=True)
    log(f"[SMOKE] src/config.py candidates = {config_hits}")
    if config_hits:
        # parent of 'src' (e.g. /kaggle/input/fpga-fault-src)
        code_root = os.path.dirname(os.path.dirname(config_hits[0]))

    if code_root is None:
        raise FileNotFoundError(
            "Could not find a 'src/config.py' under /kaggle/input. "
            f"Top-level mounts seen: {[os.path.basename(p) for p in top_level]}. "
            "Attach the fpga-fault-src dataset to the kernel."
        )
    sys.path.insert(0, code_root)

    # PROJECT_ROOT controls where caches/artifacts are written. The dataset
    # mount is read-only, so point it at the writable working directory.
    # Must be set BEFORE importing src.config.
    os.environ["PROJECT_ROOT"] = WORKING_DIR
    report["code_root"] = code_root
    report["project_root"] = WORKING_DIR
    log(f"[SMOKE] code_root = {code_root}")
    log(f"[SMOKE] PROJECT_ROOT = {WORKING_DIR}")
except Exception as exc:
    fail("locate_source", exc)


# ---------------------------------------------------------------------------
# 2) Verify imports (project + scientific stack) and CUDA/GPU
# ---------------------------------------------------------------------------
try:
    report["stage"] = "imports"
    import numpy as np
    import pandas as pd
    import torch

    from src import config as cfg
    from src.data.make_dataloaders import build_dataloaders
    from src.models.transformer_baseline import TransformerBaseline
    from src.train_transformer_baseline import run_training, set_seed

    log(f"[SMOKE] numpy={np.__version__} pandas={pd.__version__} torch={torch.__version__}")
    log(f"[SMOKE] WINDOW_SIZE={cfg.WINDOW_SIZE} INPUT_CHANNELS={cfg.INPUT_CHANNELS} "
        f"NUM_CLASSES={cfg.NUM_CLASSES} CHANNEL_SELECTION={cfg.CHANNEL_SELECTION}")

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    report["cuda_available"] = bool(cuda_available)
    report["gpu_name"] = gpu_name
    log(f"[SMOKE] CUDA available = {cuda_available}")
    log(f"[SMOKE] GPU name = {gpu_name}")

    # Probe whether the assigned GPU can ACTUALLY run a CUDA kernel. Kaggle's
    # `enable_gpu` flag may grant a Tesla P100 (compute capability sm_60), which
    # is NOT supported by the environment's PyTorch build (sm_70+). In that case
    # torch.cuda.is_available() is True but the first real kernel raises
    # "no kernel image is available for execution on the device". Detect that
    # here and fall back to CPU so the rest of the pipeline can still be verified.
    gpu_usable = False
    gpu_incompatible_reason = None
    if cuda_available:
        try:
            _probe = torch.ones(8, 8, device="cuda")
            _probe = (_probe @ _probe).sum().item()
            torch.cuda.synchronize()
            gpu_usable = True
        except Exception as gpu_exc:  # incompatible GPU / driver mismatch
            gpu_incompatible_reason = f"{type(gpu_exc).__name__}: {gpu_exc}"
            try:
                report["torch_supported_arch_list"] = torch.cuda.get_arch_list()
            except Exception:
                pass

    report["gpu_usable"] = bool(gpu_usable)
    report["gpu_incompatible_reason"] = gpu_incompatible_reason

    SMOKE_DEVICE = torch.device("cuda" if gpu_usable else "cpu")
    report["device"] = str(SMOKE_DEVICE)
    if gpu_usable:
        log(f"[SMOKE] GPU compute probe OK -> using {SMOKE_DEVICE}")
    else:
        log("[SMOKE][WARN] GPU present but NOT usable with the current PyTorch "
            f"build. Reason: {gpu_incompatible_reason}")
        log(f"[SMOKE][WARN] Falling back to CPU for compute (device={SMOKE_DEVICE}).")
except Exception as exc:
    fail("imports", exc)


# ---------------------------------------------------------------------------
# 3) Auto-detect the MAFAULDA raw directory
# ---------------------------------------------------------------------------
def detect_mafaulda_root(input_root):
    """Find the directory that holds the MAFAULDA class subfolders."""
    class_markers = ("normal", "imbalance", "horizontal-misalignment",
                     "vertical-misalignment", "overhang", "underhang")
    candidates = []
    for dirpath, dirnames, _ in os.walk(input_root):
        lowered = {d.lower() for d in dirnames}
        hits = sum(1 for m in class_markers if m in lowered)
        if hits >= 3:
            candidates.append((hits, dirpath))
    if not candidates:
        return None
    # Prefer the directory that matches the most class markers.
    candidates.sort(key=lambda t: (-t[0], len(t[1])))
    return candidates[0][1]


try:
    report["stage"] = "detect_dataset"
    raw_dir = detect_mafaulda_root(INPUT_ROOT)
    if raw_dir is None:
        raise FileNotFoundError(
            "Could not auto-detect the MAFAULDA root under /kaggle/input. "
            "Attach the vuxuancu/mafaulda-full dataset."
        )
    report["dataset_path"] = raw_dir
    class_dirs = sorted(
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    )
    report["class_dirs"] = class_dirs
    log(f"[SMOKE] dataset_path = {raw_dir}")
    log(f"[SMOKE] class_dirs = {class_dirs}")
except Exception as exc:
    fail("detect_dataset", exc)


# ---------------------------------------------------------------------------
# 4) Verify dataset structure: columns and approximate file length / rate
# ---------------------------------------------------------------------------
try:
    report["stage"] = "verify_structure"
    sample_csvs = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    report["total_csv_files"] = len(sample_csvs)
    if not sample_csvs:
        raise FileNotFoundError("No CSV files found under the detected dataset root.")

    sample_path = sample_csvs[0]
    sample_df = pd.read_csv(sample_path, header=None)
    n_rows, n_cols = int(sample_df.shape[0]), int(sample_df.shape[1])
    duration_s = n_rows / float(cfg.MAFAULDA_SAMPLE_RATE)

    report["sample_file"] = os.path.relpath(sample_path, INPUT_ROOT)
    report["sample_rows"] = n_rows
    report["sample_cols"] = n_cols
    report["assumed_sample_rate_hz"] = int(cfg.MAFAULDA_SAMPLE_RATE)
    report["approx_duration_seconds"] = round(duration_s, 3)

    # ~250,000 samples at 50 kHz => ~5 s recordings.
    near_250k = abs(n_rows - 250_000) <= 25_000   # within +/-10%
    report["approx_250k_at_50khz"] = bool(near_250k)
    report["columns_match_expected"] = bool(n_cols == cfg.EXPECTED_NUM_COLUMNS)

    log(f"[SMOKE] total_csv_files = {len(sample_csvs)}")
    log(f"[SMOKE] sample file rows={n_rows} cols={n_cols} "
        f"(~{duration_s:.2f}s at {cfg.MAFAULDA_SAMPLE_RATE} Hz)")
    log(f"[SMOKE] columns_match_expected({cfg.EXPECTED_NUM_COLUMNS}) = "
        f"{report['columns_match_expected']}")
    log(f"[SMOKE] approx_250k_at_50khz = {report['approx_250k_at_50khz']}")
except Exception as exc:
    fail("verify_structure", exc)


# ---------------------------------------------------------------------------
# 5) Build DataLoaders on a small subset (real project pipeline, unchanged)
# ---------------------------------------------------------------------------
try:
    report["stage"] = "build_dataloaders"
    set_seed(cfg.RANDOM_SEED)
    device = SMOKE_DEVICE  # chosen by the GPU compute probe above

    t0 = time.perf_counter()
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir=raw_dir,
        max_files=SMOKE_MAX_FILES,
    )
    report["build_dataloaders_seconds"] = round(time.perf_counter() - t0, 3)
    report["data_stats"] = {
        "total_files_after_filter": stats["total_files_after_filter"],
        "train_files_kept": stats["train_files_kept"],
        "val_files_kept": stats["val_files_kept"],
        "test_files_kept": stats["test_files_kept"],
        "train_samples": stats["train_samples"],
        "val_samples": stats["val_samples"],
        "test_samples": stats["test_samples"],
        "window_shape": stats["window_shape"],
        "train_class_window_counts": stats["train_class_window_counts"],
    }
    log(f"[SMOKE] dataloaders built in {report['build_dataloaders_seconds']}s")
    log(f"[SMOKE] train/val/test windows = "
        f"{stats['train_samples']}/{stats['val_samples']}/{stats['test_samples']}")
    log(f"[SMOKE] window_shape = {stats['window_shape']}")
except Exception as exc:
    fail("build_dataloaders", exc)


# ---------------------------------------------------------------------------
# 6) Model forward pass on one batch
# ---------------------------------------------------------------------------
try:
    report["stage"] = "forward_pass"
    model = TransformerBaseline().to(device)
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device)
    with torch.no_grad():
        logits = model(x_batch)
    report["forward_input_shape"] = list(x_batch.shape)
    report["forward_output_shape"] = list(logits.shape)
    report["num_model_params"] = int(sum(p.numel() for p in model.parameters()))
    log(f"[SMOKE] forward input {list(x_batch.shape)} -> output {list(logits.shape)}")
    log(f"[SMOKE] model params = {report['num_model_params']:,}")
    if logits.shape[1] != cfg.NUM_CLASSES:
        raise ValueError(
            f"Model output dim {logits.shape[1]} != NUM_CLASSES {cfg.NUM_CLASSES}"
        )
except Exception as exc:
    fail("forward_pass", exc)


# ---------------------------------------------------------------------------
# 7) Tiny training loop (1 epoch) + evaluation + artifact saving
#    Reuses the real run_training(); checkpoints/metrics land in /kaggle/working.
# ---------------------------------------------------------------------------
try:
    report["stage"] = "tiny_training"
    ckpt_dir = os.path.join(WORKING_DIR, "artifacts", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    from pathlib import Path
    ckpt_path = Path(ckpt_dir) / "transformer_smoke_test.pt"

    fresh_model = TransformerBaseline().to(device)
    t0 = time.perf_counter()
    _, results = run_training(
        model=fresh_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=cfg.LEARNING_RATE,
        num_epochs=SMOKE_NUM_EPOCHS,
        num_classes=cfg.NUM_CLASSES,
        checkpoint_path=ckpt_path,
    )
    report["tiny_training_seconds"] = round(time.perf_counter() - t0, 3)
    report["train_result"] = {
        "best_epoch": results["best_epoch"],
        "best_val_loss": results["best_val_loss"],
        "val_acc": results["val_acc"],
        "test_acc": results["test_acc"],
    }
    report["checkpoint_saved"] = bool(ckpt_path.exists())
    report["checkpoint_path"] = str(ckpt_path)

    # Save a metrics artifact for completeness of the "artifact saving" check.
    results_dir = os.path.join(WORKING_DIR, "artifacts", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "smoke_metrics.json"), "w") as f:
        json.dump({
            "best_epoch": results["best_epoch"],
            "best_val_loss": results["best_val_loss"],
            "val_acc": results["val_acc"],
            "test_acc": results["test_acc"],
            "val_confusion_matrix": results["val_confusion_matrix"],
            "test_confusion_matrix": results["test_confusion_matrix"],
        }, f, indent=2)

    log(f"[SMOKE] 1-epoch training done in {report['tiny_training_seconds']}s")
    log(f"[SMOKE] val_acc={results['val_acc']:.4f} test_acc={results['test_acc']:.4f}")
    log(f"[SMOKE] checkpoint_saved = {report['checkpoint_saved']} -> {ckpt_path}")
except Exception as exc:
    fail("tiny_training", exc)


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
report["passed"] = True
report["stage"] = "complete"
_save_report()
log("\n[SMOKE][PASS] All smoke-test stages completed successfully.")
log("[SMOKE] Report:")
log(json.dumps(report, indent=2))
