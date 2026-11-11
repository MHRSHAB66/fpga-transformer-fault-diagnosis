"""
Stage 6f — Calibrate and evaluate the full-integer Transformer reference.

Pipeline:
1. CALIBRATE activation scales on a subset of the TRAIN split (never test).
2. CROSS-CHECK the module's float path against the real PyTorch model.
3. EVALUATE the integer-only path on a subset of the TEST split; report accuracy
   drop vs float and prediction agreement.
4. SAVE scales to artifacts/quantized/integer_config.json and the accuracy
   report to artifacts/quantized/integer_accuracy.json.

Run
---
    python -m src.quantize.evaluate_integer --calib-files 40 --test-files 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import (
    PROJECT_ROOT, DATA_RAW_DIR, NUM_CLASSES, LABEL_MAP, RANDOM_SEED,
)
from src.data.load_data import build_file_index
from src.data.make_dataloaders import split_file_index, build_X_y_from_file_index
from src.models.transformer_baseline import TransformerBaseline
from src.quantize.integer_model import IntegerTransformer, CalibStats

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "20260615_223905_fpga_fault_final_t4"
    / "checkpoints"
    / "transformer_best.pt"
)


def load_model(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt:
        raise ValueError("Checkpoint has no embedded model_config.")
    cfg = ckpt["model_config"]
    model = TransformerBaseline(
        input_dim=cfg["input_dim"], seq_len=cfg["seq_len"],
        num_classes=cfg["num_classes"], d_model=cfg["d_model"],
        nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    float_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, float_state, cfg


def build_test_subset(num_files, seed):
    index_df = build_file_index(DATA_RAW_DIR)
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    _, _, test_index_df = split_file_index(index_df)
    total_test_files = len(test_index_df)
    if num_files is not None and num_files < total_test_files:
        test_index_df = test_index_df.sample(n=num_files, random_state=seed).reset_index(drop=True)
    X, y, kept, dropped, _ = build_X_y_from_file_index(test_index_df)
    return X, y, {
        "total_test_files": int(total_test_files),
        "test_files_used": int(kept),
        "test_files_dropped": int(dropped),
        "windows": int(len(y)),
    }

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "quantized"
CONFIG_PATH = OUTPUT_DIR / "integer_config.json"
REPORT_PATH = OUTPUT_DIR / "integer_accuracy.json"
LABEL_NAMES_BY_ID = {v: k for k, v in LABEL_MAP.items()}


def build_train_subset(num_files, seed):
    """Windows from a subset of the real TRAIN split, for calibration only."""
    index_df = build_file_index(DATA_RAW_DIR)
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    train_index_df, _, _ = split_file_index(index_df)
    if num_files is not None and num_files < len(train_index_df):
        train_index_df = train_index_df.sample(n=num_files, random_state=seed).reset_index(drop=True)
    X, y, kept, _, _ = build_X_y_from_file_index(train_index_df)
    return X, y, int(kept)


def calibrate(model: IntegerTransformer, X: np.ndarray, batch: int = 8) -> dict:
    stats = CalibStats.empty()
    for s in range(0, len(X), batch):
        model.collect(X[s:s + batch].astype(np.float64), stats)
    return stats.to_scales()


def int_logits(model: IntegerTransformer, X: np.ndarray, batch: int = 8) -> np.ndarray:
    out = np.empty((len(X), NUM_CLASSES), dtype=np.int64)
    for s in range(0, len(X), batch):
        out[s:s + batch] = model.infer_int(X[s:s + batch].astype(np.float64))
    return out


def float_logits(model: IntegerTransformer, X: np.ndarray, batch: int = 8) -> np.ndarray:
    out = np.empty((len(X), NUM_CLASSES), dtype=np.float64)
    for s in range(0, len(X), batch):
        out[s:s + batch] = model.float_logits(X[s:s + batch].astype(np.float64))
    return out


def accuracy(preds, y):
    overall = float((preds == y).mean())
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = y == c
        per_class[LABEL_NAMES_BY_ID[c]] = float((preds[mask] == c).mean()) if mask.any() else None
    return overall, per_class


def main():
    ap = argparse.ArgumentParser(description="Stage 6f full-integer eval.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--calib-files", type=int, default=40)
    ap.add_argument("--test-files", type=int, default=60)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    device = torch.device("cpu")
    ref_model, float_state, model_cfg = load_model(args.checkpoint, device)
    state_np = {k: v.cpu().numpy() for k, v in float_state.items()}
    model = IntegerTransformer(state_np, model_cfg)

    # --- 1. calibrate on TRAIN subset --------------------------------------
    print(f"[calib] building train subset ({args.calib_files} files)...")
    t0 = time.perf_counter()
    Xc, _, kc = build_train_subset(args.calib_files, args.seed)
    print(f"[calib] {len(Xc)} windows from {kc} files ({time.perf_counter()-t0:.1f}s)")
    scales = calibrate(model, Xc)
    model.set_scales(scales)
    print(f"[calib] {len(scales)} activation scales computed")

    # --- 2. float cross-check vs PyTorch -----------------------------------
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((3, model_cfg["seq_len"], model_cfg["input_dim"]))
    with torch.no_grad():
        ref = ref_model(torch.from_numpy(probe).float()).cpu().numpy()
    mine = model.float_logits(probe)
    max_diff = float(np.abs(ref - mine).max())
    argmatch = bool((ref.argmax(1) == mine.argmax(1)).all())
    print(f"[check] float path vs PyTorch: max|diff|={max_diff:.3e}, argmax match={argmatch}")

    # --- 3. evaluate on TEST subset ----------------------------------------
    print(f"[eval] building test subset ({args.test_files} files)...")
    Xt, yt, sub = build_test_subset(args.test_files, args.seed)
    print(f"[eval] {sub['windows']} windows from {sub['test_files_used']} files")

    fl = float_logits(model, Xt)
    float_preds = fl.argmax(1)
    float_acc, float_pc = accuracy(float_preds, yt)

    t1 = time.perf_counter()
    il = int_logits(model, Xt)
    int_preds = il.argmax(1)
    int_acc, int_pc = accuracy(int_preds, yt)
    agree = float((int_preds == float_preds).mean())
    print(f"\n[float ]  acc = {float_acc:.4f}")
    print(f"[int16 ]  acc = {int_acc:.4f}   drop = {float_acc-int_acc:+.4f}   "
          f"agreement = {agree:.4f}   ({time.perf_counter()-t1:.1f}s)")

    # --- 4. save -----------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "6f_full_integer",
            "act_bits": 16, "weight_bits": 16,
            "calibration": {"split": "train", "files": kc, "windows": int(len(Xc))},
            "scales": scales,
        }, f, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "6f_full_integer",
            "scheme": "integer-only: int16 act/weight, int softmax/LayerNorm/GELU via LUT",
            "checkpoint": str(args.checkpoint),
            "float_crosscheck": {"max_abs_logit_diff": max_diff, "argmax_match": argmatch},
            "test_subset": sub,
            "float32": {"accuracy": float_acc, "per_class_accuracy": float_pc},
            "integer": {
                "accuracy": int_acc,
                "accuracy_drop_vs_float": float_acc - int_acc,
                "prediction_agreement_vs_float": agree,
                "per_class_accuracy": int_pc,
            },
        }, f, indent=2)
    print(f"\n[info] scales -> {CONFIG_PATH}")
    print(f"[info] report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
