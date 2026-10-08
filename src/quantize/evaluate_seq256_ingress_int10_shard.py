"""Evaluate one local shard of the exact 1024-code INT10 ingress contract."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import DATA_RAW_DIR, LABEL_MAP
from src.data.load_data import build_file_index
from src.data.make_dataloaders import split_file_index, build_X_y_from_file_index
from src.data.token_aggregation import aggregate_round_div
from src.quantize.evaluate_integer import int_logits
from src.quantize.evaluate_integer_bits import calibrate_bits, set_bits
from src.quantize.integer_model import IntegerTransformer
from src.quantize.integer_ops import qint_max

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "experiments" / "transformer_f1_seq256_pool4"
    / "kaggle_v3" / "seq256_pool4" / "scratch" / "best.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")

    started = time.perf_counter()
    index_df = build_file_index(Path(DATA_RAW_DIR))
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    train_df, _, test_df = split_file_index(index_df)
    calib_df = train_df.sample(n=12, random_state=42).reset_index(drop=True)
    Xc, _, calib_kept, calib_dropped, _ = build_X_y_from_file_index(
        calib_df, window_size=1024, stride=1024, aggregation_factor=4
    )
    # Shard by source file before materializing windows. Building the complete
    # 1024-sample test tensor in every worker needlessly multiplies RAM use.
    shard_df = test_df.iloc[args.shard::args.num_shards].reset_index(drop=True)
    Xs1024, ys, test_kept, test_dropped, _ = build_X_y_from_file_index(
        shard_df, window_size=1024, stride=1024, aggregation_factor=1
    )
    Xs1024 = Xs1024.astype(np.float64)
    ys = ys.astype(np.int64)

    checkpoint_data = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_np = {
        key: value.cpu().numpy()
        for key, value in checkpoint_data["model_state_dict"].items()
    }
    cfg = {
        "input_dim": 8,
        "seq_len": 256,
        "num_classes": 4,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.1,
    }
    set_bits(10)
    model = IntegerTransformer(state_np, cfg)
    scales = calibrate_bits(model, Xc, 10)
    model.set_scales(scales)

    input_scale = scales["input"]
    qmax = qint_max(10)
    q1024 = np.clip(np.round(Xs1024 / input_scale), -qmax, qmax).astype(np.int64)
    q256 = aggregate_round_div(q1024, factor=4)
    Xs_ingress = q256.astype(np.float64) * input_scale
    del Xs1024, q1024, q256

    infer_started = time.perf_counter()
    predictions = int_logits(model, Xs_ingress).argmax(1)
    inference_seconds = time.perf_counter() - infer_started
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, prediction in zip(ys, predictions):
        confusion[int(truth), int(prediction)] += 1

    result = {
        "status": "complete",
        "stage": "seq256_ingress1024_int10_fulltest_shard",
        "numeric_order": "normalize -> quantize INT10 -> integer aggregate 4:1 -> transformer",
        "bits": 10,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "test_files_assigned": int(len(shard_df)),
        "shard_windows": int(len(ys)),
        "shard_correct": int(np.trace(confusion)),
        "confusion_matrix": confusion.tolist(),
        "calibration": {
            "split": "train",
            "seed": 42,
            "files_requested": 12,
            "files_kept": int(calib_kept),
            "files_dropped": int(calib_dropped),
            "windows": int(len(Xc)),
        },
        "test_files_kept": int(test_kept),
        "test_files_dropped": int(test_dropped),
        "integer_inference_seconds": inference_seconds,
        "total_elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(CHECKPOINT),
        "model_seq_len": 256,
        "source_seq_len": 1024,
        "aggregation_factor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
