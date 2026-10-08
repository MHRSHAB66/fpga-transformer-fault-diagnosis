"""Full-test int10 evaluation for the SEQ=256 pool4 winner, one strided shard."""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SHARD = {{SHARD}}
NUM_SHARDS = {{NUM_SHARDS}}
BITS = 10
CALIB_FILES = 12
SEED = 42
INPUT_ROOT = "/kaggle/input"
WORKING_DIR = "/kaggle/working"


def detect_mafaulda_root(root):
    markers = {"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment"}
    candidates = []
    for dirpath, dirnames, _ in os.walk(root):
        score = len(markers.intersection(d.lower() for d in dirnames))
        if score >= 3:
            candidates.append((score, len(dirpath), dirpath))
    if not candidates:
        raise FileNotFoundError("MAFAULDA root not found")
    return sorted(candidates, key=lambda x: (-x[0], x[1]))[0][2]


def main():
    started = time.perf_counter()
    src_hits = glob.glob(f"{INPUT_ROOT}/**/src/config.py", recursive=True)
    if not src_hits:
        raise FileNotFoundError("fpga-fault-src not attached")
    code_root = str(Path(src_hits[0]).parents[1])
    raw_dir = detect_mafaulda_root(INPUT_ROOT)
    ckpt_hits = [p for p in glob.glob(f"{INPUT_ROOT}/**/best.pt", recursive=True)
                 if "seq256" in p.lower() or "pool4" in p.lower()]
    if not ckpt_hits:
        raise FileNotFoundError("SEQ256 checkpoint not attached")
    checkpoint = ckpt_hits[0]

    os.environ["PROJECT_ROOT"] = WORKING_DIR
    sys.path.insert(0, code_root)

    import src.quantize.integer_model as integer_module
    from src.config import LABEL_MAP
    from src.data.load_data import build_file_index
    from src.data.make_dataloaders import split_file_index, build_X_y_from_file_index
    from src.quantize.integer_model import IntegerTransformer
    from src.quantize.evaluate_integer_bits import set_bits, calibrate_bits
    from src.quantize.evaluate_integer import int_logits

    index_df = build_file_index(Path(raw_dir))
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    train_df, _, test_df = split_file_index(index_df)
    calib_df = train_df.sample(n=CALIB_FILES, random_state=SEED).reset_index(drop=True)
    Xc, _, calib_kept, calib_dropped, _ = build_X_y_from_file_index(
        calib_df, window_size=1024, stride=1024, aggregation_factor=4)
    Xt, yt, test_kept, test_dropped, _ = build_X_y_from_file_index(
        test_df, window_size=1024, stride=1024, aggregation_factor=4)

    full_count = len(yt)
    indices = np.arange(full_count)[SHARD::NUM_SHARDS]
    Xs = Xt[indices].astype(np.float64)
    ys = yt[indices].astype(np.int64)
    del Xt, yt

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_np = {k: v.cpu().numpy() for k, v in checkpoint_data["model_state_dict"].items()}
    cfg = {
        "input_dim": 8, "seq_len": 256, "num_classes": 4,
        "d_model": 32, "nhead": 4, "num_layers": 1,
        "dim_feedforward": 64, "dropout": 0.1,
    }
    set_bits(BITS)
    model = IntegerTransformer(state_np, cfg)
    scales = calibrate_bits(model, Xc, BITS)
    model.set_scales(scales)

    infer_started = time.perf_counter()
    logits = int_logits(model, Xs)
    predictions = logits.argmax(1)
    infer_seconds = time.perf_counter() - infer_started
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, pred in zip(ys, predictions):
        confusion[int(truth), int(pred)] += 1

    result = {
        "status": "complete",
        "stage": "seq256_pool4_fulltest_int10_shard",
        "bits": BITS,
        "shard": SHARD,
        "num_shards": NUM_SHARDS,
        "full_test_windows": int(full_count),
        "shard_windows": int(len(ys)),
        "shard_correct": int(np.trace(confusion)),
        "confusion_matrix": confusion.tolist(),
        "calibration": {
            "split": "train", "seed": SEED, "files_requested": CALIB_FILES,
            "files_kept": int(calib_kept), "files_dropped": int(calib_dropped),
            "windows": int(len(Xc)),
        },
        "test_files_kept": int(test_kept),
        "test_files_dropped": int(test_dropped),
        "integer_inference_seconds": infer_seconds,
        "total_elapsed_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint,
        "model_seq_len": 256,
        "aggregation_factor": 4,
    }
    output = Path(WORKING_DIR) / f"seq256_int10_shard_{SHARD}_of_{NUM_SHARDS}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
