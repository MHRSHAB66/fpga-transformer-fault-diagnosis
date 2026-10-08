"""Export the FULL MAFAULDA test split as INT10 ingress codes + labels for a
real-FPGA accuracy run, using the EXACT SEQ=256 INT10 ingress pipeline (same
split / windowing / quantization as the golden vectors and the shard evaluator).

Nothing about the split, class map, windowing, channel order, aggregation, or
quantization is changed here -- this only *reads* the test set and serializes it
in the golden_in.txt format the board driver already consumes.

Outputs (new versioned dir, nothing overwritten):
    artifacts/results/fpga_fulltest_<ts>/
        fpga_testset_in.txt       # "N 1024 8" then N lines of 8192 INT10 codes
        labels.txt                # N true labels (0..3)
        software_int10_pred.txt   # N NumPy/INT10 predictions (hardware oracle)
        meta.json                 # input_scale, N, software accuracy + confusion
"""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "experiments" / "transformer_f1_seq256_pool4"
    / "kaggle_v3" / "seq256_pool4" / "scratch" / "best.pt"
)
CFG = {
    "input_dim": 8, "seq_len": 256, "num_classes": 4,
    "d_model": 32, "nhead": 4, "num_layers": 1,
    "dim_feedforward": 64, "dropout": 0.1,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: artifacts/results/fpga_fulltest_<ts>)")
    ap.add_argument("--limit", type=int, default=0,
                    help="optional: keep only the first N windows (0 = all)")
    ap.add_argument("--oracle", action="store_true",
                    help="also compute the (slow, single-thread) software INT10 "
                         "predictions as a bit-exact cross-check. Off by default: "
                         "the FPGA run is scored against true labels, and the "
                         "software INT10 accuracy is already known from Kaggle.")
    args = ap.parse_args()

    out = args.out or (PROJECT_ROOT / "artifacts" / "results"
                       / ("fpga_fulltest_" + time.strftime("%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)

    # --- exact pipeline (mirrors evaluate_seq256_ingress_int10_shard.py) -------
    index_df = build_file_index(Path(DATA_RAW_DIR))
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    train_df, _, test_df = split_file_index(index_df)

    calib_df = train_df.sample(n=12, random_state=42).reset_index(drop=True)
    Xc, _, _, _, _ = build_X_y_from_file_index(
        calib_df, window_size=1024, stride=1024, aggregation_factor=4)

    X1024, ys, test_kept, test_dropped, _ = build_X_y_from_file_index(
        test_df, window_size=1024, stride=1024, aggregation_factor=1)
    X1024 = X1024.astype(np.float64)
    ys = ys.astype(np.int64)
    if args.limit:
        X1024, ys = X1024[:args.limit], ys[:args.limit]

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_np = {k: v.cpu().numpy() for k, v in ckpt["model_state_dict"].items()}
    set_bits(10)
    model = IntegerTransformer(state_np, CFG)
    scales = calibrate_bits(model, Xc, 10)
    model.set_scales(scales)
    input_scale = float(scales["input"])
    qmax = qint_max(10)

    # INT10 ingress codes the hardware consumes (the IP aggregates 4:1 itself).
    q1024 = np.clip(np.round(X1024 / input_scale), -qmax, qmax).astype(np.int64)
    n, seq, ch = q1024.shape
    assert (seq, ch) == (1024, 8), (seq, ch)

    # --- serialize FPGA inputs FIRST (golden_in.txt format) -------------------
    # These are the only artifacts the board run needs; write them before any
    # slow optional oracle so the hardware run is never blocked behind it.
    flat = q1024.reshape(n, seq * ch)
    with open(out / "fpga_testset_in.txt", "w", newline="\n") as f:
        f.write("%d %d %d\n" % (n, seq, ch))
        for row in flat:
            f.write(" ".join(map(str, row.tolist())) + "\n")
    np.savetxt(out / "labels.txt", ys, fmt="%d")

    meta = {
        "n_windows": int(n), "seq": seq, "ch": ch,
        "input_scale": input_scale,
        "test_files_kept": int(test_kept), "test_files_dropped": int(test_dropped),
        "class_map": {v: k for k, v in LABEL_MAP.items()},
        "note": "codes are row-major [sample, channel]; hardware aggregates 1024->256 internally",
    }

    # --- optional software INT10 oracle (slow, single-thread) -----------------
    if args.oracle:
        q256 = aggregate_round_div(q1024, factor=4)
        sw_pred = int_logits(model, q256.astype(np.float64) * input_scale).argmax(1).astype(np.int64)
        sw_correct = int((sw_pred == ys).sum())
        conf = np.zeros((4, 4), dtype=np.int64)
        for t, p in zip(ys, sw_pred):
            conf[int(t), int(p)] += 1
        np.savetxt(out / "software_int10_pred.txt", sw_pred, fmt="%d")
        meta.update(software_int10_accuracy=sw_correct / n if n else 0.0,
                    software_int10_correct=sw_correct,
                    software_confusion_matrix=conf.tolist())

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out.relative_to(PROJECT_ROOT)),
          "n_windows": int(n), "input_scale": input_scale,
          "oracle": bool(args.oracle)}, indent=2))


if __name__ == "__main__":
    main()
