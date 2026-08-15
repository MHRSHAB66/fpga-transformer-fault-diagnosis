"""
FPGA Fault Diagnosis -- FULL test-set int10 accuracy, shard 0/4.

CPU-only Kaggle kernel (int64 numpy reference is slow, so the full 95k-window
test is split into NUM_SHARDS strided shards, each well under the 12 h per-session
wall). Each shard calibrates identically (same 12 calib files, seed 42) and runs
the full-integer int10 reference on its 1/NUM_SHARDS slice, then writes RAW COUNTS
(overall + per-class correct/total) so the shards combine into the exact
full-test accuracy.

Attach: fpga-fault-src (code), mafaulda-full (data), fpga-fault-f1-ckpt (best.pt).
Both known-good fixes (DATA_RAW_DIR patch + state_dict key remap) are baked in.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SHARD       = 0
NUM_SHARDS  = 4
BITS        = 10

INPUT_ROOT  = "/kaggle/input"
WORKING_DIR = "/kaggle/working"

WINDOW_SIZE = 1024
SEED        = 42
IN_CHANNELS = 8
NUM_CLASSES = 4
BEST_CFG = {"d_model": 32, "nhead": 4, "num_layers": 1,
            "dim_feedforward": 64, "dropout": 0.1}
CALIB_FILES = 12


def detect_mafaulda_root(root):
    markers = ("normal", "imbalance", "horizontal-misalignment",
               "vertical-misalignment", "overhang", "underhang")
    best = None
    for dirpath, dirnames, _ in os.walk(root):
        lowered = {d.lower() for d in dirnames}
        hits = sum(1 for m in markers if m in lowered)
        if hits >= 3 and (best is None or hits > best[0]):
            best = (hits, dirpath)
    return best[1] if best else None


def _remap_key(k):
    k = k.replace("pos_enc.", "positional_encoding.")
    k = k.replace("input_proj.", "input_projection.")
    if k.startswith("encoder."):
        parts = k.split(".")
        l, rest = parts[1], ".".join(parts[2:])
        for a, b in (("attn", "self_attention"),
                     ("attn_norm", "attention_norm"),
                     ("ffn_norm", "feedforward_norm"),
                     ("ffn", "feedforward")):
            if rest == a or rest.startswith(a + "."):
                rest = b + rest[len(a):]
                break
        k = f"encoder_layers.{l}.{rest}"
    return k


def main():
    t_start = time.perf_counter()

    cfg_hits = glob.glob(os.path.join(INPUT_ROOT, "**", "src", "config.py"),
                         recursive=True)
    code_root = os.path.dirname(os.path.dirname(cfg_hits[0]))
    raw_dir = detect_mafaulda_root(INPUT_ROOT)
    ckpt_path = glob.glob(os.path.join(INPUT_ROOT, "**", "best.pt"), recursive=True)[0]
    print(f"shard {SHARD}/{NUM_SHARDS}  code={code_root}  raw={raw_dir}  ckpt={ckpt_path}",
          flush=True)

    os.environ["PROJECT_ROOT"] = WORKING_DIR
    sys.path.insert(0, code_root)

    import src.quantize.evaluate_integer as _ei
    import src.quantize.evaluate_quantized as _eq
    _ei.DATA_RAW_DIR = Path(raw_dir)
    _eq.DATA_RAW_DIR = Path(raw_dir)

    import src.quantize.integer_model as im
    from src.quantize.integer_model import IntegerTransformer
    from src.quantize.evaluate_integer_bits import set_bits, calibrate_bits
    from src.quantize.evaluate_integer import int_logits, build_train_subset
    from src.quantize.evaluate_quantized import build_test_subset

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_np = {_remap_key(k): v.cpu().numpy()
                for k, v in ckpt["model_state_dict"].items()}
    cfg = {"input_dim": IN_CHANNELS, "seq_len": WINDOW_SIZE,
           "num_classes": NUM_CLASSES, **BEST_CFG}

    # -- Full test set, then take this shard's strided slice -------------------
    Xc, _, kc = build_train_subset(CALIB_FILES, SEED)
    Xt, yt, sub = build_test_subset(10 ** 6, SEED)          # all test files
    full_n = len(yt)
    idx = np.arange(full_n)[SHARD::NUM_SHARDS]
    Xs, ys = Xt[idx].astype(np.float64), yt[idx].astype(np.int64)
    del Xt
    print(f"full_test_windows={full_n}  shard_windows={len(ys)}  "
          f"calib_windows={len(Xc)} ({kc} files)", flush=True)

    # -- int10 model (calibrate identically, then infer) -----------------------
    set_bits(BITS)
    model = IntegerTransformer(state_np, cfg)
    scales = calibrate_bits(model, Xc, BITS)
    model.set_scales(scales)

    t0 = time.perf_counter()
    il = int_logits(model, Xs)
    preds = il.argmax(1)
    dt = time.perf_counter() - t0

    correct = int((preds == ys).sum())
    per_class = {}
    for c in range(NUM_CLASSES):
        m = ys == c
        per_class[str(c)] = {"n": int(m.sum()),
                             "correct": int(((preds == c) & m).sum())}
    print(f"shard int10 done in {dt:.1f}s  acc_shard={correct/len(ys):.4f}", flush=True)

    out = {
        "stage":        "6f_fulltest_int10_shard",
        "winner":       "F_1layer",
        "bits":         BITS,
        "shard":        SHARD,
        "num_shards":   NUM_SHARDS,
        "full_test_windows": int(full_n),
        "shard_windows":     int(len(ys)),
        "shard_correct":     correct,
        "per_class":         per_class,
        "elapsed_min":       round((time.perf_counter() - t_start) / 60, 1),
    }
    out_path = Path(WORKING_DIR) / f"int10_shard_{SHARD}_of_{NUM_SHARDS}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"shard result -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
