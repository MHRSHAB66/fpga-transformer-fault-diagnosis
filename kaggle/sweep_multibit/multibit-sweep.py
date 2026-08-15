"""
FPGA Fault Diagnosis -- multi-bit integer sweep on the F_1layer winner.

Standalone Kaggle kernel: loads the pre-trained F_1layer checkpoint (no
retraining), then runs the full-integer numpy reference at several bit-widths
[16, 14, 12, 10, 8] on the SAME calib/test subset. HLS (ap_int<N>) supports any
width, so this finds the smallest width that stays near-lossless.

Attach as data sources:
  - mehrdadsheikhabbasi/fpga-fault-src        (code: src/...)
  - vuxuancu/mafaulda-full                    (raw MAFAULDA CSVs)
  - mehrdadsheikhabbasi/fpga-fault-f1-ckpt    (best.pt)

Both known-good fixes are baked in:
  - patch evaluate_integer/evaluate_quantized DATA_RAW_DIR to the detected raw
    dir (Kaggle data lives under /kaggle/input, not PROJECT_ROOT/data/raw);
  - remap TuningTransformer state_dict keys 1:1 to the names IntegerTransformer
    expects (pos_enc -> positional_encoding, input_proj -> input_projection,
    encoder.N.attn/ffn... -> encoder_layers.N.self_attention/feedforward...).
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

INPUT_ROOT  = "/kaggle/input"
WORKING_DIR = "/kaggle/working"

# -- Fixed constants (must match the training kernel) --------------------------
WINDOW_SIZE = 1024
SEED        = 42
IN_CHANNELS = 8
NUM_CLASSES = 4
BEST_CFG = {  # F_1layer architecture
    "d_model": 32, "nhead": 4, "num_layers": 1,
    "dim_feedforward": 64, "dropout": 0.1,
}
BITS             = [16, 14, 12, 10, 8]
CALIB_FILES      = 12
TEST_FILES       = 40           # 40 test files include all 4 classes
TEST_MAX_WINDOWS = 4000         # stratified-subsample to bound the slow int64 ref


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

    # -- Locate inputs ---------------------------------------------------------
    cfg_hits = glob.glob(os.path.join(INPUT_ROOT, "**", "src", "config.py"),
                         recursive=True)
    if not cfg_hits:
        raise FileNotFoundError("attach mehrdadsheikhabbasi/fpga-fault-src")
    code_root = os.path.dirname(os.path.dirname(cfg_hits[0]))

    raw_dir = detect_mafaulda_root(INPUT_ROOT)
    if raw_dir is None:
        raise FileNotFoundError("attach vuxuancu/mafaulda-full")

    ckpt_hits = glob.glob(os.path.join(INPUT_ROOT, "**", "best.pt"), recursive=True)
    if not ckpt_hits:
        raise FileNotFoundError("attach mehrdadsheikhabbasi/fpga-fault-f1-ckpt")
    ckpt_path = ckpt_hits[0]

    print(f"code_root = {code_root}", flush=True)
    print(f"raw_dir   = {raw_dir}", flush=True)
    print(f"ckpt      = {ckpt_path}", flush=True)

    os.environ["PROJECT_ROOT"] = WORKING_DIR
    sys.path.insert(0, code_root)

    # patch data path BEFORE the subset builders read it
    import src.quantize.evaluate_integer as _ei
    import src.quantize.evaluate_quantized as _eq
    _ei.DATA_RAW_DIR = Path(raw_dir)
    _eq.DATA_RAW_DIR = Path(raw_dir)

    from src.quantize.evaluate_integer_bits import run_one
    from src.quantize.evaluate_integer import build_train_subset
    from src.quantize.evaluate_quantized import build_test_subset

    # -- Load checkpoint + remap keys ------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_np = {_remap_key(k): v.cpu().numpy()
                for k, v in ckpt["model_state_dict"].items()}
    cfg = {"input_dim": IN_CHANNELS, "seq_len": WINDOW_SIZE,
           "num_classes": NUM_CLASSES, **BEST_CFG}

    # -- Build subsets ---------------------------------------------------------
    Xc, _, kc = build_train_subset(CALIB_FILES, SEED)
    Xt, yt, sub = build_test_subset(TEST_FILES, SEED)

    if TEST_MAX_WINDOWS and len(Xt) > TEST_MAX_WINDOWS:
        rng = np.random.default_rng(SEED)
        frac = TEST_MAX_WINDOWS / len(Xt)
        keep = []
        for c in np.unique(yt):
            idx = np.flatnonzero(yt == c)
            n = max(1, int(round(len(idx) * frac)))
            keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        Xt, yt = Xt[keep], yt[keep]
        sub = {**sub, "windows": int(len(keep)),
               "subsampled_from": sub["windows"], "subsample_seed": SEED}

    counts = np.bincount(yt.astype(int), minlength=NUM_CLASSES).tolist()
    print(f"calib windows={len(Xc)} ({kc} files); "
          f"test windows={len(yt)} ({sub['test_files_used']} files)  "
          f"class_counts={counts}", flush=True)

    # -- Sweep -----------------------------------------------------------------
    results = []
    for b in BITS:
        t0 = time.perf_counter()
        r = run_one(b, state_np, cfg, Xc, Xt, yt)
        print(f"  int{b} done in {time.perf_counter()-t0:.1f}s "
              f"(elapsed {(time.perf_counter()-t_start)/60:.1f}m)", flush=True)
        results.append(r)

    out = {
        "stage":       "6f_multibit_sweep_winner",
        "winner":      "F_1layer",
        "model_cfg":   cfg,
        "bit_widths":  BITS,
        "test_subset": sub,
        "results":     results,
    }
    out_path = Path(WORKING_DIR) / "integer_multibit_sweep_F.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsweep -> {out_path}", flush=True)
    print(f"{'bits':>5} {'int_acc':>9} {'drop':>9} {'agree':>9}")
    for r in results:
        print(f"{r['bits']:>5} {r['int_accuracy']:>9.4f} "
              f"{r['accuracy_drop_vs_float']:>+9.4f} "
              f"{r['prediction_agreement_vs_float']:>9.4f}")


if __name__ == "__main__":
    main()
