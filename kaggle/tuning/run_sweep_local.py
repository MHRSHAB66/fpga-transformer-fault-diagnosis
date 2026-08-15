"""Local integer bit-width sweep (int16, int8) on the F_1layer winner.

Reproduces the kernel's sweep block without re-training: loads the downloaded
F_1layer checkpoint, remaps TuningTransformer state_dict keys to the names the
IntegerTransformer reference expects, then runs the pure-NumPy integer reference
at 16- and 8-bit on the same calib/test subsets. Local DATA_RAW_DIR is correct,
so build_*_subset find the real MAFAULDA files.
"""
import json
import os
import time
from pathlib import Path

import torch

from src.quantize.evaluate_integer_bits import run_one
from src.quantize.evaluate_integer import build_train_subset
from src.quantize.evaluate_quantized import build_test_subset

# -- constants copied from the kernel (fpga-fault-tuning.py) --------------------
WINDOW_SIZE = 1024
SEED        = 42
IN_CHANNELS = 8
NUM_CLASSES = 4
# Full-integer numpy reference with seq=1024 attention is far too slow on all
# test windows locally; use a representative subset (env-overridable), matching
# the day14 integer-eval scale (~25 calib / few-thousand-window test).
CALIB_FILES = int(os.environ.get("CALIB_FILES", 20))
TEST_FILES  = int(os.environ.get("TEST_FILES", 12))

BEST_CFG = {  # F_1layer arch (from the kernel log)
    "d_model": 32, "nhead": 4, "num_layers": 1,
    "dim_feedforward": 64, "dropout": 0.1,
}
CKPT = Path(__file__).parent / "_ckpt/tuning_20260814_142854/F_1layer/checkpoints/best.pt"
OUT  = Path(__file__).parent / "integer_bits_sweep_F.json"


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
    ckpt = torch.load(CKPT, map_location="cpu")
    state_np = {_remap_key(k): v.cpu().numpy()
                for k, v in ckpt["model_state_dict"].items()}
    cfg = {
        "input_dim": IN_CHANNELS, "seq_len": WINDOW_SIZE,
        "num_classes": NUM_CLASSES, **BEST_CFG,
    }
    Xc, _, kc = build_train_subset(CALIB_FILES, SEED)
    Xt, yt, sub = build_test_subset(TEST_FILES, SEED)
    print(f"calib windows={len(Xc)} ({kc} files); "
          f"test windows={sub['windows']} ({sub['test_files_used']} files)", flush=True)
    results = []
    for b in (16, 8):
        t0 = time.perf_counter()
        r = run_one(b, state_np, cfg, Xc, Xt, yt)
        print(f"  int{b} done in {time.perf_counter()-t0:.1f}s", flush=True)
        results.append(r)
    out = {
        "stage": "6f_bit_sweep_winner",
        "winner": "F_1layer",
        "model_cfg": cfg,
        "test_subset": sub,
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"integer sweep -> {OUT}")
    for r in results:
        print(f"  int{r.get('bits')}: acc={r.get('int_accuracy')} "
              f"agree={r.get('prediction_agreement_vs_float')}")


if __name__ == "__main__":
    main()
