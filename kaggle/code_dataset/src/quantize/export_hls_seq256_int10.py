"""
Export the independently trained SEQ=256/pool4 F_1layer int10 model to a new
HLS data contract without modifying the verified SEQ=1024 export.

F_1layer (1 encoder layer, d_model=32, nhead=4, ffn=64) is the FPGA-winner
architecture; the full-test int10 accuracy is 97.11% (see
kaggle/int10_shard/int10_fulltest_combined.json). HLS ap_int<10> makes the
narrow width free in hardware.

This script:
  1. loads the F checkpoint and remaps TuningTransformer state_dict keys to the
     names IntegerTransformer expects (same 1:1 adapter used in the sweep);
  2. set_bits(10) BEFORE building the model so weights quantize at 10 bits;
  3. calibrates activation scales on a TRAIN subset (train-only, as always);
  4. writes every HLS constant to hls/f1_int10/data + int10 golden vectors to
     hls/f1_int10/golden, reusing the exact writer/plan helpers from export_hls
     so the F export is byte-for-byte consistent with the A_baseline contract.

Run
---
    python -m src.quantize.export_hls_f_int10
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, DATA_RAW_DIR, LABEL_MAP
from src.data.load_data import build_file_index
from src.data.make_dataloaders import split_file_index, build_X_y_from_file_index
from src.data.token_aggregation import aggregate_mean_float, aggregate_round_div
import src.quantize.integer_model as im
from src.quantize.integer_model import IntegerTransformer, layer_positions
from src.quantize.integer_ops import qint_max
from src.quantize.evaluate_integer_bits import set_bits, calibrate_bits


def _write_ints(path: Path, arr) -> None:
    arr = np.asarray(arr).ravel().astype(np.int64)
    with open(path, "w", encoding="ascii") as f:
        f.write(f"{arr.size}\n")
        f.write(" ".join(str(int(v)) for v in arr))
        f.write("\n")


def _write_floats(path: Path, arr) -> None:
    arr = np.asarray(arr, dtype=np.float64).ravel()
    with open(path, "w", encoding="ascii") as f:
        f.write(f"{arr.size}\n")
        f.write(" ".join(repr(float(v)) for v in arr))
        f.write("\n")


def _linear_plan(m: IntegerTransformer, s: dict):
    def layer_in(l):
        return s["embed"] if l == 0 else s[layer_positions(l - 1)["ln2"]]

    plan = [("proj", s["input"], s["embed"])]
    for l in range(m.num_layers):
        pos = layer_positions(l)
        lin = layer_in(l)
        plan += [
            (f"l{l}.wq", lin, s[pos["q"]]),
            (f"l{l}.wk", lin, s[pos["k"]]),
            (f"l{l}.wv", lin, s[pos["v"]]),
            (f"l{l}.out", s[pos["ctx"]], s[pos["attn_out"]]),
            (f"l{l}.ff0", s[pos["ln1"]], s[pos["ff1"]]),
            (f"l{l}.ff3", s[pos["ff_act"]], s[pos["ff2"]]),
        ]
    plan += [
        ("cls1", s["cls_ln"], s["cls1"]),
        ("cls4", s["cls_act"], s["logits"]),
    ]
    return plan


BITS        = 10
CALIB_FILES = 20
SEED        = 42
CKPT_PATH   = (PROJECT_ROOT / "artifacts" / "experiments"
               / "transformer_f1_seq256_pool4" / "kaggle_v3"
               / "seq256_pool4" / "scratch" / "best.pt")
GOLDEN_SRC  = PROJECT_ROOT / "artifacts" / "golden" / "golden_vectors.npz"

HLS_DIR    = PROJECT_ROOT / "hls" / "f1_int10_seq256"
DATA_DIR   = HLS_DIR / "data"
GOLDEN_DIR = HLS_DIR / "golden"
INGRESS_GOLDEN_DIR = HLS_DIR / "golden_ingress1024"
CONFIG_DIR = PROJECT_ROOT / "artifacts" / "quantized" / "f_seq256_int10"

MODEL_CFG = {
    "input_dim": 8, "seq_len": 256, "num_classes": 4,
    "d_model": 32, "nhead": 4, "num_layers": 1,
    "dim_feedforward": 64, "dropout": 0.1,
}


def _build_calibration_subset(num_files: int, seed: int):
    """Use only the TRAIN split and apply the approved post-zscore pool4."""
    index_df = build_file_index(DATA_RAW_DIR)
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    train_df, _, _ = split_file_index(index_df)
    train_df = train_df.sample(n=num_files, random_state=seed).reset_index(drop=True)
    X, y, kept, _, _ = build_X_y_from_file_index(
        train_df,
        window_size=1024,
        stride=1024,
        aggregation_factor=4,
    )
    return X, y, int(kept)


def main():
    # Defaults export the deployed model into hls/f1_int10_seq256/. --ckpt with
    # --hls-dir/--config-dir exports another checkpoint of the same architecture
    # (e.g. a paper-study model) elsewhere and leaves the deployed export untouched.
    global CKPT_PATH, HLS_DIR, DATA_DIR, GOLDEN_DIR, INGRESS_GOLDEN_DIR, CONFIG_DIR
    ap = argparse.ArgumentParser(description="SEQ=256 INT10 HLS export")
    ap.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    ap.add_argument("--hls-dir", type=Path, default=HLS_DIR)
    ap.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    args = ap.parse_args()
    CKPT_PATH, HLS_DIR, CONFIG_DIR = args.ckpt, args.hls_dir, args.config_dir
    DATA_DIR, GOLDEN_DIR = HLS_DIR / "data", HLS_DIR / "golden"
    INGRESS_GOLDEN_DIR = HLS_DIR / "golden_ingress1024"

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_np = {k: v.cpu().numpy() for k, v in ckpt["model_state_dict"].items()}

    # bit-width must be set before construction so weights quantize at 10 bits
    set_bits(BITS)
    m = IntegerTransformer(state_np, MODEL_CFG)

    print(f"[calib] building train subset ({CALIB_FILES} files)...")
    Xc, _, kc = _build_calibration_subset(CALIB_FILES, SEED)
    print(f"[calib] {len(Xc)} windows from {kc} files")
    scales = calibrate_bits(m, Xc, BITS)
    m.set_scales(scales)
    s = scales
    print(f"[calib] {len(scales)} activation scales")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- linear weights, folded biases, requant multipliers -----------------
    for name, s_in, s_out in _linear_plan(m, s):
        qW, sW, b = m.qW[name], m.sW[name], m.bias[name]
        qb = np.round(b / (s_in * sW)).astype(np.int64)
        _write_ints(DATA_DIR / f"W_{name}.txt", qW)
        _write_ints(DATA_DIR / f"b_{name}.txt", qb)
        _write_floats(DATA_DIR / f"M_{name}.txt", [(s_in * sW) / s_out])

    # --- attention requant multipliers --------------------------------------
    s_attn = 1.0 / qint_max(im.ATTN_BITS)
    for l in range(m.num_layers):
        pos = layer_positions(l)
        m_scores = (s[pos["q"]] * s[pos["k"]]) / (math.sqrt(m.head_dim) * s[pos["scores"]])
        m_ctx = (s_attn * s[pos["v"]]) / s[pos["ctx"]]
        _write_floats(DATA_DIR / f"M_l{l}.scores.txt", [m_scores])
        _write_floats(DATA_DIR / f"M_l{l}.ctx.txt", [m_ctx])

    # --- mean-pool requant ---------------------------------------------------
    s_last = s[layer_positions(m.num_layers - 1)["ln2"]]
    _write_floats(DATA_DIR / "M_pool.txt", [s_last / (MODEL_CFG["seq_len"] * s["pool"])])

    # --- residual-add scale ratios ------------------------------------------
    def layer_in(l):
        return s["embed"] if l == 0 else s[layer_positions(l - 1)["ln2"]]
    for l in range(m.num_layers):
        pos = layer_positions(l)
        r1, r2 = s[pos["res1"]], s[pos["res2"]]
        _write_floats(DATA_DIR / f"add_l{l}.res1.txt",
                      [layer_in(l) / r1, s[pos["attn_out"]] / r1])
        _write_floats(DATA_DIR / f"add_l{l}.res2.txt",
                      [s[pos["ln1"]] / r2, s[pos["ff2"]] / r2])

    # --- positional-encoding codes ------------------------------------------
    _write_ints(DATA_DIR / "pe_codes.txt", np.round(m.pe / s["embed"]).astype(np.int64))

    # --- LayerNorm folded affine + integer eps ------------------------------
    RS_FRAC = 15
    ln_plan = []
    for l in range(m.num_layers):
        pos = layer_positions(l)
        pfx = f"encoder_layers.{l}."
        ln_plan.append((f"l{l}.ln1", pfx + "attention_norm", s[pos["res1"]], s[pos["ln1"]]))
        ln_plan.append((f"l{l}.ln2", pfx + "feedforward_norm", s[pos["res2"]], s[pos["ln2"]]))
    ln_plan.append(("cls_ln", "classifier.0", s["pool"], s["cls_ln"]))
    for name, key, s_in, s_out in ln_plan:
        _write_floats(DATA_DIR / f"ln_gm_{name}.txt", m.f[key + ".weight"] / ((1 << RS_FRAC) * s_out))
        _write_floats(DATA_DIR / f"ln_bt_{name}.txt", m.f[key + ".bias"] / s_out)
        _write_ints(DATA_DIR / f"ln_epsq_{name}.txt", [int(round(im.LN_EPS / (s_in * s_in)))])

    # --- LUTs ----------------------------------------------------------------
    for l in range(m.num_layers):
        _write_ints(DATA_DIR / f"gelu_l{l}.txt", m.gelu_lut[l])
        lut, clamp = m.exp_lut[l]
        _write_ints(DATA_DIR / f"exp_l{l}.txt", lut)
        _write_ints(DATA_DIR / f"exp_clamp_l{l}.txt", [int(clamp)])
    _write_ints(DATA_DIR / "gelu_cls.txt", m.cls_gelu_lut)
    _write_ints(DATA_DIR / "rsqrt.txt", m.rsqrt_lut)

    # --- integer golden vectors (reuse model-independent golden inputs) ------
    golden = np.load(GOLDEN_SRC)
    norm1024 = golden["normalized_input"].astype(np.float64)
    norm = aggregate_mean_float(norm1024, factor=4, output_dtype=np.float64)
    labels = golden["label_id"]
    qmax = qint_max(im.ACT_BITS)
    in_codes = np.clip(np.round(norm / s["input"]), -qmax, qmax).astype(np.int64)
    logits = m.infer_int(norm)
    preds = logits.argmax(1)

    with open(GOLDEN_DIR / "golden_in.txt", "w", encoding="ascii") as f:
        f.write(f"{in_codes.shape[0]} {in_codes.shape[1]} {in_codes.shape[2]}\n")
        for w in in_codes:
            f.write(" ".join(str(int(v)) for v in w.ravel()) + "\n")
    with open(GOLDEN_DIR / "golden_expected.txt", "w", encoding="ascii") as f:
        f.write("# idx true pred logit0 logit1 logit2 logit3\n")
        for i in range(len(logits)):
            row = " ".join(str(int(v)) for v in logits[i])
            f.write(f"{i} {int(labels[i])} {int(preds[i])} {row}\n")

    # Hardware-ingress contract: the interface receives 1024 normalized INT10
    # samples and performs the 4:1 signed round-half-away-from-zero mean before
    # the 256-token core. Quantize-then-average is not generally identical to
    # average-then-quantize, so keep a separate golden set and recompute the
    # expected logits from the exact aggregated integer codes.
    INGRESS_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    in_codes1024 = np.clip(
        np.round(norm1024 / s["input"]), -qmax, qmax
    ).astype(np.int64)
    ingress_codes256 = aggregate_round_div(in_codes1024, factor=4)
    ingress_logits = m.infer_int(ingress_codes256.astype(np.float64) * s["input"])
    ingress_preds = ingress_logits.argmax(1)
    with open(INGRESS_GOLDEN_DIR / "golden_in.txt", "w", encoding="ascii") as f:
        f.write(
            f"{in_codes1024.shape[0]} {in_codes1024.shape[1]} "
            f"{in_codes1024.shape[2]}\n"
        )
        for w in in_codes1024:
            f.write(" ".join(str(int(v)) for v in w.ravel()) + "\n")
    with open(
        INGRESS_GOLDEN_DIR / "golden_expected.txt", "w", encoding="ascii"
    ) as f:
        f.write("# idx true pred logit0 logit1 logit2 logit3\n")
        for i in range(len(ingress_logits)):
            row = " ".join(str(int(v)) for v in ingress_logits[i])
            f.write(
                f"{i} {int(labels[i])} {int(ingress_preds[i])} {row}\n"
            )

    # --- config + manifest ---------------------------------------------------
    with open(CONFIG_DIR / "integer_config.json", "w", encoding="utf-8") as f:
        json.dump({"stage": "6f_full_integer_seq256_int10", "model": "F_1layer_seq256_pool4",
                   "act_bits": BITS, "weight_bits": BITS, "attn_bits": BITS,
                   "calibration": {"split": "train", "files": kc, "windows": int(len(Xc))},
                   "scales": scales}, f, indent=2)

    manifest = {
        "stage": "7_hls_export_seq256_int10", "model": "F_1layer_seq256_pool4",
        "input_scale": s["input"],
        "act_bits": im.ACT_BITS, "weight_bits": im.W_BITS, "attn_bits": im.ATTN_BITS,
        "exp_frac": im.EXP_FRAC, "rsqrt_frac": RS_FRAC,
        "seq_len": MODEL_CFG["seq_len"], "in_channels": MODEL_CFG["input_dim"],
        "d_model": MODEL_CFG["d_model"], "nhead": MODEL_CFG["nhead"],
        "num_layers": MODEL_CFG["num_layers"], "dim_feedforward": MODEL_CFG["dim_feedforward"],
        "num_classes": MODEL_CFG["num_classes"],
        "golden": {"count": int(len(logits)), "correct": int((preds == labels).sum())},
        "ingress1024_golden": {
            "count": int(len(ingress_logits)),
            "correct": int((ingress_preds == labels).sum()),
            "prediction_changes_vs_float_mean": int((ingress_preds != preds).sum()),
        },
        "aggregation": "post-zscore non-overlapping mean, factor=4",
        "source_window_len": 1024,
        "note": "Independent SEQ=256 export; SEQ=1024 artifacts are untouched.",
    }
    with open(HLS_DIR / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[export] data   -> {DATA_DIR}")
    print(f"[export] golden -> {GOLDEN_DIR}")
    print(f"[export] ingress golden -> {INGRESS_GOLDEN_DIR}")
    print(f"[export] config -> {CONFIG_DIR / 'integer_config.json'}")
    print(f"[export] golden self-check: {manifest['golden']['correct']}/"
          f"{manifest['golden']['count']} correct")


if __name__ == "__main__":
    main()
