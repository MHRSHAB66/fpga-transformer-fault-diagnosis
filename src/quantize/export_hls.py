"""
Stage 7/8 — Export the full-integer model to a hardware-facing data contract.

Reads the calibrated scales (integer_config.json) and the checkpoint, rebuilds
the verified ``IntegerTransformer``, and dumps every constant the HLS core needs
as plain-text files under ``hls/data/`` plus integer golden vectors under
``hls/golden/``. Plain text keeps the C++ loader trivial and the values
inspectable.

What is exported (all derived from the SAME arithmetic the Python reference uses,
so the HLS core reproduces it bit-for-bit):

- per linear: int16 weight matrix ``W_<name>``, int bias ``b_<name>`` already
  divided by ``s_in*s_w``, and the requant multiplier ``M_<name> = s_in*s_w/s_out``;
- attention: ``M_l{l}.scores``, ``M_l{l}.ctx``; mean-pool ``M_pool``;
- residual adds: the two scale ratios ``add_l{l}.res1`` / ``.res2``;
- positional-encoding integer codes ``pe_codes``;
- LayerNorm: folded per-channel affine ``ln_gm`` / ``ln_bt`` and integer eps
  ``ln_epsq`` for each norm;
- LUTs: ``gelu_*`` (per layer + classifier), ``exp_l{l}`` (+ clamp), ``rsqrt``.

Run
---
    python -m src.quantize.export_hls
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.quantize.evaluate_quantized import DEFAULT_CHECKPOINT, load_model
from src.quantize.integer_model import (
    IntegerTransformer, layer_positions, ACT_BITS, ATTN_BITS, EXP_FRAC, LN_EPS,
)
from src.quantize.integer_ops import qint_max

CONFIG_PATH = PROJECT_ROOT / "artifacts" / "quantized" / "integer_config.json"
HLS_DIR = PROJECT_ROOT / "hls"
DATA_DIR = HLS_DIR / "data"
GOLDEN_DIR = HLS_DIR / "golden"
GOLDEN_SRC = PROJECT_ROOT / "artifacts" / "golden" / "golden_vectors.npz"


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
        f.write(" ".join(repr(float(v)) for v in arr))   # repr = full precision
        f.write("\n")


# names of the linear layers and the (s_in, s_out) positions feeding each requant
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


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        scales = json.load(f)["scales"]

    device = torch.device("cpu")
    _, float_state, model_cfg = load_model(DEFAULT_CHECKPOINT, device)
    state_np = {k: v.cpu().numpy() for k, v in float_state.items()}
    m = IntegerTransformer(state_np, model_cfg, scales)
    s = scales

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # --- linear weights, folded biases, requant multipliers -----------------
    for name, s_in, s_out in _linear_plan(m, s):
        qW, sW, b = m.qW[name], m.sW[name], m.bias[name]
        qb = np.round(b / (s_in * sW)).astype(np.int64)
        _write_ints(DATA_DIR / f"W_{name}.txt", qW)
        _write_ints(DATA_DIR / f"b_{name}.txt", qb)
        _write_floats(DATA_DIR / f"M_{name}.txt", [(s_in * sW) / s_out])

    # --- attention requant multipliers --------------------------------------
    s_attn = 1.0 / qint_max(ATTN_BITS)
    for l in range(m.num_layers):
        pos = layer_positions(l)
        m_scores = (s[pos["q"]] * s[pos["k"]]) / (math.sqrt(m.head_dim) * s[pos["scores"]])
        m_ctx = (s_attn * s[pos["v"]]) / s[pos["ctx"]]
        _write_floats(DATA_DIR / f"M_l{l}.scores.txt", [m_scores])
        _write_floats(DATA_DIR / f"M_l{l}.ctx.txt", [m_ctx])

    # --- mean-pool requant ---------------------------------------------------
    s_last = s[layer_positions(m.num_layers - 1)["ln2"]]
    _write_floats(DATA_DIR / "M_pool.txt", [s_last / (model_cfg["seq_len"] * s["pool"])])

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
    pe_codes = np.round(m.pe / s["embed"]).astype(np.int64)
    _write_ints(DATA_DIR / "pe_codes.txt", pe_codes)

    # --- LayerNorm folded affine + integer eps ------------------------------
    RS_FRAC = 15                                            # must match integer_ops
    ln_plan = []
    for l in range(m.num_layers):
        pos = layer_positions(l)
        pfx = f"encoder_layers.{l}."
        ln_plan.append((f"l{l}.ln1", pfx + "attention_norm", s[pos["res1"]], s[pos["ln1"]]))
        ln_plan.append((f"l{l}.ln2", pfx + "feedforward_norm", s[pos["res2"]], s[pos["ln2"]]))
    ln_plan.append(("cls_ln", "classifier.0", s["pool"], s["cls_ln"]))
    for name, key, s_in, s_out in ln_plan:
        gamma = m.f[key + ".weight"]
        beta = m.f[key + ".bias"]
        _write_floats(DATA_DIR / f"ln_gm_{name}.txt", gamma / ((1 << RS_FRAC) * s_out))
        _write_floats(DATA_DIR / f"ln_bt_{name}.txt", beta / s_out)
        _write_ints(DATA_DIR / f"ln_epsq_{name}.txt", [int(round(LN_EPS / (s_in * s_in)))])

    # --- LUTs ----------------------------------------------------------------
    for l in range(m.num_layers):
        _write_ints(DATA_DIR / f"gelu_l{l}.txt", m.gelu_lut[l])
        lut, clamp = m.exp_lut[l]
        _write_ints(DATA_DIR / f"exp_l{l}.txt", lut)
        _write_ints(DATA_DIR / f"exp_clamp_l{l}.txt", [int(clamp)])
    _write_ints(DATA_DIR / "gelu_cls.txt", m.cls_gelu_lut)
    _write_ints(DATA_DIR / "rsqrt.txt", m.rsqrt_lut)

    # --- integer golden vectors ---------------------------------------------
    golden = np.load(GOLDEN_SRC)
    norm = golden["normalized_input"].astype(np.float64)   # (N,1024,8)
    labels = golden["label_id"]
    qmax = qint_max(ACT_BITS)
    in_codes = np.clip(np.round(norm / s["input"]), -qmax, qmax).astype(np.int64)
    logits = m.infer_int(norm)                             # reference int logits
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

    # --- manifest ------------------------------------------------------------
    manifest = {
        "stage": "7_hls_export",
        "input_scale": s["input"],
        "act_bits": ACT_BITS, "weight_bits": 16,
        "attn_bits": ATTN_BITS, "exp_frac": EXP_FRAC, "rsqrt_frac": RS_FRAC,
        "seq_len": model_cfg["seq_len"], "in_channels": model_cfg["input_dim"],
        "d_model": model_cfg["d_model"], "nhead": model_cfg["nhead"],
        "num_layers": model_cfg["num_layers"],
        "dim_feedforward": model_cfg["dim_feedforward"],
        "num_classes": model_cfg["num_classes"],
        "golden": {"count": int(len(logits)),
                   "correct": int((preds == labels).sum())},
        "note": "requant/affine constants are float64 here (bit-exact with the "
                "Python reference); convert to fixed-point per target board.",
    }
    with open(HLS_DIR / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[export] wrote data -> {DATA_DIR}")
    print(f"[export] wrote golden -> {GOLDEN_DIR}")
    print(f"[export] golden self-check: {manifest['golden']['correct']}/"
          f"{manifest['golden']['count']} correct")


if __name__ == "__main__":
    main()
