"""
Standard software metrics (precision / recall / F1 / confusion) for the deployed
int10 F_1layer model.

The definitive int10 FULL-test *accuracy* (97.11%) and per-class *recall* were
already measured by the 4-way sharded Kaggle run
(kaggle/int10_shard/int10_fulltest_combined.json). Those shards recorded only
counts, so they give recall but not precision/F1/confusion. Re-running the full
int64 reference locally is ~8 h (≈0.29 s/window x 95,404), so this script computes
a *measured confusion matrix* on a stratified subset of the test set, which yields
precision/recall/F1 for int10, and echoes the full-test accuracy/recall for
reference.

Setup matches the deployed int10 exactly (SEED=42, CALIB_FILES=12, bits=10,
F_1layer checkpoint with TuningTransformer key remap).

Usage
-----
    python -m src.quantize.evaluate_int10_metrics --per-class 400
Output
------
    artifacts/results/f_1layer_int10_test_metrics.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT, NUM_CLASSES, RANDOM_SEED, LABEL_MAP
from src.quantize.integer_model import IntegerTransformer
from src.quantize.evaluate_integer_bits import set_bits, calibrate_bits
from src.quantize.evaluate_integer import int_logits, build_train_subset
from src.quantize.evaluate_quantized import build_test_subset
from src.evaluation_metrics import precision_recall_fbeta_from_confusion

CKPT = PROJECT_ROOT / "kaggle" / "f1_ckpt_dataset" / "best.pt"
FULLTEST = PROJECT_ROOT / "kaggle" / "int10_shard" / "int10_fulltest_combined.json"
OUT = PROJECT_ROOT / "artifacts" / "results" / "f_1layer_int10_test_metrics.json"

BITS = 10
CALIB_FILES = 12
BEST_CFG = {"d_model": 32, "nhead": 4, "num_layers": 1,
            "dim_feedforward": 64, "dropout": 0.1}
CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]


def _remap_key(k: str) -> str:
    """TuningTransformer -> TransformerBaseline attribute names (name-only adapter)."""
    k = k.replace("pos_enc.", "positional_encoding.")
    k = k.replace("input_proj.", "input_projection.")
    if k.startswith("encoder."):
        parts = k.split(".")
        l, rest = parts[1], ".".join(parts[2:])
        for a, b in (("attn", "self_attention"), ("attn_norm", "attention_norm"),
                     ("ffn_norm", "feedforward_norm"), ("ffn", "feedforward")):
            if rest == a or rest.startswith(a + "."):
                rest = b + rest[len(a):]
                break
        k = f"encoder_layers.{l}.{rest}"
    return k


def stratified_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(NUM_CLASSES):
        c_idx = np.where(y == c)[0]
        take = min(per_class, len(c_idx))
        idx.append(rng.choice(c_idx, size=take, replace=False))
    return np.sort(np.concatenate(idx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400,
                    help="windows per class in the measured subset")
    args = ap.parse_args()

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state_np = {_remap_key(k): v.cpu().numpy()
                for k, v in ckpt["model_state_dict"].items()}
    cfg = {"input_dim": 8, "seq_len": 1024, "num_classes": NUM_CLASSES, **BEST_CFG}

    print(f"[calib] building train subset ({CALIB_FILES} files) ...")
    Xc, _, kc = build_train_subset(CALIB_FILES, RANDOM_SEED)
    print(f"[calib] {len(Xc)} calibration windows from {kc} files")

    print("[test] building full test set ...")
    Xt, yt, sub = build_test_subset(10 ** 6, RANDOM_SEED)
    print(f"[test] {len(yt)} windows from {sub['test_files_used']} files")

    sel = stratified_indices(yt, args.per_class, RANDOM_SEED)
    Xs, ys = Xt[sel].astype(np.float64), yt[sel].astype(np.int64)
    del Xt
    print(f"[subset] {len(ys)} stratified windows "
          f"({np.bincount(ys, minlength=NUM_CLASSES).tolist()})")

    set_bits(BITS)
    model = IntegerTransformer(state_np, cfg)
    scales = calibrate_bits(model, Xc, BITS)
    model.set_scales(scales)

    print("[int10] running integer inference on subset ...")
    preds = int_logits(model, Xs).argmax(1)

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(ys, preds):
        confusion[t, p] += 1
    prf1 = precision_recall_fbeta_from_confusion(confusion, CLASS_NAMES, beta=1.0)
    subset_acc = float((preds == ys).mean())

    full = json.loads(FULLTEST.read_text())

    result = {
        "model": "F_1layer",
        "quantization": "int10 (weights+activations+attention)",
        "checkpoint": str(CKPT.relative_to(PROJECT_ROOT)),
        "calibration": {"files": CALIB_FILES, "windows": int(len(Xc)), "seed": RANDOM_SEED},
        "measured_on": "stratified test subset",
        "subset_per_class": args.per_class,
        "subset_windows": int(len(ys)),
        "subset_accuracy": subset_acc,
        "precision_recall_f1": prf1,
        "confusion_matrix": confusion.tolist(),
        "class_names": CLASS_NAMES,
        "full_test_reference": {
            "source": str(FULLTEST.relative_to(PROJECT_ROOT)),
            "windows": full["full_test_windows"],
            "accuracy": full["int_accuracy"],
            "per_class_recall": {k: v["accuracy"] for k, v in full["per_class"].items()},
        },
        "generated": datetime.now().isoformat(timespec="seconds"),
        "note": "Full-test accuracy/recall are from the definitive 4-shard run; "
                "precision/F1/confusion are measured on the stratified subset "
                "(full int10 confusion would need ~8 h locally).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"int10 subset accuracy : {subset_acc*100:.2f}%  ({len(ys)} windows)")
    print(f"macro    P/R/F1       : {prf1['macro']['precision']:.4f} / "
          f"{prf1['macro']['recall']:.4f} / {prf1['macro']['f1']:.4f}")
    print("per-class:")
    for name in CLASS_NAMES:
        pc = prf1["per_class"][name]
        print(f"  {name:<14} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")
    print(f"\nfull-test accuracy (4-shard) : {full['int_accuracy']*100:.2f}%")
    print(f"-> {OUT.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
