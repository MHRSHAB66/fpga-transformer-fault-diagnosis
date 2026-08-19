"""
Evaluate the deployed F_1layer model on the TEST split and export the standard
software metrics (accuracy, precision, recall, F1, confusion matrix).

The tuning run only saved VALIDATION metrics (and as F2, not F1). For the thesis
narrative "F_1layer is the primary/deployed model" we need a clean TEST metrics
set with standard F1. This script produces it from the local F_1layer checkpoint,
reusing the exact same data pipeline (file-level split, per-window z-score) as the
2-layer baseline so the numbers are comparable.

Usage
-----
    python -m src.evaluate.evaluate_f1layer
Output
------
    artifacts/results/f_1layer_test_metrics.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import PROJECT_ROOT, NUM_CLASSES, LABEL_MAP, RANDOM_SEED
from src.models.transformer_baseline import TransformerBaseline
from src.data.make_dataloaders import build_dataloaders
from src.train_transformer_baseline import evaluate_with_details, set_seed
from src.evaluation_metrics import precision_recall_fbeta_from_confusion

CKPT = PROJECT_ROOT / "kaggle" / "f1_ckpt_dataset" / "best.pt"
OUT = PROJECT_ROOT / "artifacts" / "results" / "f_1layer_test_metrics.json"
CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]


def _remap_key(k: str) -> str:
    """
    The F_1layer checkpoint was trained with TuningTransformer, which uses the
    same architecture as TransformerBaseline but shorter attribute names. Remap
    keys 1:1 (pure name adapter, no numeric change) -- identical to the mapping in
    kaggle/tuning/fpga-fault-tuning.py.
    """
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


def main() -> int:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[INFO] F_1layer config: {json.dumps(cfg)}")

    model = TransformerBaseline(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    state = {_remap_key(k): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] loaded F_1layer ({n_params:,} params, "
          f"ckpt epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc')})")

    # The G: drive is currently full. Disable NEW disk cache writes so eval runs
    # in-memory: already-cached files still load from disk; uncached files are
    # computed in memory and simply not written. No numeric change to results.
    import src.data.preprocessed_cache as _pc
    _pc.save_preprocessed_array_cache = lambda *a, **k: None
    print("[INFO] disk cache writes disabled (disk full) -- computing in-memory")

    print("[INFO] building dataloaders (file-level split, per-window z-score) ...")
    _, _, test_loader, stats = build_dataloaders()

    criterion = nn.CrossEntropyLoss()
    loss, acc, confusion, per_class_acc = evaluate_with_details(
        model, test_loader, criterion, device, NUM_CLASSES
    )
    confusion = np.asarray(confusion, dtype=np.int64)

    prf1 = precision_recall_fbeta_from_confusion(confusion, CLASS_NAMES, beta=1.0)

    result = {
        "model": "F_1layer",
        "checkpoint": str(CKPT.relative_to(PROJECT_ROOT)),
        "n_params": int(n_params),
        "split": "test",
        "num_test_windows": int(confusion.sum()),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "precision_recall_f1": prf1,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
        "class_names": CLASS_NAMES,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "note": "Standard F1 (beta=1) on the TEST split, same pipeline as the "
                "2-layer baseline. Float model (not int10).",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"F_1layer TEST accuracy : {acc*100:.2f}%  ({confusion.sum()} windows)")
    print(f"macro    P/R/F1        : "
          f"{prf1['macro']['precision']:.4f} / {prf1['macro']['recall']:.4f} / "
          f"{prf1['macro']['f1']:.4f}")
    print(f"weighted P/R/F1        : "
          f"{prf1['weighted']['precision']:.4f} / {prf1['weighted']['recall']:.4f} / "
          f"{prf1['weighted']['f1']:.4f}")
    print("\nper-class:")
    for name in CLASS_NAMES:
        pc = prf1["per_class"][name]
        print(f"  {name:<14} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")
    print(f"\n-> {OUT.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
