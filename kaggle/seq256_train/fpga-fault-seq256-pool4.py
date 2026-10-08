"""Private Kaggle experiment: SEQ=1024 windows -> 4:1 mean tokens -> SEQ=256.

Train warm-start and scratch candidates on the identical file-level split,
select by validation macro-F1, then evaluate the test split once for the winner.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_ROOT = "/kaggle/input"
OUTPUT = Path("/kaggle/working/seq256_pool4")
SEED = 42
EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 64
LR = 1e-3


def locate_code_root():
    hits = glob.glob(f"{INPUT_ROOT}/**/src/config.py", recursive=True)
    if not hits:
        raise FileNotFoundError("fpga-fault-src with src/config.py is not attached")
    return Path(hits[0]).parents[1]


def locate_data_root():
    markers = {"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment"}
    candidates = []
    for root, dirs, _ in os.walk(INPUT_ROOT):
        score = len(markers.intersection(d.lower() for d in dirs))
        if score >= 3:
            candidates.append((score, len(root), root))
    if not candidates:
        raise FileNotFoundError("MAFAULDA root not found")
    return Path(sorted(candidates, key=lambda v: (-v[0], v[1]))[0][2])


def locate_checkpoint():
    hits = [p for p in glob.glob(f"{INPUT_ROOT}/**/best.pt", recursive=True)
            if "f1" in p.lower() or "ckpt" in p.lower()]
    if not hits:
        raise FileNotFoundError("F_1layer best.pt not found")
    return Path(hits[0])


def remap_key(key):
    key = key.replace("pos_enc.", "positional_encoding.")
    key = key.replace("input_proj.", "input_projection.")
    if key.startswith("encoder."):
        parts = key.split(".")
        layer, rest = parts[1], ".".join(parts[2:])
        for old, new in (("attn", "self_attention"),
                         ("attn_norm", "attention_norm"),
                         ("ffn_norm", "feedforward_norm"),
                         ("ffn", "feedforward")):
            if rest == old or rest.startswith(old + "."):
                rest = new + rest[len(old):]
                break
        key = f"encoder_layers.{layer}.{rest}"
    return key


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confusion_and_metrics(model, loader, device, criterion):
    model.eval()
    cm = np.zeros((4, 4), dtype=np.int64)
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.numel()
            pred = logits.argmax(1)
            for truth, guess in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[int(truth), int(guess)] += 1
            count += y.numel()
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(4), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(4), where=(precision + recall) != 0)
    return {
        "loss": loss_sum / count,
        "accuracy": float(tp.sum() / count),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "count": int(count),
    }


def class_weighted_loss(train_loader, device):
    labels = train_loader.dataset.tensors[1].numpy()
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    weights = labels.size / (4.0 * counts)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def train_candidate(name, model, train_loader, val_loader, device):
    criterion = class_weighted_loss(train_loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    history = []
    stale = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
        val = confusion_and_metrics(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / seen,
               "train_accuracy": correct / seen, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"[{name}] epoch={epoch} train_acc={correct/seen:.5f} "
              f"val_acc={val['accuracy']:.5f} val_macro_f1={val['macro_f1']:.5f}", flush=True)
        if val["macro_f1"] > best_f1 + 1e-12:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    out = OUTPUT / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1, "seq_len": 256,
                "aggregation_factor": 4, "seed": SEED}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, {"best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                   "elapsed_seconds": time.time() - started}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set_seed()
    code_root = locate_code_root()
    data_root = locate_data_root()
    checkpoint = locate_checkpoint()
    os.environ["PROJECT_ROOT"] = "/kaggle/working"
    sys.path.insert(0, str(code_root))
    from src.data.make_dataloaders import build_dataloaders
    from src.models.transformer_baseline import TransformerBaseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU was requested but CUDA is unavailable")
    print(f"device={device} gpu={torch.cuda.get_device_name(0)} code={code_root} data={data_root}", flush=True)
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir=data_root, batch_size=BATCH_SIZE, window_size=1024, stride=1024,
        aggregation_factor=4)
    assert stats["model_seq_len"] == 256

    reference_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference_cfg = reference_ckpt["config"]
    model_kwargs = {
        "seq_len": 256,
        "d_model": int(reference_cfg["d_model"]),
        "nhead": int(reference_cfg["nhead"]),
        "num_layers": int(reference_cfg["num_layers"]),
        "dim_feedforward": int(reference_cfg["dim_feedforward"]),
        "dropout": float(reference_cfg.get("dropout", 0.1)),
    }
    if model_kwargs["num_layers"] != 1:
        raise RuntimeError(f"Expected F_1layer checkpoint, got {model_kwargs}")

    candidates = {}
    trained = {}
    for name in ("warm_start", "scratch"):
        set_seed()
        model = TransformerBaseline(**model_kwargs).to(device)
        if name == "warm_start":
            state = {remap_key(k): v for k, v in reference_ckpt["model_state_dict"].items()
                     if remap_key(k) != "positional_encoding.pe"}
            incompat = model.load_state_dict(state, strict=False)
            if set(incompat.missing_keys) != {"positional_encoding.pe"} or incompat.unexpected_keys:
                raise RuntimeError(f"warm-start incompatibility: {incompat}")
        model, summary = train_candidate(name, model, train_loader, val_loader, device)
        candidates[name] = summary
        trained[name] = model.cpu()
        torch.cuda.empty_cache()

    winner = max(candidates, key=lambda n: candidates[n]["best_val_macro_f1"])
    winner_model = trained[winner].to(device)
    criterion = class_weighted_loss(train_loader, device)
    test = confusion_and_metrics(winner_model, test_loader, device, criterion)
    report = {
        "status": "complete", "selection_rule": "highest validation macro-F1",
        "winner": winner, "candidates": candidates, "winner_test": test,
        "data_stats": stats, "checkpoint_source": str(checkpoint),
        "device": torch.cuda.get_device_name(0), "seed": SEED,
        "test_evaluated_once_after_selection": True,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
