"""Private Kaggle experiment (paper study): 10-class MAFAULDA labels on the
deployed F_1layer SEQ=256 architecture. The deployed 4-class model is untouched.

Same as kaggle/seq256_train (scratch candidate): identical project file split
(split_file_index, stratified on the 4 groups; asserted by hashing the file
lists), identical windowing/normalization/4:1 aggregation, seed 42, Adam 1e-3,
batch 64, <= 30 epochs, patience 6 on validation macro-F1, class-weighted
cross-entropy, scratch init. Only the labels change (file level, from the path):

  0 normal                  5 underhang/cage_fault
  1 imbalance               6 underhang/outer_race
  2 horizontal-misalignment 7 overhang/ball_fault
  3 vertical-misalignment   8 overhang/cage_fault
  4 underhang/ball_fault    9 overhang/outer_race

The test split is evaluated once after selection; the same 10-class predictions
are also scored after collapsing to the 6 MAFAULDA groups and to the project's
4 classes. Outputs (small): /kaggle/working/seq256_10class/.
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import random
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

INPUT_ROOT = "/kaggle/input"
OUTPUT = Path("/kaggle/working/seq256_10class")
SEED = 42
EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 64
LR = 1e-3
MODEL_KW = dict(seq_len=256, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1)

CLASSES10 = ["normal", "imbalance", "horizontal-misalignment", "vertical-misalignment",
             "underhang/ball_fault", "underhang/cage_fault", "underhang/outer_race",
             "overhang/ball_fault", "overhang/cage_fault", "overhang/outer_race"]
TO6 = [0, 1, 2, 3, 4, 4, 4, 5, 5, 5]          # normal, imb, h-mis, v-mis, underhang, overhang
TO4 = [0, 1, 2, 2, 3, 3, 3, 3, 3, 3]          # the project's 4 classes (config LABEL_MAP)
# sha256 of the sorted relative paths of each split, computed locally from the
# project split (artifacts/results/... deployed model: 391 test files, 95,404 windows)
SPLIT_SHA = {"train": "743560d8b18ce945610a60938b77171ec0bceb518999ada304c776dad5f1c9a0",
             "val": "8cc36038fc7b968787452624b3ac513811019cf0fb017dc1dd75a0da5c659a85",
             "test": "3db5cc4652a5995196bdf4a6359275e52ead906e82e6c3cec01b30ba52161246"}


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


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label10(rel):
    parts = rel.split("/")
    name = parts[0] if parts[0] in CLASSES10 else parts[0] + "/" + parts[1]
    return CLASSES10.index(name)


def windows_of(path):
    """Same steps and checks as src.data.make_dataloaders.build_X_y_from_file_index."""
    from src.config import EXPECTED_NUM_COLUMNS
    from src.data.load_data import load_one_csv
    from src.data.prepare_dataset import (create_windows_from_dataframe, normalize_windows_array,
                                          preprocess_dataframe)
    from src.data.token_aggregation import aggregate_mean_float
    df = preprocess_dataframe(load_one_csv(path))
    if df is None or df.shape[1] != EXPECTED_NUM_COLUMNS or len(df) < 1024:
        return None
    w = create_windows_from_dataframe(df.to_numpy(copy=True).astype(np.float32, copy=False),
                                      window_size=1024, stride=1024)
    if not w:
        return np.empty((0, 256, EXPECTED_NUM_COLUMNS), np.float32)
    w = normalize_windows_array(np.asarray(w, dtype=np.float32))
    return aggregate_mean_float(w, 4, output_dtype=np.float32)


def build_split(df, data_root, pool):
    rel = [Path(p).resolve().relative_to(data_root.resolve()).as_posix() for p in df["filepath"]]
    arrays = pool.map(windows_of, list(df["filepath"]), chunksize=4)
    X, y, f = [], [], []
    for k, (a, r) in enumerate(zip(arrays, rel)):
        if a is None or len(a) == 0:
            continue
        X.append(a)
        y.append(np.full(len(a), label10(r), np.int64))
        f.append(np.full(len(a), k, np.int32))
    return np.concatenate(X), np.concatenate(y), np.concatenate(f), rel


def metrics(y, pred, n):
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) != 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) != 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) != 0)
    return {"accuracy": float(tp.sum() / len(y)), "macro_f1": float(f1.mean()),
            "per_class_precision": prec.tolist(), "per_class_recall": rec.tolist(),
            "per_class_f1": f1.tolist(), "confusion_matrix": cm.tolist(), "count": int(len(y))}


def predict(model, X, device, batch=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).to(device)).float().cpu())
    return torch.cat(out).numpy()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set_seed()
    code_root, data_root = locate_code_root(), locate_data_root()
    os.environ["PROJECT_ROOT"] = "/kaggle/working"
    # build_file_index only accepts directories inside PROJECT_ROOT/data/raw/mafaulda
    # (DATA_RAW_DIR); the read-only input mount is linked there.
    link = Path("/kaggle/working/data/raw/mafaulda")
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(data_root, target_is_directory=True)
    sys.path.insert(0, str(code_root))
    from src.config import LABEL_MAP
    from src.data.load_data import build_file_index
    from src.data.make_dataloaders import split_file_index
    from src.models.transformer_baseline import TransformerBaseline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU was requested but CUDA is unavailable")
    print(f"device={torch.cuda.get_device_name(0)} code={code_root} data={data_root}", flush=True)

    idx = build_file_index(data_root)
    idx = idx[idx["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)
    splits = dict(zip(("train", "val", "test"), split_file_index(idx)))
    t0 = time.time()
    data = {}
    with Pool(4) as pool:
        for name, df in splits.items():
            X, y, f, rel = build_split(df, data_root, pool)
            digest = hashlib.sha256("\n".join(sorted(rel)).encode()).hexdigest()
            if digest != SPLIT_SHA[name]:
                raise RuntimeError(f"{name} split differs from the project split ({digest})")
            data[name] = (X, y, f)
            print(f"{name}: files={len(rel)} windows={len(y)} per_class={np.bincount(y, minlength=10).tolist()}"
                  f" ({time.time() - t0:.0f} s)", flush=True)
    assert len(data["test"][1]) == 95404, len(data["test"][1])

    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    Xte, yte, fte = data["test"]
    set_seed()
    model = TransformerBaseline(num_classes=10, **MODEL_KW).to(device)
    counts = np.bincount(ytr, minlength=10).astype(np.float64)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (10 * counts), dtype=torch.float32,
                                                        device=device))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    best, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        seen = correct = 0
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item()
            seen += yb.numel()
        val = metrics(yva, predict(model, Xva, device).argmax(1), 10)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "val_per_class_recall": val["per_class_recall"]})
        print(f"epoch={epoch} train_acc={correct / seen:.5f} val_acc={val['accuracy']:.5f} "
              f"val_macro_f1={val['macro_f1']:.5f} ({time.time() - started:.0f} s)", flush=True)
        if val["macro_f1"] > best + 1e-12:
            best, best_epoch, best_state, stale = val["macro_f1"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    logits = predict(model, Xte, device)
    pred = logits.argmax(1)
    from sklearn.metrics import roc_auc_score
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    to6, to4 = np.array(TO6), np.array(TO4)
    report = {
        "status": "complete", "classes10": CLASSES10, "classes6": ["normal", "imbalance",
                  "horizontal-misalignment", "vertical-misalignment", "underhang", "overhang"],
        "best_epoch": best_epoch, "best_val_macro_f1": best,
        "train_seconds": round(time.time() - started, 1),
        "windows": {k: int(len(v[1])) for k, v in data.items()},
        "test_10class": dict(metrics(yte, pred, 10),
                             macro_auroc=float(roc_auc_score(yte, prob, multi_class="ovr", average="macro"))),
        "test_collapsed_to_6": metrics(to6[yte], to6[pred], 6),
        "test_collapsed_to_4": metrics(to4[yte], to4[pred], 4),
        "model_kwargs": dict(MODEL_KW, num_classes=10), "seed": SEED,
        "device": torch.cuda.get_device_name(0), "split_sha256": SPLIT_SHA,
        "test_evaluated_once_after_selection": True,
    }
    torch.save({"model_state_dict": best_state, "model_kwargs": dict(MODEL_KW, num_classes=10),
                "classes": CLASSES10, "best_epoch": best_epoch, "best_val_macro_f1": best, "seed": SEED},
               OUTPUT / "best.pt")
    (OUTPUT / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    np.savez_compressed(OUTPUT / "test_predictions.npz", label10=yte.astype(np.int8),
                        pred10=pred.astype(np.int8), file_index=fte, logits=logits.astype(np.float16))
    print(json.dumps({k: report[k] for k in ("best_epoch", "best_val_macro_f1", "windows")}), flush=True)
    for k in ("test_10class", "test_collapsed_to_6", "test_collapsed_to_4"):
        print(k, "acc=%.5f macro_f1=%.5f" % (report[k]["accuracy"], report[k]["macro_f1"]),
              "recall", [round(v, 4) for v in report[k]["per_class_recall"]], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
