"""Stricter generalization splits and noise-augmented training for the deployed
F_1layer SEQ=256 architecture (paper analysis; the deployed model is unchanged).

The project split is a stratified random split of files: every test file has a
training file of the same fault type and severity at a rotation speed ~1 Hz away.
This study retrains the SAME architecture with the SAME recipe as the deployed
model (kaggle/seq256_train: scratch, seed 42, Adam 1e-3, batch 64, <= 30 epochs,
patience 6 on validation macro-F1, class-weighted cross-entropy) on stricter
file-level splits, and tests noise-augmented training:

  random        the project split (control, same test files as the board results)
  sev_mid       hold out a middle severity / extra-load level of every fault family
                (test) and another one (validation); normal: a middle speed band
  sev_max       hold out the highest level of every family (extrapolation)
  speed_mid     hold out the 37-47 Hz speed band of every condition (test) and the
                22-27 Hz band (validation)
  random_noise  project split, training windows get Gaussian noise at a random SNR
                in [0, 30] dB (half of the batches), applied in the z-scored,
                4:1-aggregated domain (see augment())

    python -m tools.generalization_study cache
    python -m tools.generalization_study train --split sev_mid
    python -m tools.generalization_study noise-eval --models random random_noise
Results: artifacts/results/generalization_20261110/
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("D:/fpga_cache/seq256_all")
OUT = REPO / "artifacts" / "results" / "generalization_20261110"
RAW_TEST = REPO / "artifacts" / "results" / "fpga_rawinput_testset_20261109" / "fpga_testset_raw.i32.gz"
SEED, EPOCHS, PATIENCE, BATCH, LR = 42, 30, 6, 64, 1e-3
MODEL_KW = dict(seq_len=256, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1)

SEV_MID = {"imbalance": ("20g", "10g"), "horizontal-misalignment": ("1.0mm", "1.5mm"),
           "vertical-misalignment": ("1.27mm", "0.63mm")}
SEV_MAX = {"imbalance": ("35g", "30g"), "horizontal-misalignment": ("2.0mm", "1.5mm"),
           "vertical-misalignment": ("1.90mm", "1.78mm")}
BEARING_MID, BEARING_MAX = ("20g", "6g"), ("35g", "20g")


# ---------------------------------------------------------------------------
# data cache: every window of every file, z-scored and 4:1 aggregated (float32)
# ---------------------------------------------------------------------------
def _file_rows():
    from src.config import DATA_RAW_DIR, LABEL_MAP
    from src.data.load_data import build_file_index
    idx = build_file_index(Path(DATA_RAW_DIR))
    idx = idx[idx["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)
    return idx, LABEL_MAP


def _windows_of(path):
    from src.config import EXPECTED_NUM_COLUMNS
    from src.data.load_data import load_one_csv
    from src.data.prepare_dataset import (create_windows_from_dataframe, normalize_windows_array,
                                          preprocess_dataframe)
    from src.data.token_aggregation import aggregate_mean_float
    df = preprocess_dataframe(load_one_csv(path))
    if df is None or df.shape[1] != EXPECTED_NUM_COLUMNS or len(df) < 1024:
        return None
    w = create_windows_from_dataframe(df.to_numpy(copy=True).astype(np.float32), window_size=1024, stride=1024)
    w = normalize_windows_array(np.asarray(w, dtype=np.float32))
    return aggregate_mean_float(w, 4, output_dtype=np.float32)


def cmd_cache(_args):
    idx, label_map = _file_rows()
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(6) as pool:
        arrays = pool.map(_windows_of, list(idx["filepath"]), chunksize=4)
    keep = [i for i, a in enumerate(arrays) if a is not None]
    n = sum(len(arrays[i]) for i in keep)
    x = np.lib.format.open_memmap(CACHE / "x.npy", mode="w+", dtype=np.float32, shape=(n, 256, 8))
    file_of, pos = np.empty(n, np.int32), 0
    for i in keep:
        a = arrays[i]
        x[pos:pos + len(a)] = a
        file_of[pos:pos + len(a)] = i
        pos += len(a)
    x.flush()
    parts = [Path(p).parts for p in idx["filepath"]]
    cond = ["/".join(pp[pp.index("mafaulda") + 1:-1]) for pp in parts]
    np.savez(CACHE / "meta.npz", file_of=file_of,
             file_label=np.array([label_map[l] for l in idx["label"]], np.int64),
             file_cond=np.array(cond), file_speed=np.array([float(Path(p).stem) for p in idx["filepath"]]),
             file_path=np.array([str(p) for p in idx["filepath"]]), kept=np.array(keep))
    print("cached %d windows from %d files in %.0f s" % (n, len(keep), time.time() - t0))


# ---------------------------------------------------------------------------
# splits (file level)
# ---------------------------------------------------------------------------
def file_split(name, meta):
    cond, speed, label = meta["file_cond"], meta["file_speed"], meta["file_label"]
    nfiles = len(cond)
    assign = np.full(nfiles, "train", dtype=object)
    if name.startswith("random"):
        from src.data.make_dataloaders import split_file_index
        idx, _ = _file_rows()
        tr, va, te = split_file_index(idx)
        where = {p: i for i, p in enumerate(meta["file_path"])}
        for df, tag in ((va, "val"), (te, "test")):
            for p in df["filepath"]:
                assign[where[str(p)]] = tag
        return assign
    if name == "speed_mid":
        assign[(speed >= 37) & (speed < 47)] = "test"
        assign[(speed >= 22) & (speed < 27)] = "val"
        return assign
    table = SEV_MID if name == "sev_mid" else SEV_MAX
    bearing = BEARING_MID if name == "sev_mid" else BEARING_MAX
    for i, c in enumerate(cond):
        parts = c.split("/")
        if parts[0] == "normal":
            continue
        test_lvl, val_lvl = table[parts[0]] if parts[0] in table else bearing
        if parts[-1] == test_lvl:
            assign[i] = "test"
        elif parts[-1] == val_lvl:
            assign[i] = "val"
    normal = np.where(cond == "normal")[0]
    s = speed[normal]
    if name == "sev_mid":
        assign[normal[(s >= 37) & (s < 47)]] = "test"
        assign[normal[(s >= 22) & (s < 27)]] = "val"
    else:
        assign[normal[s >= 52]] = "test"
        assign[normal[(s >= 47) & (s < 52)]] = "val"
    return assign


# ---------------------------------------------------------------------------
# training / evaluation (same recipe as the deployed model)
# ---------------------------------------------------------------------------
def set_seed(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def augment(xb, torch, gen):
    """Noise at SNR ~ U(0, 30) dB relative to the raw channel power, in the z-scored,
    4:1-aggregated domain: z' = (z + n_agg) / sqrt(1 + s2), n_agg ~ N(0, s2 / 4),
    s2 = 10^(-SNR/10) (white noise averaged over 4 samples)."""
    b = xb.shape[0]
    snr = torch.rand(b, 1, xb.shape[2], device=xb.device, generator=gen) * 30.0
    s2 = 10 ** (-snr / 10)
    noise = torch.randn(xb.shape, device=xb.device, generator=gen) * torch.sqrt(s2 / 4)
    return (xb + noise) / torch.sqrt(1 + s2)


def evaluate(model, x, y, torch, batch=2048, rows=None):
    """x: array or memmap; rows: optional window indices into x (read batch by batch,
    so a split never has to be copied into RAM)."""
    from sklearn.metrics import f1_score, recall_score, roc_auc_score
    model.eval()
    logits = []
    n = len(rows) if rows is not None else len(x)
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = x[rows[i:i + batch]] if rows is not None else x[i:i + batch]
            logits.append(model(torch.from_numpy(np.ascontiguousarray(xb)).cuda()).float().cpu())
    logits = torch.cat(logits).numpy()
    pred = logits.argmax(1)
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    return {"windows": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "recall_per_class": [float(v) for v in recall_score(y, pred, average=None, labels=range(4))],
            "macro_auroc": float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))}, pred


def cmd_train(args):
    import torch
    import torch.nn as nn
    from src.models.transformer_baseline import TransformerBaseline
    meta = dict(np.load(CACHE / "meta.npz"))
    x_all = np.load(CACHE / "x.npy", mmap_mode="r")
    file_of = meta["file_of"]
    assign = file_split(args.split, meta)
    win_tag = assign[file_of]
    y_all = meta["file_label"][file_of]
    sel = {t: np.where(win_tag == t)[0] for t in ("train", "val", "test")}
    files = {t: int(sum(assign == t)) for t in ("train", "val", "test")}
    print(args.split, "files", files, "windows", {t: len(v) for t, v in sel.items()}, flush=True)
    # Windows stay in the memory-mapped cache and are gathered per batch: copying the
    # training split (2.7 GB) pushed Windows to its commit limit on this PC.
    tr_rows = sel["train"]
    ytr = torch.from_numpy(y_all[tr_rows])
    yva, yte = y_all[sel["val"]], y_all[sel["test"]]

    set_seed(torch)
    model = TransformerBaseline(**MODEL_KW).cuda()
    counts = np.bincount(ytr.numpy(), minlength=4).astype(np.float64)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (4 * counts), dtype=torch.float32).cuda())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    order_gen = torch.Generator().manual_seed(SEED)
    best = (-1.0, 0, None)
    history, stale, t0 = [], 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(ytr), generator=order_gen)
        for k, i in enumerate(range(0, len(perm), BATCH)):
            b = perm[i:i + BATCH].numpy()
            xb = torch.from_numpy(np.ascontiguousarray(x_all[tr_rows[b]])).cuda(non_blocking=True)
            yb = ytr[b].cuda(non_blocking=True)
            if args.split.endswith("_noise") and k % 2 == 0:
                xb = augment(xb, torch, gen)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        val, _ = evaluate(model, x_all, yva, torch, rows=sel["val"])
        history.append({"epoch": epoch, **{"val_" + k: v for k, v in val.items()}})
        print("[%s] epoch %d val_acc %.5f val_macro_f1 %.5f (%.0f s)" % (
            args.split, epoch, val["accuracy"], val["macro_f1"], time.time() - t0), flush=True)
        if val["macro_f1"] > best[0] + 1e-12:
            best, stale = (val["macro_f1"], epoch, copy.deepcopy(model.state_dict())), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best[2])
    test, pred = evaluate(model, x_all, yte, torch, rows=sel["test"])
    out = OUT / args.split
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best[2], "model_kwargs": MODEL_KW, "split": args.split,
                "best_epoch": best[1], "best_val_macro_f1": best[0], "seed": SEED}, out / "best.pt")
    test_files = np.where(assign == "test")[0]
    result = {"split": args.split, "files": files, "windows": {t: int(len(v)) for t, v in sel.items()},
              "best_epoch": best[1], "best_val_macro_f1": best[0], "test": test,
              "seconds": round(time.time() - t0, 1), "history": history,
              "test_conditions": sorted(set(meta["file_cond"][test_files].tolist()))}
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("split", "files", "best_epoch", "test")}, indent=1), flush=True)


def cmd_noise_eval(args):
    """Float models on the EXACT raw-domain noisy test windows of tools/robustness_cmodel.py."""
    import torch
    from src.data.prepare_dataset import normalize_windows_array
    from src.data.token_aggregation import aggregate_mean_float
    from src.models.transformer_baseline import TransformerBaseline
    from tools.robustness_cmodel import perturb
    y = np.loadtxt(REPO / "artifacts" / "results" / "fpga_rawinput_testset_20261109" / "labels.txt", dtype=np.int64)
    models = {}
    for name in args.models:
        ck = torch.load(OUT / name / "best.pt", map_location="cpu", weights_only=False)
        m = TransformerBaseline(**ck["model_kwargs"]).cuda()
        m.load_state_dict(ck["model_state_dict"])
        models[name] = m.eval()
    results = {}
    for cond in args.conditions:
        preds = {n: [] for n in models}
        with gzip.open(RAW_TEST, "rb") as f:
            done = 0
            while done < len(y):
                k = min(512, len(y) - done)
                w = np.frombuffer(f.read(k * 32768), dtype="<i4").astype(np.int64).reshape(k, 1024, 8)
                xin = perturb(w, done, cond).astype(np.float32)
                xin = aggregate_mean_float(normalize_windows_array(xin), 4, output_dtype=np.float32)
                with torch.no_grad():
                    xb = torch.from_numpy(xin).cuda()
                    for n, m in models.items():
                        preds[n].append(m(xb).argmax(1).cpu().numpy())
                done += k
        from sklearn.metrics import f1_score, recall_score
        for n in models:
            p = np.concatenate(preds[n])
            results.setdefault(n, {})[cond] = {
                "accuracy": round(float((p == y).mean()), 6),
                "macro_f1": round(float(f1_score(y, p, average="macro")), 6),
                "recall_per_class": [round(float(v), 4) for v in recall_score(y, p, average=None, labels=range(4))]}
        print(cond, {n: results[n][cond]["accuracy"] for n in models}, flush=True)
    (OUT / "noise_eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    t = sub.add_parser("train")
    t.add_argument("--split", required=True, choices=["random", "sev_mid", "sev_max", "speed_mid", "random_noise"])
    n = sub.add_parser("noise-eval")
    n.add_argument("--models", nargs="+", required=True)
    n.add_argument("--conditions", nargs="+", default=["clean", "snr20", "snr10", "snr5", "snr0", "snr-5"])
    args = ap.parse_args()
    {"cache": cmd_cache, "train": cmd_train, "noise-eval": cmd_noise_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
