"""
Run the three tuning configs that were SKIPPED on Kaggle (time budget) --
E_3layers, C_d48, D_d64_slow -- locally, using the *exact same* pipeline,
model, fixed settings and metric definitions as the original run so the
results are directly comparable to F_1layer / A_baseline / B_wide_ffn.

It reuses the self-contained Kaggle kernel (kaggle/tuning/fpga-fault-tuning.py)
for the model, criterion, training loop and confusion-matrix metrics, and the
project's own ``build_dataloaders`` for the leakage-free file-level split.

Fixed (unchanged) settings: window 1024 · stride 1024 · per-window z-score ·
batch 64 · max 12 epochs · patience 3 · seed 42.

Run (needs the local MAFAULDA data + a CUDA GPU is used if available):
    python -m src.tools.run_skipped_tuning_configs
    python -m src.tools.run_skipped_tuning_configs --smoke   # fast path check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.data.make_dataloaders import build_dataloaders

KERNEL_PATH = PROJECT_ROOT / "kaggle" / "tuning" / "fpga-fault-tuning.py"
EXISTING_REPORT = (PROJECT_ROOT / "artifacts" / "runs" / "tuning_20260616_161108"
                   / "tuning_final_report.json")
TARGETS = ["E_3layers", "C_d48", "D_d64_slow"]


def load_kernel():
    """Import the hyphenated kernel file as a module (defs only; main() guarded)."""
    spec = importlib.util.spec_from_file_location("fpga_fault_tuning", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fpga_fault_tuning"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 config, few files, 1 epoch -- just checks the path works")
    args = ap.parse_args()

    k = load_kernel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "tuning_skipped_smoke" if args.smoke else "tuning_skipped"
    outdir = PROJECT_ROOT / "artifacts" / "runs" / f"{tag}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device} ({gpu})  out={outdir}", flush=True)

    if args.smoke:
        k.MAX_EPOCHS = 1
        targets = ["E_3layers"]
        max_files = 40
    else:
        targets = TARGETS
        max_files = None

    raw = PROJECT_ROOT / "data" / "raw" / "mafaulda"
    print("[info] building dataloaders (leakage-free file-level split)...", flush=True)
    t0 = time.perf_counter()
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir=raw, max_files=max_files, batch_size=k.BATCH_SIZE,
        window_size=k.WINDOW_SIZE, stride=k.STRIDE,
        normalization_scope=k.NORM_SCOPE, normalization_method=k.NORM_METHOD,
    )
    print(f"[info] loaders ready in {(time.perf_counter()-t0)/60:.1f}m  "
          f"train={len(train_loader.dataset):,} val={len(val_loader.dataset):,} "
          f"test={len(test_loader.dataset):,}", flush=True)

    if hasattr(train_loader.dataset, "tensors"):
        train_labels = train_loader.dataset.tensors[1].numpy()
    else:
        train_labels = np.asarray(train_loader.dataset.targets)
    criterion = k.build_criterion(train_labels, device)

    configs = [c for c in k.TUNING_CONFIGS if c["name"] in targets]
    results = []
    for cfg in configs:
        r = k.train_one_config(cfg, train_loader, val_loader, criterion, device, outdir)
        results.append(r)
        # incremental save so a crash/interrupt never loses completed configs
        with open(outdir / "skipped_results.json", "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[done] {r['name']}: val_acc={r['val_acc']*100:.2f}%  "
              f"macro_f2={r['val_macro_f2']:.4f}  params={r['n_params']:,}  "
              f"fault_recall={r['val_fault_recall']*100:.2f}%  "
              f"false_alarm={r['val_false_alarm']*100:.2f}%", flush=True)

    _write_combined_report(k, results, outdir, args.smoke)
    print(f"[all done] results in {outdir}", flush=True)


def _write_combined_report(k, new_results, outdir, smoke):
    """Merge the 3 new configs with the original F/A/B and write JSON + Markdown."""
    prior = []
    if EXISTING_REPORT.exists():
        prior = json.load(open(EXISTING_REPORT)).get("all_val_results", [])

    def row(r):
        return {
            "name": r["name"], "n_params": r["n_params"],
            "val_acc": r["val_acc"], "val_macro_f2": r["val_macro_f2"],
            "val_fault_recall": r["val_fault_recall"],
            "val_false_alarm": r["val_false_alarm"],
            "best_epoch": r.get("best_epoch"), "source": "local_skipped_run",
        }

    combined = list(prior) + [row(r) for r in new_results]
    # rank by (macro_f2, val_acc)
    combined.sort(key=lambda r: (r.get("val_macro_f2") or 0, r.get("val_acc") or 0),
                  reverse=True)
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Skipped tuning configs E_3layers/C_d48/D_d64_slow run locally with the "
                 "same pipeline/settings/metrics as the original Kaggle run; merged with "
                 "F_1layer/A_baseline/B_wide_ffn for a complete 6-config comparison."),
        "fixed_settings": {
            "window_size": k.WINDOW_SIZE, "stride": k.STRIDE,
            "normalization": f"{k.NORM_SCOPE}/{k.NORM_METHOD}",
            "batch_size": k.BATCH_SIZE, "max_epochs": k.MAX_EPOCHS,
            "patience": k.PATIENCE, "seed": k.SEED,
        },
        "smoke": smoke,
        "new_configs": [row(r) for r in new_results],
        "all_six_val_results_ranked": combined,
    }
    with open(outdir / "combined_tuning_comparison.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    lines = ["# Tuning — all six configs (validation set)", "",
             f"_Generated {payload['generated']}._  ",
             "Configs E_3layers / C_d48 / D_d64_slow were run locally with the same "
             "pipeline, settings and metric definitions as the original F/A/B run.", "",
             "| Rank | Config | Params | Val acc | Macro F2 | Fault recall | False alarm |",
             "|-----:|--------|-------:|--------:|---------:|-------------:|------------:|"]
    for i, r in enumerate(combined, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['n_params']:,} | "
            f"{(r['val_acc'] or 0)*100:.2f}% | {r.get('val_macro_f2') or 0:.4f} | "
            f"{(r.get('val_fault_recall') or 0)*100:.2f}% | "
            f"{(r.get('val_false_alarm') or 0)*100:.2f}% |")
    lines += ["", "Fixed: window 1024 · stride 1024 · per-window z-score · batch 64 · "
              "max 12 epochs · patience 3 · seed 42.", ""]
    (outdir / "combined_tuning_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
