"""
Guarded Kaggle runner for the final FPGA fault Transformer training.

Default behavior is SMOKE MODE on a small subset. Full training only starts
when RUN_FULL_TRAINING=1 is set in this file/environment.
"""

import glob
import json
import os
import sys
import traceback

WORKING_DIR = "/kaggle/working"
INPUT_ROOT = "/kaggle/input"

RUN_FULL_TRAINING = os.getenv("RUN_FULL_TRAINING", "1") == "1"  # default: full training
SMOKE_MAX_FILES = int(os.getenv("MAX_FILES", "24"))


def log(message):
    print(message, flush=True)


def detect_mafaulda_root(input_root):
    class_markers = (
        "normal",
        "imbalance",
        "horizontal-misalignment",
        "vertical-misalignment",
        "overhang",
        "underhang",
    )
    candidates = []
    for dirpath, dirnames, _ in os.walk(input_root):
        lowered = {d.lower() for d in dirnames}
        hits = sum(1 for marker in class_markers if marker in lowered)
        if hits >= 3:
            candidates.append((hits, dirpath))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    return candidates[0][1]


def locate_source_root(input_root):
    hits = glob.glob(os.path.join(input_root, "**", "src", "config.py"), recursive=True)
    if not hits:
        raise FileNotFoundError(
            "Could not find src/config.py. Attach mehrdadsheikhabbasi/fpga-fault-src."
        )
    return os.path.dirname(os.path.dirname(hits[0]))


def save_runner_report(report):
    os.makedirs(WORKING_DIR, exist_ok=True)
    with open(os.path.join(WORKING_DIR, "final_runner_report.json"), "w") as f:
        json.dump(report, f, indent=2)


def main():
    report = {
        "passed": False,
        "mode": "full" if RUN_FULL_TRAINING else "smoke",
        "max_files": None if RUN_FULL_TRAINING else SMOKE_MAX_FILES,
        "errors": [],
    }
    try:
        code_root = locate_source_root(INPUT_ROOT)
        raw_dir = detect_mafaulda_root(INPUT_ROOT)
        if raw_dir is None:
            raise FileNotFoundError(
                "Could not find MAFAULDA root. Attach vuxuancu/mafaulda-full."
            )

        os.environ["PROJECT_ROOT"] = WORKING_DIR
        sys.path.insert(0, code_root)

        import torch

        from src.train_transformer_baseline import train

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        report.update(
            {
                "code_root": code_root,
                "dataset_path": raw_dir,
                "cuda_available": bool(cuda_available),
                "gpu_name": gpu_name,
            }
        )
        log(f"[FINAL] mode={report['mode']} max_files={report['max_files']}")
        log(f"[FINAL] code_root={code_root}")
        log(f"[FINAL] dataset_path={raw_dir}")
        log(f"[FINAL] cuda_available={cuda_available} gpu_name={gpu_name}")

        if not RUN_FULL_TRAINING:
            log("[FINAL] RUN_FULL_TRAINING is not set. Running small smoke subset only.")

        summary = train(
            data_dir=raw_dir,
            max_files=None if RUN_FULL_TRAINING else SMOKE_MAX_FILES,
            run_name="fpga_fault_final_t4" if RUN_FULL_TRAINING else "fpga_fault_final_smoke",
            num_epochs=30 if RUN_FULL_TRAINING else 1,
            batch_size=64,
            window_size=1024,
            stride=1024,
            normalization_scope="per_window",
            normalization_method="zscore",
        )
        report["training_summary"] = summary
        report["passed"] = True
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        save_runner_report(report)
        raise

    save_runner_report(report)
    log(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
