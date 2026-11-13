"""F00: freeze the identity of the canonical baseline before research variants.

Writes artifacts/research/F00_<date>/frozen_baseline_manifest.json with SHA-256
of every canonical input a research variant is compared against (sources, float
checkpoint, INT10 export and scales, HLS sources, golden sets, deployed overlays,
raw test input), the canonical file split (per-split file-list hashes), the
calibration subset of the INT10 export, tool versions and the timing boundary.
Large binaries already hashed in artifacts/manifests/artifact_manifest.json are
taken from there (not re-read). Nothing is modified.

    python -m tools.research.freeze_baseline --out artifacts/research/F00_20261113
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SOURCES = [
    "src/config.py", "src/data/load_data.py", "src/data/prepare_dataset.py",
    "src/data/make_dataloaders.py", "src/data/token_aggregation.py",
    "src/models/transformer_baseline.py", "src/quantize/integer_model.py",
    "src/quantize/integer_ops.py", "src/quantize/evaluate_integer_bits.py",
    "src/quantize/export_hls_seq256_int10.py", "src/quantize/onchip_preprocess.py",
    "host/temporal_vote.py", "host/confidence.py", "pynq/fault_classifier.py",
    "tools/board/fpga_live_runner.py", "tools/board/fpga_rt_acceptance.py",
    "tools/board/temp_guard.sh", "tools/robustness_cmodel.py",
]
MODEL = [
    "artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/scratch/best.pt",
    "artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/summary.json",
    "artifacts/quantized/f_seq256_int10/integer_config.json",
    "hls/f1_int10_seq256/model_manifest.json",
    "artifacts/preprocessing/raw_input_contract_v1.json",
]
HLS_GLOBS = ["hls/f1_int10_seq256/vitis/*.h", "hls/f1_int10_seq256/vitis/transformer_top_seq256_raw1024_*.cpp",
             "hls/f1_int10_seq256/gpp_csim/batch_logits.cpp"]
GOLDEN = ["hls/f1_int10_seq256/golden_raw1024_argmax/golden_in.txt",
          "hls/f1_int10_seq256/golden_raw1024_argmax/golden_expected.txt",
          "hls/f1_int10_seq256/golden_ingress1024/golden_in.txt",
          "hls/f1_int10_seq256/golden_ingress1024/golden_expected.txt"]
REFERENCE_OUTPUTS = [
    "artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt",
    "artifacts/results/fpga_rawinput_testset_20261109/labels.txt",
]
# Hashes taken from the artifact manifest (multi-GB or board binaries).
FROM_MANIFEST_PREFIXES = [
    "artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz",
    "artifacts/deploy/seq256_raw_argmax_z7020_20261109/overlay/design_1_wrapper.bit",
    "artifacts/deploy/seq256_vote_z7020_20261110/overlay/design_1_wrapper.bit",
    "artifacts/results/seq256_dma_raw_argmax_z7020_20261109/bitstream/design_1.hwh",
    "artifacts/results/seq256_dma_vote_z7020_20261110/bitstream/design_1.hwh",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(rel: str) -> dict:
    p = REPO / rel
    return {"path": rel, "present": p.exists(),
            "sha256": sha256(p) if p.exists() else None, "size_bytes": p.stat().st_size if p.exists() else None}


def split_identity() -> dict:
    """Canonical split (split_file_index) and the INT10 export's calibration files."""
    from src.config import DATA_RAW_DIR, LABEL_MAP
    from src.data.load_data import build_file_index
    from src.data.make_dataloaders import split_file_index
    root = Path(DATA_RAW_DIR).resolve()
    idx = build_file_index(Path(DATA_RAW_DIR))
    idx = idx[idx["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)
    tr, va, te = split_file_index(idx)
    rel = lambda df: [Path(p).resolve().relative_to(root).as_posix() for p in df["filepath"]]
    out = {}
    for name, df in (("train", tr), ("val", va), ("test", te)):
        r = rel(df)
        out[name] = {"files": len(r), "sha256_sorted_relpaths": hashlib.sha256("\n".join(sorted(r)).encode()).hexdigest()}
    # export_hls_seq256_int10._build_calibration_subset: train_df.sample(n=20, random_state=42)
    calib = tr.sample(n=20, random_state=42).reset_index(drop=True)
    cr = rel(calib)
    out["calibration_int10_export"] = {"files": len(cr), "relpaths_in_sample_order": cr,
                                       "rule": "train split, DataFrame.sample(n=20, random_state=42)"}
    out["validation_files_relpaths"] = sorted(rel(va))
    return out


def versions() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0]
        except Exception as exc:  # tool missing is recorded, not fatal
            return "unavailable: %s" % exc.__class__.__name__
    import numpy, torch, sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform(),
            "gpp": run(["g++", "--version"]),
            "vivado_vitis_hls": "2020.2 (G:/Xilinx/Vivado/2020.2, G:/Xilinx/Vitis_HLS/2020.2)",
            "board": "MYIR Z-Turn XC7Z020 (xc7z020clg400-1), PYNQ 2.4 image, Linux 4.14.0-xilinx SMP PREEMPT, Python 3.6",
            "git_head": run(["git", "-C", str(REPO), "rev-parse", "HEAD"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((REPO / "artifacts/manifests/artifact_manifest.json").read_text())
    by_path = {a["path"]: a for a in manifest["artifacts"]}
    hls = sorted({p.relative_to(REPO).as_posix() for g in HLS_GLOBS for p in REPO.glob(g)})
    frozen = {
        "id": "F00", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Identity of the canonical baseline that every research variant is compared against. "
                   "Canonical artifacts are read-only for research work.",
        "canonical_contract": {
            "classes": ["normal", "imbalance", "misalignment", "bearing_fault"],
            "channels": 8, "raw_window_samples": 1024, "sample_rate_hz": 50000, "window_period_ms": 20.48,
            "tokens_after_on_chip_aggregation": 256, "aggregation": "exact 4:1 integer mean (aggregate_round_div)",
            "numeric": "INT10 weights/activations/attention probabilities (signed, qmax 511), Q40 requant",
            "raw_input": "signed 24-bit ADC codes, per-window integer z-score inside the IP (onchip_preprocess.h)",
            "decision": "argmax in FPGA logic (pred_class register); optional temporal vote k in 1..15 (vote overlay)",
        },
        "timing_boundary": "prepared RAM window released every 20.48 ms on CPU1 (SCHED_FIFO 50) -> copy to DMA buffer"
                           " -> MM2S -> core -> S2MM + class register read; response = done - release; acquisition,"
                           " SSH/TCP transport and GUI are separate boundaries",
        "sources": [entry(p) for p in SOURCES],
        "model_and_export": [entry(p) for p in MODEL],
        "hls_sources": [entry(p) for p in hls],
        "golden": [entry(p) for p in GOLDEN],
        "reference_outputs": [entry(p) for p in REFERENCE_OUTPUTS],
        "large_binaries_from_artifact_manifest": [by_path.get(p, {"path": p, "present": False}) for p in FROM_MANIFEST_PREFIXES],
        "split": split_identity(),
        "versions": versions(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frozen_baseline_manifest.json").write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    missing = [e["path"] for k in ("sources", "model_and_export", "golden", "reference_outputs") for e in frozen[k] if not e["present"]]
    print("wrote", args.out / "frozen_baseline_manifest.json", "missing:", missing)


if __name__ == "__main__":
    main()
