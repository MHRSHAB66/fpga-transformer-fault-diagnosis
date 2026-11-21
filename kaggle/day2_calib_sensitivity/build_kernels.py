"""Build the Day-2 calibration-sensitivity Kaggle kernels (validation split, sharded).

NEW private CPU kernels; the verified tuning / int10 shard kernels are untouched.
Each kernel embeds src/quantize/calibration_sensitivity.py and the locally
produced, train-only calibration_sets.json, and asserts that the attached
fpga-fault-src dataset matches the local core modules by SHA-256 before running.

    python kaggle/day2_calib_sensitivity/build_kernels.py
    kaggle kernels push -p kaggle/day2_calib_sensitivity/shard0   (x4)
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "artifacts" / "experiments" / "day2_calibration_sensitivity_20261120"
NUM_SHARDS = 4
# SHA-256 of the core modules in the attached fpga-fault-src dataset (2026-11-11
# upload), equal to the local files when calibration_sets.json was produced. The
# 2026-11-20 Day-2 edits to integer_model.py / integer_ops.py are docstring-only,
# so the local hashes of those two files now differ while the code does not.
CORE_SHA = {
    "config.py": "4469ce10a62e065e840d7fa75d017c13d32310034b90e3ca75ca3fef00093cdf",
    "data/load_data.py": "e4e4559d0b60c850860e78effe49d8d1208a6f73a2878f3e3650712c8ec1cade",
    "data/make_dataloaders.py": "2be3100b79f4243ba68c07777b4b7fc7471544002b5f371e292ee43153e6e9e4",
    "data/prepare_dataset.py": "7800f1cf36bb0b5904a7fce5995cc7df7edc66ae06c30288937947319ab10cf2",
    "data/token_aggregation.py": "80c144d80ca3f757407c3e50bb70d300972899e211420eca0b84e11725cb6905",
    "quantize/integer_model.py": "991c3cc4acf1ad2c0418761220afe9dd92742de52aaa906f7a1d93a551888b97",
    "quantize/integer_ops.py": "eb646ab8d4a7655029f12743a35bd3255708f957cc7aa14c5ab0821dc1889c20",
    "quantize/evaluate_integer_bits.py": "bba57237fc80f77fee4bc5c7a133c9f054d6efa80596562bba0480ce572d22e9",
    "quantize/evaluate_integer.py": "98a38bc60483d103829d72cf7abbc797f115ddc0020e00e337fc2161e08db73f",
}

TEMPLATE = r'''"""Day-2 INT10 calibration sensitivity, validation shard {shard}/{num_shards} (generated)."""
import glob, hashlib, json, os, subprocess, sys
from pathlib import Path

SHARD, NUM_SHARDS = {shard}, {num_shards}
MODULE_SRC = {module_src!r}
CALIB_JSON = {calib_json!r}
CORE_SHA = {core_sha!r}
W = Path("/kaggle/working")


def find_raw(root="/kaggle/input"):
    markers = {{"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment"}}
    best = []
    for d, dirs, _ in os.walk(root):
        s = len(markers.intersection(x.lower() for x in dirs))
        if s >= 3:
            best.append((-s, len(d), d))
    return sorted(best)[0][2]


code_root = Path(glob.glob("/kaggle/input/**/src/config.py", recursive=True)[0]).parents[1]
for rel, sha in CORE_SHA.items():
    got = hashlib.sha256((code_root / "src" / rel).read_bytes()).hexdigest()
    assert got == sha, f"attached src differs from the local repository: {{rel}}"
ckpt = [p for p in glob.glob("/kaggle/input/**/best.pt", recursive=True) if "seq256" in p.lower() or "pool4" in p.lower()][0]
(W / "calibration_sensitivity.py").write_text(MODULE_SRC)
(W / "calibration_sets.json").write_text(CALIB_JSON)
# build_file_index only accepts PROJECT_ROOT/data/raw/mafaulda (DATA_RAW_DIR):
# link the read-only input mount there, as fpga-fault-seq256-10class does.
link = W / "data" / "raw" / "mafaulda"
if not link.exists():
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(find_raw(), target_is_directory=True)
env = dict(os.environ, PYTHONPATH=f"{{W}}:{{code_root}}", PROJECT_ROOT=str(W))
cmd = [sys.executable, "-m", "calibration_sensitivity", "evaluate", "--raw-dir", str(link),
       "--checkpoint", ckpt, "--scales", str(W / "calibration_sets.json"), "--split", "val",
       "--shard", str(SHARD), "--num-shards", str(NUM_SHARDS), "--workers", str(os.cpu_count()),
       "--out", str(W / f"val_shard{{SHARD}}.json")]
print("cpu_count", os.cpu_count(), "ckpt", ckpt, flush=True)
subprocess.run(cmd, check=True, cwd=W, env=env)
'''


def main():
    module_src = (ROOT / "src" / "quantize" / "calibration_sensitivity.py").read_text(encoding="utf-8")
    calib_json = (EXP / "calibration_sets.json").read_text(encoding="utf-8")
    core_sha = CORE_SHA
    for s in range(NUM_SHARDS):
        d = HERE / f"shard{s}"
        d.mkdir(exist_ok=True)
        code = f"day2-calib-sens-shard{s}.py"
        (d / code).write_text(TEMPLATE.format(shard=s, num_shards=NUM_SHARDS, module_src=module_src,
                                              calib_json=calib_json, core_sha=core_sha),
                              encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"mehrdadsheikhabbasi/fpga-fault-day2-calib-sens-shard-{s}",
            "title": f"FPGA Fault Day2 Calib Sens Shard {s}",
            "code_file": code, "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["mehrdadsheikhabbasi/fpga-fault-src",
                                "mehrdadsheikhabbasi/fpga-fault-seq256-pool4-ckpt",
                                "vuxuancu/mafaulda-full"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2),
            encoding="utf-8")
    print(f"built {NUM_SHARDS} kernels in {HERE}")


if __name__ == "__main__":
    main()
