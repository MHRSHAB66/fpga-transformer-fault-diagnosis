"""
Export Transformer weights to a portable NumPy .npz file for PYNQ deployment.

This script runs on the dev machine (requires PyTorch).
The output file is then copied to the PYNQ board's SD card.

Run
---
    python -m src.pynq.export_weights

Output
------
    artifacts/pynq/numpy_weights.npz   (~200 KB, all float32 arrays)
"""

from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT
from src.pynq.numpy_inference import NumpyTransformer

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "20260615_223905_fpga_fault_final_t4"
    / "checkpoints"
    / "transformer_best.pt"
)
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "pynq" / "numpy_weights.npz"


def export(checkpoint_path: Path = DEFAULT_CHECKPOINT,
           output_path: Path = OUTPUT_PATH) -> Path:
    print(f"[export] loading checkpoint: {checkpoint_path}")
    model = NumpyTransformer.from_checkpoint(checkpoint_path)
    model.save(output_path)
    print(f"[export] done — copy {output_path} to the PYNQ board SD card")
    return output_path


if __name__ == "__main__":
    export()
