"""
Stage 6c — Weight export for hardware.

Exports the trained Transformer's parameters in the int16 symmetric fixed-point
format selected in Stage 6a (int16 gave zero accuracy drop; per-tensor scaling
is used because it is the cheapest for hardware). The export is what the Python
fixed-point simulator (Stage 6b) and the HLS/RTL core (Stage 8) both consume, so
the numeric contract lives in one place.

What is exported and how
------------------------
- 2-D weight matrices (Linear weights + attention ``in_proj_weight``):
  quantized to int16 with one per-tensor ``scale`` each. Stored as integer codes
  plus the scale; the real value is ``code * scale``.
- 1-D parameters (biases, LayerNorm affine gamma/beta): kept as float32. These
  are kept at higher precision in hardware (standard practice) and their final
  fixed-point form depends on the activation scales chosen in Stage 6b.
- Positional-encoding buffer (fixed sinusoidal constant): kept as float32 for
  now; it is a constant table that can be quantized later without retraining.

Outputs (under artifacts/quantized/):
- ``transformer_int16_weights.npz`` — compact arrays for the Python simulator.
- ``transformer_int16_weights.json`` — human-readable manifest: shapes, dtypes,
  scales, bit widths, class map, normalization contract, model config.

Nothing here retrains or changes the model; it only reads the frozen checkpoint.

Run
---
    python -m src.quantize.export_weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import (
    PROJECT_ROOT,
    LABEL_MAP,
    NORMALIZATION_METHOD,
    NORMALIZATION_SCOPE,
    EPSILON,
)
from src.models.transformer_baseline import TransformerBaseline
from src.quantize.fixed_point import QuantizedTensor

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "20260615_223905_fpga_fault_final_t4"
    / "checkpoints"
    / "transformer_best.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "quantized"

WEIGHT_BITS = 16  # selected in Stage 6a


def load_checkpoint(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt:
        raise ValueError("Checkpoint has no embedded model_config.")
    cfg = ckpt["model_config"]
    model = TransformerBaseline(
        input_dim=cfg["input_dim"],
        seq_len=cfg["seq_len"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def export(checkpoint_path: Path, output_dir: Path):
    model, cfg = load_checkpoint(checkpoint_path)
    state = model.state_dict()

    npz_arrays: dict[str, np.ndarray] = {}
    manifest_params = {}

    for name, tensor in state.items():
        arr = tensor.detach().cpu()

        if arr.dim() == 2:
            # Matmul weight -> int16 per-tensor symmetric.
            qt = QuantizedTensor.from_tensor(arr, WEIGHT_BITS, per_channel=False)
            codes = qt.int_codes.to(torch.int16).numpy()
            scale = float(qt.scale.item())
            npz_arrays[f"{name}__int16"] = codes
            npz_arrays[f"{name}__scale"] = np.array([scale], dtype=np.float64)
            manifest_params[name] = {
                "role": "weight_matrix",
                "shape": list(arr.shape),
                "dtype": "int16",
                "quant": "symmetric_per_tensor",
                "n_bits": WEIGHT_BITS,
                "scale": scale,
                "value_formula": "code * scale",
            }
        else:
            # 1-D affine/bias params and the fixed PE buffer -> float32.
            f = arr.to(torch.float32).numpy()
            npz_arrays[f"{name}__float32"] = f
            manifest_params[name] = {
                "role": ("pe_buffer" if name.endswith("pe") else "vector_float"),
                "shape": list(arr.shape),
                "dtype": "float32",
                "quant": "none",
                "note": "kept high precision; fixed-point form set in Stage 6b",
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "transformer_int16_weights.npz"
    np.savez_compressed(npz_path, **npz_arrays)

    total_weight_codes = sum(
        int(np.prod(p["shape"])) for p in manifest_params.values()
        if p["dtype"] == "int16"
    )
    total_float_vals = sum(
        int(np.prod(p["shape"])) for p in manifest_params.values()
        if p["dtype"] == "float32"
    )

    manifest = {
        "stage": "6c_weight_export",
        "checkpoint": str(checkpoint_path),
        "weight_bits": WEIGHT_BITS,
        "weight_quantization": "symmetric_per_tensor_int16",
        "value_formula": "real = int_code * scale (per-tensor)",
        "model_config": cfg,
        "class_map": LABEL_MAP,
        "normalization_contract": {
            "scope": NORMALIZATION_SCOPE,
            "method": NORMALIZATION_METHOD,
            "epsilon": EPSILON,
            "note": (
                "per_window z-score: hardware must compute mean/std over each "
                "1024-sample window per channel before the model, matching training."
            ),
        },
        "memory_summary": {
            "int16_weight_values": total_weight_codes,
            "int16_weight_bytes": total_weight_codes * 2,
            "float32_values": total_float_vals,
            "float32_bytes": total_float_vals * 4,
        },
        "parameters": manifest_params,
        "npz_file": npz_path.name,
    }
    json_path = output_dir / "transformer_int16_weights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest, npz_path, json_path


def verify_roundtrip(checkpoint_path: Path, npz_path: Path):
    """
    Sanity check: reconstruct each int16 weight from codes*scale and report the
    max reconstruction error vs the original float weight. Confirms the export is
    faithful and matches the Stage 6a fake-quant math.
    """
    model, _ = load_checkpoint(checkpoint_path)
    state = model.state_dict()
    data = np.load(npz_path)
    max_err = 0.0
    worst = None
    for name, tensor in state.items():
        if tensor.dim() != 2:
            continue
        codes = data[f"{name}__int16"].astype(np.float64)
        scale = float(data[f"{name}__scale"][0])
        recon = codes * scale
        err = float(np.abs(recon - tensor.numpy()).max())
        if err > max_err:
            max_err, worst = err, name
    return max_err, worst


def main():
    parser = argparse.ArgumentParser(description="Export int16 weights for hardware (Stage 6c).")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    manifest, npz_path, json_path = export(args.checkpoint, args.output_dir)
    max_err, worst = verify_roundtrip(args.checkpoint, npz_path)

    ms = manifest["memory_summary"]
    print("=== Stage 6c weight export ===")
    print(f"checkpoint : {args.checkpoint.name}")
    print(f"weight fmt : symmetric per-tensor int16")
    print(f"int16 weights : {ms['int16_weight_values']} values "
          f"({ms['int16_weight_bytes'] / 1024:.1f} KiB)")
    print(f"float32 params: {ms['float32_values']} values "
          f"({ms['float32_bytes'] / 1024:.1f} KiB)")
    print(f"npz : {npz_path}")
    print(f"json: {json_path}")
    print(f"roundtrip max reconstruction error: {max_err:.3e}  (worst: {worst})")


if __name__ == "__main__":
    main()
