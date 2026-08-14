"""
PYNQ Lab Demo — Transformer fault classifier running in pure NumPy.

Runs the trained model over the 16 golden test windows and prints results
in a clear, presentable format. Designed to run on the PYNQ board ARM
processor with only NumPy available (no PyTorch, no GPU).

On the PYNQ board, copy these files to the SD card:
    numpy_weights.npz          (from artifacts/pynq/)
    golden_vectors.npz         (from artifacts/golden/)
    demo.py                    (this file)

Then run:
    python demo.py

Or, on the dev machine:
    python -m src.pynq.demo
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution: works both on PYNQ (flat files) and from project root
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent   # src/pynq -> src -> project root

_WEIGHTS = _PROJECT_ROOT / "artifacts" / "pynq" / "numpy_weights.npz"
_GOLDEN  = _PROJECT_ROOT / "artifacts" / "golden" / "golden_vectors.npz"

# When running as a flat script on PYNQ, fall back to current directory
if not _WEIGHTS.exists():
    _WEIGHTS = Path("numpy_weights.npz")
if not _GOLDEN.exists():
    _GOLDEN = Path("golden_vectors.npz")

CLASS_NAMES = {0: "normal", 1: "imbalance", 2: "misalignment", 3: "bearing_fault"}
CLASS_COLORS = {0: "", 1: "", 2: "", 3: ""}   # extended later if ANSI supported


def _bar(prob: float, width: int = 20) -> str:
    filled = int(round(prob * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def run_demo(weights_path: Path = _WEIGHTS, golden_path: Path = _GOLDEN) -> None:
    print("=" * 62)
    print("  FPGA Fault Diagnosis — PYNQ Inference Demo")
    print("  Model: Transformer (pure NumPy, ARM-compatible)")
    print("=" * 62)
    print()

    # -- Load model ------------------------------------------------------------
    try:
        # Try importing from installed package location first
        from src.pynq.numpy_inference import NumpyTransformer
    except ModuleNotFoundError:
        # Flat-file import when running directly on PYNQ
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "numpy_inference", Path(__file__).parent / "numpy_inference.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        NumpyTransformer = mod.NumpyTransformer

    print(f"Loading model weights from: {weights_path}")
    t0 = time.perf_counter()
    model = NumpyTransformer.load(weights_path)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_time:.1f} ms")
    print()

    # -- Load test data --------------------------------------------------------
    print(f"Loading golden vectors from: {golden_path}")
    golden = np.load(str(golden_path))
    raw_inputs     = golden["raw_input"]      # [16, 1024, 8]
    true_labels    = golden["label_id"]       # [16]
    n = raw_inputs.shape[0]
    print(f"  {n} test windows loaded (4 per class)")
    print()

    # -- Inference loop --------------------------------------------------------
    print("-" * 62)
    print(f"  {'#':>3}  {'True class':<16} {'Predicted':<16}  Conf   Time")
    print("-" * 62)

    correct = 0
    times_ms = []

    for i in range(n):
        t_start = time.perf_counter()
        class_id, label, probs, _ = model.predict_raw(raw_inputs[i])
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        times_ms.append(elapsed_ms)

        true_label = CLASS_NAMES[int(true_labels[i])]
        conf = float(probs[class_id])
        ok = class_id == int(true_labels[i])
        if ok:
            correct += 1
        mark = "OK" if ok else "!!"

        print(f"  {i:>3}  {true_label:<16} {label:<16}  {conf:.1%}  {elapsed_ms:6.1f} ms  {mark}")

    print("-" * 62)
    print()

    # -- Summary ---------------------------------------------------------------
    acc = correct / n * 100
    avg_ms = float(np.mean(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))

    print(f"  Accuracy    : {correct}/{n}  ({acc:.1f}%)")
    print(f"  Latency     : avg {avg_ms:.1f} ms  |  min {min_ms:.1f} ms  |  max {max_ms:.1f} ms")
    print(f"  Window size : 1024 samples x 8 channels")
    print()

    # -- Per-class breakdown ---------------------------------------------------
    print("  Per-class results:")
    for cid, cname in CLASS_NAMES.items():
        mask = true_labels == cid
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        n_class = len(indices)
        n_correct = sum(
            1 for idx in indices
            if int(np.argmax(model.predict_raw(raw_inputs[idx])[2])) == cid
        )
        print(f"    {cname:<16} {n_correct}/{n_class}")

    print()
    print("=" * 62)
    status = "PASS" if correct == n else f"FAIL ({n - correct} wrong)"
    print(f"  Status: {status}")
    print("=" * 62)


if __name__ == "__main__":
    run_demo()
