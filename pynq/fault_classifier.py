"""
PYNQ-side driver for the F_1layer @ int10 fault-classification core.

Two interchangeable backends behind one `FaultClassifier.infer(window)` API:

  * HardwareBackend -- loads the synthesized overlay (.bit + .hwh), streams one
    quantized window through an AXI-DMA to the HLS IP and reads back the 4 int
    logits. Fill in the IP/DMA names after synthesis (see IP CONTRACT below).
  * NumpyBackend    -- runs the exported int10 reference (hls/f1_int10/) in pure
    NumPy, so the notebook / UART demo runs end-to-end on ANY machine with no
    board. This is the SAME arithmetic the hardware must reproduce (verified
    bit-exact by hls/f1_int10/verify_export.py), so it is a faithful stand-in.

The board never re-defines the class map; label ids come from
host.frame_protocol.LABELS (== src.config.LABEL_MAP).

IP CONTRACT (what the Vitis HLS export must expose, matches transformer.h):
  - input : SEQ*CH = 1024*8 int codes, row-major [seq, channel], quantized with
            input_scale from hls/f1_int10/model_manifest.json (code = round(x/scale),
            clamped to +/-511);
  - output: NUM_CLASSES = 4 int logits; prediction = argmax.
  - transport assumed here: AXI-Stream in/out via one axi_dma; a control bit to
    start and an ap_done to signal completion. Adjust to your generated design.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parents[1]
HLS_DIR     = REPO_ROOT / "hls" / "f1_int10"
MANIFEST    = HLS_DIR / "model_manifest.json"

SEQ, CH, NUM_CLASSES = 1024, 8, 4


def _load_manifest():
    import json
    return json.loads(MANIFEST.read_text())


def quantize_window(window: np.ndarray, input_scale: float, qmax: int = 511) -> np.ndarray:
    """Float (SEQ,CH) normalized window -> int codes the core consumes."""
    if window.shape != (SEQ, CH):
        raise ValueError(f"expected window {(SEQ, CH)}, got {window.shape}")
    codes = np.round(window.astype(np.float64) / input_scale)
    return np.clip(codes, -qmax, qmax).astype(np.int64)


# ---------------------------------------------------------------------------
# NumPy reference backend (reuses the verified export mirror)
# ---------------------------------------------------------------------------
class NumpyBackend:
    def __init__(self):
        # load hls/f1_int10/verify_export.py by path (not importable as a package)
        spec = importlib.util.spec_from_file_location(
            "hls_f1_int10_verify", HLS_DIR / "verify_export.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._mod = mod
        self._params = mod.P()

    def infer_codes(self, in_codes: np.ndarray) -> np.ndarray:
        """int-code window (SEQ,CH) -> 4 int logits."""
        return self._mod.forward(in_codes, self._params)


# ---------------------------------------------------------------------------
# Hardware backend (real board; needs the pynq runtime + a synthesized overlay)
# ---------------------------------------------------------------------------
class HardwareBackend:
    def __init__(self, bitstream: str, dma_name: str = "axi_dma_0"):
        from pynq import Overlay, allocate  # only available on the board
        self.overlay = Overlay(bitstream)
        self.dma = getattr(self.overlay, dma_name)
        self._in = allocate(shape=(SEQ * CH,), dtype=np.int32)
        self._out = allocate(shape=(NUM_CLASSES,), dtype=np.int32)

    def infer_codes(self, in_codes: np.ndarray) -> np.ndarray:
        self._in[:] = in_codes.astype(np.int32).ravel()
        self.dma.sendchannel.transfer(self._in)
        self.dma.recvchannel.transfer(self._out)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        return np.array(self._out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Front end
# ---------------------------------------------------------------------------
class FaultClassifier:
    def __init__(self, backend=None, bitstream: str | None = None):
        self.manifest = _load_manifest()
        self.input_scale = float(self.manifest["input_scale"])
        if backend is not None:
            self.backend = backend
        elif bitstream is not None:
            self.backend = HardwareBackend(bitstream)
        else:
            self.backend = NumpyBackend()

    def infer(self, window: np.ndarray):
        """Normalized float window (SEQ,CH) -> (class_id, logits[4])."""
        codes = quantize_window(window, self.input_scale)
        logits = np.asarray(self.backend.infer_codes(codes), dtype=np.int64)
        return int(np.argmax(logits)), logits

    def infer_from_codes(self, in_codes: np.ndarray):
        logits = np.asarray(self.backend.infer_codes(in_codes), dtype=np.int64)
        return int(np.argmax(logits)), logits
