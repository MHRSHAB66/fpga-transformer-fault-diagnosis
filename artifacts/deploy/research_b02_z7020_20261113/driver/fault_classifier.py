"""
PYNQ-side driver for the SEQ=256 INT10 fault-classification core, integrated as a
Zynq PS + AXI-DMA + transformer_top HLS IP system (provisional/generic XC7Z020).

This single file is used in BOTH layouts and resolves its resources relative to
its own location:

  * Repository layout : pynq/fault_classifier.py
      manifest  -> ../hls/f1_int10_seq256/model_manifest.json
      golden    -> ../hls/f1_int10_seq256/golden_ingress1024/
  * Deployment bundle : artifacts/deploy/seq256_z7020_pynq/driver/fault_classifier.py
      manifest  -> ../config/model_manifest.json
      golden    -> ../golden/

The HARDWARE path is fully standalone: it needs only numpy, the vendor `pynq`
runtime, and the deployment bundle's overlay + config. It does NOT import torch,
`src`, a checkpoint, or the dataset. The NumPy reference backend is a separate,
REPOSITORY-ONLY block (below the marker) and is never imported or initialised by
the hardware path.

Vendor-package note: on the board, run `python driver/run_demo.py` from inside the
bundle's `driver/` folder so `from pynq import Overlay` imports the real vendor
PYNQ runtime, not this project's `pynq/` folder. See pynq/README.md.

Python compatibility: this hardware-path module targets Python 3.6+ (the MYIR
Z-Turn PYNQ 2.4 image runs Python 3.6.5). It therefore avoids PEP 563/604 syntax
and allocates DMA buffers through `_dma_allocate`, which supports both
`pynq.allocate` (>=2.5) and `Xlnk().cma_array` (2.4).

Class map (never redefined here): 0 normal, 1 imbalance, 2 misalignment,
3 bearing_fault -- from host.frame_protocol.LABELS (== src.config.LABEL_MAP).
"""
from typing import Optional

import json
from pathlib import Path

import numpy as np

_DRIVER_DIR = Path(__file__).resolve().parent

# --- model / transport constants (match the generated design_1.hwh) ----------
SEQ_IN, CH, NUM_CLASSES = 1024, 8, 4      # host-facing acquisition window
SEQ_MODEL, AGG_FACTOR = 256, 4            # aggregated token count / factor
ACT_QMAX = 511                            # INT10 clamp
INPUT_WORDS = SEQ_IN * CH                 # 8192 AXI-stream words in
OUTPUT_WORDS = NUM_CLASSES                # 4 AXI-stream words out

# Verified from design_1.hwh (not comments): instance names + AXI-Lite bases.
DMA_NAME = "axi_dma_0"
IP_NAME = "transformer_top_0"
DMA_AXILITE_BASE = 0x41E00000
IP_CTRL_BASE = 0x40000000
# transformer_top ap_ctrl_hs control register (s_axilite bundle `ctrl`, offset 0):
AP_CTRL_OFFSET = 0x00
AP_START = 0x01            # bit0
AP_AUTO_RESTART = 0x81     # bit0 ap_start | bit7 auto_restart
# Argmax-variant IP only: AXI-Lite output register holding the FPGA class id
# (lowest class id wins ties, == numpy.argmax). Absent on the deployed overlay.
CLASS_REG = "pred_class"
# Vote-variant IP only: AXI-Lite input vote_k and outputs vote_class / vote_count
# (temporal majority vote over the last k window classes, computed in FPGA logic).
VOTE_K_REG, VOTE_CLASS_REG, VOTE_COUNT_REG = "vote_k", "vote_class", "vote_count"
# Research integrity overlay only (B02): AXI-Lite output integrity_status
# (bit0 input-buffer signature, bit1 duplicated classifier tail, bit2 projection range).
INTEGRITY_REG = "integrity_status"

MODEL_CFG = {
    "input_dim": 8, "seq_len": 256, "num_classes": 4,
    "d_model": 32, "nhead": 4, "num_layers": 1,
    "dim_feedforward": 64, "dropout": 0.1,
}


# ---------------------------------------------------------------------------
# Self-locating resource resolution (bundle first, then repository)
# ---------------------------------------------------------------------------
def _resolve(*relative_candidates: Path) -> Path:
    for candidate in relative_candidates:
        if candidate.exists():
            return candidate
    looked = "; ".join(str(c) for c in relative_candidates)
    raise FileNotFoundError(
        f"required deployment file not found. Looked in: {looked}")


def manifest_path() -> Path:
    return _resolve(
        _DRIVER_DIR.parent / "config" / "model_manifest.json",                     # bundle
        _DRIVER_DIR.parent / "hls" / "f1_int10_seq256" / "model_manifest.json",    # repo
    )


def golden_dir() -> Path:
    return _resolve(
        _DRIVER_DIR.parent / "golden",                                             # bundle
        _DRIVER_DIR.parent / "hls" / "f1_int10_seq256" / "golden_ingress1024",     # repo
    )


def load_input_scale() -> float:
    """INT10 input scale, read from the bundle/repo model manifest."""
    return float(json.loads(manifest_path().read_text())["input_scale"])


def quantize_window(window: np.ndarray, input_scale: float, qmax: int = ACT_QMAX) -> np.ndarray:
    """Normalized (1024, 8) window -> INT10 codes the hardware ingress consumes.

    Returns int64 [1024, 8]; the hardware/NumPy backends do the 4:1 aggregation.
    """
    if window.shape != (SEQ_IN, CH):
        raise ValueError(f"expected window {(SEQ_IN, CH)}, got {window.shape}")
    codes = np.round(np.asarray(window, dtype=np.float64) / input_scale)
    return np.clip(codes, -qmax, qmax).astype(np.int64)


# ---------------------------------------------------------------------------
# Physically-contiguous DMA buffer allocator, compatible across PYNQ versions.
# ---------------------------------------------------------------------------
_XLNK = None


def _reg_offset(registers: dict, name: str) -> Optional[int]:
    """Address offset of an HWH register by exact name, or None if absent."""
    reg = registers.get(name)
    if not reg:
        return None
    off = reg.get("address_offset")
    return int(off) if off is not None else None


def _dma_allocate(shape, dtype):
    """Allocate a physically-contiguous DMA buffer on any supported PYNQ.

    PYNQ >= 2.5 exposes ``pynq.allocate``; PYNQ 2.4 (the MYIR Z-Turn image) uses
    ``Xlnk().cma_array``. A single Xlnk instance is reused. Raises a clear error
    if no allocator is available; it NEVER falls back to a plain numpy array,
    which would not be DMA-capable (it has no physical address).
    """
    try:
        from pynq import allocate            # PYNQ >= 2.5
    except ImportError:
        allocate = None
    if allocate is not None:
        return allocate(shape=shape, dtype=dtype)
    try:
        from pynq import Xlnk                 # PYNQ 2.4 fallback
    except ImportError as exc:
        raise RuntimeError(
            "no PYNQ DMA allocator available: neither pynq.allocate (>=2.5) nor "
            "pynq.Xlnk (2.4) is importable; cannot create a DMA buffer") from exc
    global _XLNK
    if _XLNK is None:
        _XLNK = Xlnk()
    return _XLNK.cma_array(shape=shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Hardware backend (real board). Standalone: numpy + vendor pynq only.
# ---------------------------------------------------------------------------
class HardwareBackend:
    """Streams one 1024x8 INT10 window through axi_dma_0 to transformer_top_0.

    The IP performs the 4:1 integer aggregation (1024->256) internally, so the
    host sends all 8192 codes and never aggregates. Confidence is NA.

    Control: the transformer_top core uses ap_ctrl_hs. With `auto_restart=True`
    (default) bit7 is set once so the core re-arms itself for each streamed
    window; set `auto_restart=False` to arm ap_start per call is NOT done here
    because the DMA-driven stream design expects a free-running core.

    Blocking note: PYNQ's `sendchannel.wait()` / `recvchannel.wait()` block until
    the DMA descriptors complete. Older PYNQ DMA APIs expose no portable timeout;
    if the core never asserts ap_done (e.g. wrong word count) these waits can
    block. Optionally pass `wait_timeout` (seconds) which is used only if the
    installed PYNQ `wait` accepts a timeout argument.
    """

    def __init__(self, bitstream: str, dma_name: str = DMA_NAME,
                 ip_name: str = IP_NAME, auto_restart: bool = True,
                 wait_timeout: Optional[float] = None):
        from pynq import Overlay  # vendor runtime; board-only

        bit = Path(bitstream)
        if not bit.exists():
            raise FileNotFoundError(f"bitstream not found: {bit}")
        # PYNQ discovers metadata from the BIT basename, i.e. <stem>.hwh.
        expected_hwh = bit.with_suffix(".hwh")
        if not expected_hwh.exists():
            raise FileNotFoundError(
                f"overlay metadata missing: expected {expected_hwh.name} next to "
                f"{bit.name} (PYNQ resolves the HWH by the BIT basename)")

        self.auto_restart = auto_restart
        self.wait_timeout = wait_timeout
        self.overlay = Overlay(str(bit))

        # Verify the instances actually exist in the generated design rather than
        # trusting comments; fail before any DMA is armed.
        ip_dict = dict(getattr(self.overlay, "ip_dict", {}) or {})
        if dma_name not in ip_dict:
            raise RuntimeError(
                f"DMA instance '{dma_name}' not found in overlay.ip_dict "
                f"({sorted(ip_dict)}); cannot start DMA")
        if ip_name not in ip_dict:
            raise RuntimeError(
                f"Transformer IP '{ip_name}' not found in overlay.ip_dict "
                f"({sorted(ip_dict)}); refusing to run without the classifier core")

        self.dma = getattr(self.overlay, dma_name)
        self.ip = getattr(self.overlay, ip_name)
        if self.ip is None:
            raise RuntimeError(f"Transformer IP '{ip_name}' resolved to None")

        # Optional on-FPGA class decision (argmax variant of the IP): an extra
        # AXI-Lite output register `pred_class` (+ `pred_class_ctrl` ap_vld,
        # clear-on-read). Offsets are taken from the HWH, never hard-coded. The
        # deployed dma16 overlay has no such register -> legacy path (None).
        regs = dict(ip_dict[ip_name].get("registers", {}) or {})
        self.class_reg_offset = _reg_offset(regs, CLASS_REG)
        self.class_vld_offset = _reg_offset(regs, CLASS_REG + "_ctrl")
        self.vote_k_offset = _reg_offset(regs, VOTE_K_REG)
        self.vote_class_offset = _reg_offset(regs, VOTE_CLASS_REG)
        self.vote_class_vld_offset = _reg_offset(regs, VOTE_CLASS_REG + "_ctrl")
        self.vote_count_offset = _reg_offset(regs, VOTE_COUNT_REG)
        self.vote_count_vld_offset = _reg_offset(regs, VOTE_COUNT_REG + "_ctrl")
        self.integrity_offset = _reg_offset(regs, INTEGRITY_REG)
        self.integrity_vld_offset = _reg_offset(regs, INTEGRITY_REG + "_ctrl")

        # Start the core (ap_ctrl_hs). auto_restart keeps it armed per window.
        self.ip.write(AP_CTRL_OFFSET, AP_AUTO_RESTART if auto_restart else AP_START)

        self._in = _dma_allocate(shape=(INPUT_WORDS,), dtype=np.int32)
        self._out = _dma_allocate(shape=(OUTPUT_WORDS,), dtype=np.int32)

    @property
    def has_hw_class(self) -> bool:
        return self.class_reg_offset is not None

    @property
    def has_vote(self) -> bool:
        return None not in (self.vote_k_offset, self.vote_class_offset, self.vote_count_offset)

    def set_vote_k(self, k: int) -> None:
        """Number of recent windows the FPGA votes over (the IP clamps to 1..15)."""
        if not self.has_vote:
            raise RuntimeError("overlay has no temporal vote registers")
        self.ip.write(self.vote_k_offset, int(k))

    def read_vote(self):
        """(vote_class, vote_count) of the window just collected; ap_vld must be set."""
        if not self.has_vote:
            raise RuntimeError("overlay has no temporal vote registers")
        for off in (self.vote_class_vld_offset, self.vote_count_vld_offset):
            if off is not None and not (int(self.ip.read(off)) & 0x1):
                raise RuntimeError("vote ap_vld not set after S2MM completion")
        return (int(self.ip.read(self.vote_class_offset)) & 0xFFFFFFFF,
                int(self.ip.read(self.vote_count_offset)) & 0xFFFFFFFF)

    @property
    def has_integrity(self) -> bool:
        return getattr(self, "integrity_offset", None) is not None

    def read_integrity(self) -> int:
        """integrity_status of the window just collected (research overlay); ap_vld must be set."""
        if not self.has_integrity:
            raise RuntimeError("overlay has no integrity_status register")
        if self.integrity_vld_offset is not None and not (int(self.ip.read(self.integrity_vld_offset)) & 0x1):
            raise RuntimeError("integrity_status ap_vld not set after S2MM completion")
        return int(self.ip.read(self.integrity_offset)) & 0xFFFFFFFF

    def read_hw_class(self) -> int:
        """Class id computed by the FPGA logic for the window just collected.

        Call only after `infer_codes` returned (S2MM done). If the HWH exposes
        the ap_vld control register, a missing valid bit is an error rather
        than a silently stale value.
        """
        if self.class_reg_offset is None:
            raise RuntimeError("overlay has no pred_class register (legacy IP)")
        if self.class_vld_offset is not None:
            if not (int(self.ip.read(self.class_vld_offset)) & 0x1):
                raise RuntimeError("pred_class ap_vld not set after S2MM completion")
        return int(self.ip.read(self.class_reg_offset)) & 0xFFFFFFFF

    def _wait(self, channel) -> None:
        if self.wait_timeout is not None:
            try:
                channel.wait(self.wait_timeout)   # some PYNQ builds accept a timeout
                return
            except TypeError:
                pass
        channel.wait()

    def infer_codes(self, in_codes: np.ndarray) -> np.ndarray:
        """INT10 code window (1024, 8) -> 4 INT logits.

        MM2S words are row-major [sample, channel] (word = sample*8 + channel).
        The S2MM receive channel is armed BEFORE the MM2S send to avoid output
        backpressure if the core produces logits promptly.
        """
        self._in[:] = np.asarray(in_codes, dtype=np.int32).reshape(SEQ_IN, CH).ravel()
        self.dma.recvchannel.transfer(self._out)   # arm S2MM first
        self.dma.sendchannel.transfer(self._in)    # then MM2S
        self._wait(self.dma.sendchannel)
        self._wait(self.dma.recvchannel)
        return np.array(self._out, dtype=np.int64)

    def close(self) -> None:
        """Release the contiguous DMA buffers (Xlnk 2.4 or allocate 2.5+).

        Both APIs expose a release method -- `pynq.allocate` buffers use
        `freebuffer()`, Xlnk `cma_array` buffers use `freebuffer()`/`close()`.
        Guarded so repeated/partial cleanup never raises.
        """
        for name in ("_in", "_out"):
            buf = getattr(self, name, None)
            if buf is None:
                continue
            for meth in ("freebuffer", "close"):
                fn = getattr(buf, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
            setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Front end
# ---------------------------------------------------------------------------
class FaultClassifier:
    def __init__(self, backend=None, bitstream: Optional[str] = None):
        self.input_scale = load_input_scale()
        if backend is not None:
            self.backend = backend
        elif bitstream is not None:
            self.backend = HardwareBackend(bitstream)
        else:
            self.backend = NumpyBackend()   # repository-only reference

    @property
    def class_source(self) -> str:
        """Where the returned class id is decided.

        "fpga_register"    argmax-variant IP: class computed in FPGA logic
        "ps_argmax_legacy" deployed IP: np.argmax of the 4 FPGA logits on the PS
        "numpy_reference"  repository-only software reference backend
        """
        if getattr(self.backend, "has_hw_class", False):
            return "fpga_register"
        if isinstance(self.backend, HardwareBackend):
            return "ps_argmax_legacy"
        return "numpy_reference"

    def _decide(self, logits: np.ndarray) -> int:
        ref = int(np.argmax(logits))       # lowest index wins ties
        if getattr(self.backend, "has_hw_class", False):
            hw = self.backend.read_hw_class()
            if hw != ref:
                raise RuntimeError(
                    "FPGA pred_class=%d disagrees with argmax of FPGA logits %s (=%d)"
                    % (hw, logits.tolist(), ref))
            return hw
        return ref

    def infer(self, window: np.ndarray):
        """Normalized float window (1024, 8) -> (class_id, logits[4])."""
        codes = quantize_window(window, self.input_scale)
        logits = np.asarray(self.backend.infer_codes(codes), dtype=np.int64)
        return self._decide(logits), logits

    def infer_from_codes(self, in_codes: np.ndarray):
        """Pre-quantized INT10 window (1024, 8) -> (class_id, logits[4])."""
        logits = np.asarray(self.backend.infer_codes(in_codes), dtype=np.int64)
        return self._decide(logits), logits


# ===========================================================================
# REPOSITORY-ONLY NumPy reference backend.
#
# Everything below this marker requires the repository (torch + the trained
# checkpoint + the exported INT10 scales). It is NEVER touched by the hardware
# path above -- torch/src are imported lazily inside __init__, so importing this
# module or constructing a HardwareBackend does not import torch. In the
# deployment bundle (no checkpoint/src) constructing NumpyBackend raises a clear,
# actionable error.
# ===========================================================================
def _scales_path() -> Path:
    return _resolve(
        _DRIVER_DIR.parent / "artifacts" / "quantized" / "f_seq256_int10" / "integer_config.json",  # repo
        _DRIVER_DIR.parent / "config" / "integer_config_scales.json",                                # bundle (no ckpt)
    )


def _checkpoint_path() -> Path:
    return (_DRIVER_DIR.parent / "artifacts" / "experiments"
            / "transformer_f1_seq256_pool4" / "kaggle_v3" / "seq256_pool4"
            / "scratch" / "best.pt")


class NumpyBackend:
    """Bit-exact NumPy stand-in for the hardware (repository-only).

    Reproduces the hardware exactly: aggregate_round_div(factor=4) +
    IntegerTransformer at INT10. Matches the 16/16 CSim golden vectors and the
    full-test result. Requires torch + the checkpoint + `src` (NOT the dataset).
    """

    def __init__(self):
        checkpoint = _checkpoint_path()
        if not checkpoint.exists():
            raise FileNotFoundError(
                "NumpyBackend is repository-only and needs the trained checkpoint "
                f"at {checkpoint}. It is unavailable in the deployment bundle; use "
                "HardwareBackend (real board) there.")
        try:
            import torch
            import src.quantize.integer_model as im  # noqa: F401
            from src.quantize.integer_model import IntegerTransformer
            from src.quantize.evaluate_integer_bits import set_bits
            from src.data.token_aggregation import aggregate_round_div
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "NumpyBackend needs the repository environment (torch + src.*). "
                "It is not available from the standalone deployment bundle.") from exc

        self._aggregate = aggregate_round_div
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_np = {k: v.cpu().numpy() for k, v in ckpt["model_state_dict"].items()}
        set_bits(10)                       # weights quantize at INT10
        model = IntegerTransformer(state_np, MODEL_CFG)
        scales = json.loads(_scales_path().read_text())["scales"]
        model.set_scales(scales)
        self._model = model
        self._input_scale = float(scales["input"])

    def infer_codes(self, in_codes: np.ndarray) -> np.ndarray:
        """INT10 code window (1024, 8) -> 4 INT logits (matches the FPGA)."""
        codes = np.asarray(in_codes, dtype=np.int64).reshape(SEQ_IN, CH)
        codes256 = self._aggregate(codes, factor=AGG_FACTOR)             # [256, 8]
        real = codes256.astype(np.float64) * self._input_scale
        logits = self._model.infer_int(real[None])                      # [1, 4]
        return np.asarray(logits[0], dtype=np.int64)
