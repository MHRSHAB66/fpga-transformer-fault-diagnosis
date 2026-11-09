"""
Deterministically (re)build the standalone PYNQ deployment bundle for the SEQ=256
INT10 Transformer system:

    artifacts/deploy/seq256_z7020_pynq/

It sources every file from canonical, tracked locations (never from the ephemeral
`/tmp/` Vivado project), renames the HWH to match the BIT basename (so PYNQ can
discover overlay metadata), copies the driver/config/golden files, writes a
hwh_contract.json recorded from the generated HWH, and regenerates MANIFEST.sha256.

It never deletes unrelated files and fails loudly if any expected source is
missing.

Run:
    python tools/package_seq256_pynq_deploy.py
    python tools/package_seq256_pynq_deploy.py --check   # verify only, no writes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKG = REPO_ROOT / "artifacts" / "results" / "seq256_dma_system_dsp_margin_z7020"
DEFAULT_BUNDLE = REPO_ROOT / "artifacts" / "deploy" / "seq256_z7020_pynq"

# BIT/HWH basenames MUST match for PYNQ metadata discovery.
BIT_DEST = "overlay/design_1_wrapper.bit"
HWH_DEST = "overlay/design_1_wrapper.hwh"

INPUT_WORDS = 8192          # one MM2S frame = 8192 x 32-bit = 32768 bytes
INPUT_FRAME_BYTES = INPUT_WORDS * 4

# Populated by _configure() from the chosen build package + bundle directory, so
# a new hardware build can be packaged into a NEW versioned bundle without
# editing code or hand-patching artifacts, e.g.:
#   python tools/package_seq256_pynq_deploy.py \
#       --pkg artifacts/results/<new_build> \
#       --bundle artifacts/deploy/seq256_z7020_pynq_dma32k_py36_<timestamp>
PKG = BUNDLE = HWH_SOURCE = None
OVERLAY = DRIVER = CONFIG = GOLDEN = ALL_COPIES = None


def _configure(pkg, bundle, golden=None) -> None:
    """Set the source->dest copy spec for a given build package + bundle dir.

    Overlay binaries come from the (tracked, stable) build package `pkg`, NOT a
    `/tmp/` project. Driver/config/golden come from their canonical repo paths,
    so source fixes propagate into any newly built bundle automatically.
    """
    global PKG, BUNDLE, HWH_SOURCE, OVERLAY, DRIVER, CONFIG, GOLDEN, ALL_COPIES
    # Resolve CLI-supplied relative paths against the invocation working
    # directory.  _hwh_contract records paths relative to REPO_ROOT, so mixing
    # an absolute repository root with an unresolved relative --pkg path would
    # make Path.relative_to() fail even when the package is inside the repo.
    PKG, BUNDLE = Path(pkg).resolve(), Path(bundle).resolve()
    HWH_SOURCE = PKG / "bitstream" / "design_1.hwh"   # original same-build HWH
    OVERLAY = [
        (PKG / "bitstream" / "design_1_wrapper.bit", "overlay/design_1_wrapper.bit"),
        (PKG / "bitstream" / "design_1.hwh",         "overlay/design_1_wrapper.hwh"),  # renamed
        (PKG / "bitstream" / "seq256_dma_system.xsa", "overlay/seq256_dma_system.xsa"),
    ]
    DRIVER = [
        (REPO_ROOT / "pynq" / "fault_classifier.py", "driver/fault_classifier.py"),
        (REPO_ROOT / "pynq" / "run_demo.py",         "driver/run_demo.py"),
        (REPO_ROOT / "host" / "frame_protocol.py",   "driver/frame_protocol.py"),
    ]
    CONFIG = [
        (REPO_ROOT / "hls" / "f1_int10_seq256" / "model_manifest.json", "config/model_manifest.json"),
        (REPO_ROOT / "artifacts" / "quantized" / "f_seq256_int10" / "integer_config.json",
         "config/integer_config_scales.json"),
    ]
    golden_dir = (Path(golden).resolve() if golden is not None
                  else REPO_ROOT / "hls" / "f1_int10_seq256" / "golden_ingress1024")
    GOLDEN = [
        (golden_dir / "golden_in.txt", "golden/golden_in.txt"),
        (golden_dir / "golden_expected.txt", "golden/golden_expected.txt"),
    ]
    ALL_COPIES = OVERLAY + DRIVER + CONFIG + GOLDEN


_configure(DEFAULT_PKG, DEFAULT_BUNDLE)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dma_length_width(hwh_text: str):
    """Parse the AXI-DMA buffer-length register width (C_SG_LENGTH_WIDTH) from
    the HWH. Max single simple-mode transfer is 2**width bytes."""
    m = re.search(r'C_SG_LENGTH_WIDTH"\s*VALUE="(\d+)"', hwh_text)
    return int(m.group(1)) if m else None


def _hwh_contract() -> dict:
    """Record the transport contract read from the generated HWH (not comments)."""
    hwh_text = HWH_SOURCE.read_text(errors="ignore")
    width = _dma_length_width(hwh_text)
    max_xfer = (1 << width) if width is not None else None
    return {
        "source_hwh": str(HWH_SOURCE.relative_to(REPO_ROOT)),
        "source_hwh_sha256": _sha256(HWH_SOURCE),
        "dma_instance": "axi_dma_0",
        "dma_axilite_base": "0x41E00000",
        "transformer_ip_instance": "transformer_top_0",
        "transformer_ctrl_base": "0x40000000",
        "mm2s_tdata_bits": 32,
        "s2mm_tdata_bits": 32,
        "input_words": INPUT_WORDS,
        "output_words": 4,
        # DMA data-path capability, read from the HWH. One input frame is a
        # SINGLE 32768-byte MM2S transfer (TLAST on word 8191); the simple-mode
        # DMA must accept it in one go (needs length width >= 16).
        "dma_sg_length_width": width,
        "dma_max_transfer_bytes": max_xfer,
        "input_frame_bytes": INPUT_FRAME_BYTES,
        "dma_single_frame_ok": (max_xfer is not None and max_xfer >= INPUT_FRAME_BYTES),
        "ap_ctrl": "ap_ctrl_hs; ctrl s_axilite bundle; ap_start=reg0 bit0, auto_restart=reg0 bit7",
        "verified_present": {
            "axi_dma_0": "axi_dma_0" in hwh_text,
            "transformer_top_0": "transformer_top_0" in hwh_text,
            "ctrl_base_0x40000000": "0x40000000" in hwh_text,
            "dma_base_0x41E00000": "0x41E00000" in hwh_text,
            "mm2s_width_32": 'C_M_AXIS_MM2S_TDATA_WIDTH" VALUE="32"' in hwh_text,
            "s2mm_width_32": 'C_S_AXIS_S2MM_TDATA_WIDTH" VALUE="32"' in hwh_text,
            "dma_fits_32768B_single_transfer": (
                max_xfer is not None and max_xfer >= INPUT_FRAME_BYTES),
        },
    }


def _check_sources() -> None:
    missing = [str(src) for src, _ in ALL_COPIES if not src.exists()]
    if missing:
        raise SystemExit("missing source file(s):\n  " + "\n  ".join(missing))
    if Path(BIT_DEST).stem != Path(HWH_DEST).stem:
        raise SystemExit("BIT and HWH destination basenames differ")


def build(check_only: bool = False) -> int:
    _check_sources()

    if check_only:
        problems = []
        for _, dest in ALL_COPIES:
            if not (BUNDLE / dest).exists():
                problems.append(f"missing in bundle: {dest}")
        # BIT/HWH basename match
        if (BUNDLE / BIT_DEST).exists() and not (BUNDLE / HWH_DEST).exists():
            problems.append("HWH basename does not match BIT basename")
        # HWH hash identity
        if (BUNDLE / HWH_DEST).exists():
            if _sha256(BUNDLE / HWH_DEST) != _sha256(HWH_SOURCE):
                problems.append("bundle HWH hash != original same-build HWH hash")
        if problems:
            print("CHECK FAILED:\n  " + "\n  ".join(problems))
            return 1
        print("CHECK OK: bundle matches canonical sources")
        return 0

    for src, dest in ALL_COPIES:
        out = BUNDLE / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)

    # Verify the renamed HWH is byte-identical to the original same-build HWH.
    assert _sha256(BUNDLE / HWH_DEST) == _sha256(HWH_SOURCE), \
        "renamed HWH hash must equal the original same-build HWH"

    (BUNDLE / "config" / "hwh_contract.json").write_text(
        json.dumps(_hwh_contract(), indent=2) + "\n", encoding="utf-8", newline="\n")

    _write_readme()
    _write_manifest()
    print(f"bundle rebuilt at {BUNDLE.relative_to(REPO_ROOT)}")
    print(f"  BIT  sha256 {_sha256(BUNDLE / BIT_DEST)}")
    print(f"  HWH  sha256 {_sha256(BUNDLE / HWH_DEST)}  (== original same-build HWH)")
    print(f"  XSA  sha256 {_sha256(BUNDLE / 'overlay' / 'seq256_dma_system.xsa')}")
    return 0


def _write_manifest() -> None:
    entries = []
    for path in sorted(BUNDLE.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rel = path.relative_to(BUNDLE).as_posix()
            entries.append(f"{_sha256(path)} *./{rel}")
    (BUNDLE / "MANIFEST.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8", newline="\n")


def _write_readme() -> None:
    (BUNDLE / "README.md").write_text(README, encoding="utf-8", newline="\n")


README = """\
# Deployment bundle -- SEQ=256 INT10 Transformer on MYIR Z-Turn XC7Z020

**Deployment package complete and statically/mock validated; physical validation
of this rebuilt bitstream is pending.** The selected board is the MYIR Z-Turn
`MYS-7Z020-C-S`, populated with `XC7Z020-1CLG400C`; the Vivado target part is
`xc7z020clg400-1`. Do not treat packaging or mock validation as physical-board
execution.

This bundle is standalone for the HARDWARE path: it needs only numpy + the vendor
`pynq` runtime. It does NOT need torch, a checkpoint, the dataset, or the project
repository. (This bundle is regenerated by `tools/package_seq256_pynq_deploy.py`.)

## Contents

| Path | What |
|---|---|
| `overlay/design_1_wrapper.bit` | Bitstream (PS + AXI-DMA + transformer_top) |
| `overlay/design_1_wrapper.hwh` | Hardware handoff -- **same basename as .bit** (PYNQ discovers it by basename); byte-identical to the same-build HWH |
| `overlay/seq256_dma_system.xsa` | Exported hardware platform |
| `driver/fault_classifier.py` | Standalone driver (HardwareBackend; self-locating config) |
| `driver/run_demo.py` | Demo launcher (classify -> UART frame) |
| `driver/frame_protocol.py` | FPGA->PC text frame protocol |
| `config/model_manifest.json` | Model + quantization manifest (`input_scale`) |
| `config/integer_config_scales.json` | Calibrated INT10 scales (NumPy reference only) |
| `config/hwh_contract.json` | Transport contract recorded from the generated HWH |
| `golden/golden_in.txt` | 16 approved input windows (each 1024x8 INT10) |
| `golden/golden_expected.txt` | Expected 4 logits + class per window |
| `MANIFEST.sha256` | SHA-256 of every file in this bundle |

The `driver/` files are the canonical repository files copied verbatim; they
resolve their resources relative to their own location, so they work here and in
the repo. The NumPy reference backend inside `fault_classifier.py` is
repository-only and is never imported by the hardware path.

## Contract (verified from `overlay/design_1_wrapper.hwh`)

- Class map (fixed): `0 normal, 1 imbalance, 2 misalignment, 3 bearing_fault`.
- Input: normalized `[1024, 8]` window -> INT10 codes
  `clip(round(x / input_scale), -511, +511)`, `input_scale` from
  `config/model_manifest.json`.
- DMA MM2S (32-bit): 8192 codes, row-major `[sample, channel]`
  (word = `sample*8 + channel`), TLAST on word 8191. This is ONE 32768-byte
  simple-mode transfer, so the AXI-DMA buffer-length register must be wide enough
  (>= 16 bits): see `dma_sg_length_width` / `dma_single_frame_ok` in
  `config/hwh_contract.json` (must be `true`). **The IP aggregates 1024->256
  internally.** The input is never split into multiple transfers (that would move
  TLAST off word 8191 and violate the frame contract).
- Driver Python compatibility: 3.6+ (the board image is Python 3.6.5). DMA
  buffers are allocated via `pynq.allocate` (>=2.5) or `Xlnk().cma_array` (2.4).
- DMA S2MM (32-bit): 4 INT32 logits, TLAST on logit 3; class = argmax.
- DMA instance `axi_dma_0` (0x41E00000); IP `transformer_top_0` (ctrl 0x40000000,
  `ap_ctrl_hs`, reg 0x00 bit0 ap_start / bit7 auto_restart; the driver uses
  auto_restart so the core re-arms per window).
- Confidence: **NA** (the integer core emits no confidence).

## Board-side commands (Zynq / PYNQ), once validated

Run the demo as a SCRIPT from inside `driver/` so `from pynq import Overlay`
resolves to the vendor runtime (not a project folder). Do NOT use
`python -m pynq.run_demo` on the board.

```bash
cd artifacts/deploy/seq256_z7020_pynq
python driver/run_demo.py --bitstream overlay/design_1_wrapper.bit --port /dev/ttyPS0
```

Golden sanity check on the board (expect 16/16 exact, matching CSim):

```python
import numpy as np
from fault_classifier import FaultClassifier          # run from driver/
clf = FaultClassifier(bitstream="../overlay/design_1_wrapper.bit")
lines = open("../golden/golden_in.txt").read().split("\\n")
n, seq, ch = map(int, lines[0].split())
for i in range(n):
    w = np.array(lines[1+i].split(), dtype=np.int64).reshape(seq, ch)
    print(clf.infer_from_codes(w))   # compare to ../golden/golden_expected.txt
```

## PC-side command

```bash
python -m host.fault_monitor --source serial --port <PC-COM> --baud 115200
```

## Remaining before this is "validated"

1. Program the board with `overlay/design_1_wrapper.bit`.
2. Confirm `axi_dma_0` / `transformer_top_0` via `overlay.ip_dict`.
3. Run the golden sanity check above (expect 16/16 exact).
4. Confirm the UART link and the PC monitor display end to end.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    ap.add_argument("--pkg", default=None,
                    help="build package dir containing bitstream/ (default: the "
                         "dsp_margin z7020 build). Point this at a new Vivado build "
                         "to package the DMA-fixed bitstream.")
    ap.add_argument("--bundle", default=None,
                    help="output bundle dir (default: artifacts/deploy/seq256_z7020_pynq). "
                         "Use a NEW versioned path to avoid overwriting the old bundle.")
    ap.add_argument("--golden", default=None,
                    help="golden vector dir copied into golden/ (default: "
                         "hls/f1_int10_seq256/golden_ingress1024); e.g. the raw-input set "
                         "hls/f1_int10_seq256/golden_raw1024_argmax")
    args = ap.parse_args()
    if args.pkg is not None or args.bundle is not None or args.golden is not None:
        _configure(args.pkg or DEFAULT_PKG, args.bundle or DEFAULT_BUNDLE, args.golden)
    sys.exit(build(check_only=args.check))
