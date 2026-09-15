# PYNQ-side driver + demo (Stage 8/9/10 board glue)

Drives the `F_1layer @ int10` core on a PYNQ/Zynq board and sends each
classification to the PC over UART. Everything here also runs **without a board**
(NumPy backend), so the full data path is demonstrable today; swapping in the
synthesized overlay is the only change needed once the bitstream exists.

## Files
- `fault_classifier.py` — `FaultClassifier` with two backends:
  - `NumpyBackend` — runs the verified `hls/f1_int10/` int10 reference in NumPy
    (no board, no torch). Bit-exact with the hardware contract.
  - `HardwareBackend` — loads the overlay (`.bit`/`.hwh`) and streams a window
    through an `axi_dma` to the HLS IP. Fill in the IP/DMA names after synthesis.
- `run_demo.py` — window → int10 classify → UART frame, end to end.

## Class map
Fixed in `host/frame_protocol.LABELS` (== `src.config.LABEL_MAP`):
`0 normal, 1 imbalance, 2 misalignment, 3 bearing_fault`. The board never
re-defines it.

## Data contract for the HLS IP (matches `hls/f1_int10/transformer.h`)
- **input**: `SEQ*CH = 1024*8` int codes, row-major `[seq, channel]`, quantized
  as `code = round(x_normalized / input_scale)` clamped to ±511, where
  `input_scale` comes from `hls/f1_int10/model_manifest.json`.
- **output**: 4 int logits; prediction = `argmax`.
- **transport** (assumed): AXI-Stream in/out over one `axi_dma`, `s_axilite`
  start/done. Adjust `HardwareBackend` to your generated design.

## Run without a board (numpy inference + printed frames)
```bash
python -m pynq.run_demo --print
```

## Run inference in numpy but send real UART to the PC
```bash
python -m pynq.run_demo --port COM6            # Windows;  /dev/ttyUSB0 on Linux
```

## Run on the board (hardware inference + UART)
```bash
python -m pynq.run_demo --bitstream design.bit --port /dev/ttyPS0
```

Pair any of the above with the PC monitor:
```bash
python -m host.fault_monitor --source serial --port <PC-side-COM>
```

## Bring-up checklist once you have the bitstream
1. Copy `design.bit` + `design.hwh` to the board.
2. In `HardwareBackend`, set the overlay path and confirm the DMA name
   (`overlay.ip_dict` / `overlay?`).
3. Sanity-check against the golden vectors: feed `hls/f1_int10/golden/golden_in.txt`
   rows and confirm the logits match `golden_expected.txt` (same check the
   NumPy backend passes 16/16).
4. Then run `run_demo.py --bitstream ... --port ...`.
