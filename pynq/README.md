# PYNQ-side driver + demo (Stage 8/10 board glue)

Drives the **SEQ=256 INT10** Transformer core, integrated as a Zynq PS + AXI-DMA
+ `transformer_top` HLS IP system (provisional/generic XC7Z020), and sends each
classification to the PC over UART. Everything here also runs **without a board**
(NumPy backend), so the full data path is demonstrable today; swapping in the
synthesized overlay is the only change needed on the board.

> The bitstream/overlay is **deployment-ready but pending physical-board
> validation** — no board has been programmed or tested yet.

## Files
- `fault_classifier.py` — `FaultClassifier` with two backends:
  - `NumpyBackend` — runs the verified INT10 reference (4:1 `aggregate_round_div`
    + `IntegerTransformer`) in Python. **Bit-exact** with the hardware: matches
    the 16/16 CSim golden vectors and the full-test result. Needs torch + the
    checkpoint + the exported scales (not the dataset).
  - `HardwareBackend` — loads the overlay (`.bit`/`.hwh`) and streams one 1024×8
    window through `axi_dma_0` to `transformer_top_0`. Names read from the
    generated `design_1.hwh`.
- `run_demo.py` — window → INT10 classify → UART frame, end to end.

## Class map
Fixed in `host/frame_protocol.LABELS` (== `src.config.LABEL_MAP`):
`0 normal, 1 imbalance, 2 misalignment, 3 bearing_fault`. The board never
re-defines it.

## Data contract for the integrated IP (from `design_1.hwh`)
- **acquisition window**: normalized `[1024, 8]` (50 kHz, 20.48 ms span).
- **quantization**: `code = clip(round(x_normalized / input_scale), -511, +511)`,
  signed INT10, `input_scale = 0.028626115354773118`
  (`hls/f1_int10_seq256/model_manifest.json`).
- **DMA ingress (MM2S, 32-bit)**: all `1024×8 = 8192` INT codes, **row-major
  `[sample, channel]`** (word = `sample*8 + channel`), one code per 32-bit word,
  `TLAST` on word 8191. **The IP performs the 4:1 integer aggregation
  (1024 → 256 tokens) internally** — the host does not aggregate.
- **DMA egress (S2MM, 32-bit)**: 4 INT32 logits, `TLAST` on logit 3.
- **prediction**: deployed dma16 bitstream — the core streams only the 4 logits;
  the class is `np.argmax` of those FPGA logits on the Zynq PS (ARM), lowest
  class id wins ties. Argmax-variant IP (`transformer_top_seq256_ingress1024_dsp_margin_argmax.cpp`,
  C-model verified, **not yet synthesized or run on the board**) additionally
  computes the class in FPGA logic and exposes it as AXI-Lite register
  `pred_class`; `HardwareBackend` reads it when the HWH lists that register
  and checks it against the logits (`FaultClassifier.class_source`).
- **control**: `transformer_top` `ap_ctrl_hs`, ctrl AXI-Lite at `0x40000000`,
  register `0x00` (bit0 ap_start, bit7 auto_restart). DMA AXI-Lite at `0x41E00000`.
- **confidence**: the integer core emits **no** confidence → reported as `NA`.
  `run_demo.py --sw-conf` adds a *software-only* softmax convenience (not a
  hardware signal).

> **Vendor package-name clash.** This project folder is named `pynq`, which
> collides with the installed vendor `pynq` runtime. On the **board**, run the
> demo as a *script from the deployment bundle's `driver/` folder* so
> `from pynq import Overlay, allocate` resolves to the vendor library, not this
> folder. Do **not** use `python -m pynq.run_demo` on the board. The
> `python -m pynq.run_demo` form is only for repository development/tests on a
> machine without the vendor package.

## Run without a board (numpy inference + printed frames), repo dev
```bash
python -m pynq.run_demo --print
```

## Run inference in numpy but send real UART to the PC, repo dev
```bash
python -m pynq.run_demo --port COM6            # Windows;  /dev/ttyUSB0 on Linux
```

## Run on the board (hardware inference + UART) — from the deployment bundle
```bash
cd artifacts/deploy/seq256_z7020_pynq
python driver/run_demo.py --bitstream overlay/design_1_wrapper.bit --port /dev/ttyPS0
```

Pair any of the above with the PC monitor:
```bash
python -m host.fault_monitor --source serial --port <PC-side-COM>
```

## Bring-up checklist once on the board (PENDING — not yet done)
1. Copy the whole `artifacts/deploy/seq256_z7020_pynq/` bundle to the board. The
   overlay is `overlay/design_1_wrapper.bit` + `overlay/design_1_wrapper.hwh`
   (matching basenames, same build).
2. `HardwareBackend` already targets `axi_dma_0` / `transformer_top_0`; confirm
   with `overlay.ip_dict`.
3. Sanity-check against the golden vectors: feed the rows of
   `hls/f1_int10_seq256/golden_ingress1024/golden_in.txt` and confirm the logits
   match `golden_expected.txt` (the same 16/16 check the NumPy backend passes and
   the CSim passed).
4. Then run `run_demo.py --bitstream ... --port ...`.
