# Vitis HLS synthesis guide — F_1layer @ int10

Everything here runs on the laptop with Vitis; this desktop has no compiler, so
the C++ below is unverified-by-compiler (the *arithmetic* is verified bit-exact
by `../verify_export.py`, PASS 16/16). Follow the stages in order.

Target part: **TBD**. The final FPGA board/device has not been selected. Before
running the flow, set `FPGA_PART` to the exact Vitis part and record that
device's DSP, BRAM, LUT and FF budget in the project report.

## Files here
| file | role |
|------|------|
| `run_hls.tcl` | project script (C-sim now; csynth/cosim/export gated by flags) |
| `weights_int10.h` | all int10 constants as flat `static const` arrays (regen: `python gen_weights_header.py`) |
| `ap_types.h` | `act_t/acc_t/lut_t` typedefs, `ap_int` under synthesis / wide int under g++ |
| `transformer_top.cpp` | synthesizable top: AXI-Stream in/out, LAYERS=1 inlined, params from the header |

## Stage A — C-simulation now (validates in the real tool, no synthesis)
```bash
cd hls/f1_int10/vitis
vitis_hls -f run_hls.tcl        # DO_CSIM=1: builds ../transformer.cpp + testbench
```
Expect `RESULT: PASS` (16/16 class, 16/16 bit-exact logits) — same as
`python ../verify_export.py`. This confirms the toolchain + golden vectors.

## Stage B — baseline C-synthesis (get the first resource/latency report)
In `run_hls.tcl` set `DO_CSYNTH 1`, `set_top transformer_top`, and add
`$HLSDIR/vitis/transformer_top.cpp` (drop the portable `transformer.cpp` from the
design fileset — keep it only as -tb). Re-run. This gives a BASELINE report; it
does not establish fit until it is compared with the selected device (see Stage C).

Why it won't fit as-is: the core keeps full `SEQ*D = 32768`-element activation
buffers (`x, q, k, v, ctx, res, ln1, ff2`) plus `SEQ*DFF = 65536` for `ff`.
Attention is O(SEQ²)=~4M MACs/head. Measure this baseline against the selected
device before deciding which optimizations are mandatory.

## Stage C — optimize to fit the board
1. **Narrow the datapath.** `ap_types.h` already sets `act_t=ap_int<10>`,
   `acc_t=ap_int<32>`. Change the buffers/params in `transformer_top.cpp` from
   `long` to `act_t`/`acc_t` (proven safe: max dot product ≈ 64·2¹⁸ < 2³²).
2. **Kill floating point.** Convert each `double` requant/affine constant to an
   integer multiplier `M0 · 2^-shift` (or `ap_fixed<18,2>`); regenerate them in
   `gen_weights_header.py`. Removes FP DSPs/LUTs.
3. **Don't materialize the whole sequence.** Stream the token dimension:
   - attention: process one query row `i` at a time; keep only K,V for the head
     in BRAM (SEQ·HD·10b ≈ 10 KB) and stream Q; recompute per head. This trades
     latency for BRAM and is the key fit lever.
   - fuse proj→+PE→attention→residual→LN→FFN→LN as a `#pragma HLS dataflow`
     pipeline of per-token (or per-tile) stages instead of full-array buffers.
4. **Pipeline the MAC loops.** `#pragma HLS PIPELINE` on the inner `Din`/`HD`
   loops; `#pragma HLS ARRAY_PARTITION` on the small dims (`D=32`, `HD=8`,
   `DFF=64`) for parallel MACs; bound DSP use to ≤ 80.
5. **Re-run csynth**, read II/latency + LUT/FF/BRAM/DSP, iterate until it fits.
   F_1layer was chosen precisely to halve the encoder HW vs the 2-layer baseline;
   if attention still overflows BRAM, the documented fallbacks are a shorter
   window (512) or windowed/strided attention — both require a re-export, not a
   silent change.

## Stage D — export + integrate
1. `DO_EXPORT 1` → `export_design -format ip_catalog` (Verilog RTL IP).
2. Vivado block design: Zynq7 PS + AXI DMA + this IP (AXIS in/out) + AXI-Lite
   control; generate bitstream → `design.bit` + `design.hwh`.
3. Copy both to the board; in `pynq/fault_classifier.py` `HardwareBackend`, set
   the overlay path and confirm the DMA name (`overlay.ip_dict`).
4. Sanity-check on hardware with `hls/f1_int10/golden/` (logits must match), then
   `python -m pynq.run_demo --bitstream design.bit --port /dev/ttyPS0` with the
   PC running `python -m host.fault_monitor --source serial --port <COM>`.

## Report artifacts to capture (for the thesis, Stage 12)
From the Vitis reports: **LUT, FF, BRAM, DSP, latency, II, Fmax**; from the board:
UART throughput/stability. Save alongside the software metrics (float 97.55% /
int10 97.11%).
